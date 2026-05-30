#!/usr/bin/env python3
"""
PhlexSensor 触觉手套 Linux 数据采集工具
=========================================

功能：
  - 通过串口读取手套 48 通道传感器数据
  - 支持加载标定文件，输出校准后压力值
  - 实时终端可视化（热力图 + 数值表）
  - 数据保存为 CSV 文件

通信协议：
  - UART 115200bps, 8N1
  - 上位机发送 "A\\r\\n" → 下位机回复 96 字节
  - 48 通道 × 2 字节/通道，小端序
  - 通道顺序：大拇指(8) → 食指(8) → 中指(8) → 无名指(8) → 小指(8) → 手掌(8)

标定公式：
  f(x) = b * c * x / (c - a * x)
  其中 x 为 ADC 值，a/b/c 为拟合系数

依赖安装：
  pip install pyserial

可选依赖（实时绘图）：
  pip install matplotlib numpy

用法：
  # 列出可用串口
  python phlex_reader.py --list-ports

  # 基础数据采集（打印到终端）
  python phlex_reader.py --port /dev/ttyUSB0

  # 加载标定文件 + 保存 CSV
  python phlex_reader.py --port /dev/ttyUSB0 --calibration cal.txt --save

  # 实时 matplotlib 可视化
  python phlex_reader.py --port /dev/ttyUSB0 --plot

  # 单次采集（调试用）
  python phlex_reader.py --port /dev/ttyUSB0 --once
"""

from __future__ import annotations

import argparse
import csv
import os
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    import serial
    import serial.tools.list_ports
    HAS_SERIAL = True
except ImportError:
    HAS_SERIAL = False

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import numpy as np
    HAS_PLOT = True
except ImportError:
    HAS_PLOT = False


# ─────────────────────────────────────────────
#  常量定义
# ─────────────────────────────────────────────

BAUD_RATE = 115200
FRAME_SIZE = 96          # 48 通道 × 2 字节
NUM_CHANNELS = 48
CHANNELS_PER_REGION = 8
REQUEST_CMD = b"A\r\n"

REGION_NAMES = ["大拇指", "食指", "中指", "无名指", "小指", "手掌"]
REGION_NAMES_EN = ["Thumb", "Index", "Middle", "Ring", "Pinky", "Palm"]


# ─────────────────────────────────────────────
#  标定模块
# ─────────────────────────────────────────────

class CalibrationData:
    """加载并应用标定文件。

    标定文件格式（每行 7 列，逗号或空格分隔）：
      行号, 列号, a, b, c, rsquare, 有效计数值

    标定公式：
      f(x) = b * c * x / (c - a * x)
      约束：c - a * x > 0，否则输出上限值
      若 x > c，则令 x = c
    """

    def __init__(self, filepath: str):
        self.params = {}  # (row, col) -> (a, b, c, rsquare)
        self._load(filepath)

    def _load(self, filepath: str):
        """解析标定文件。支持逗号、制表符、空格分隔。"""
        with open(filepath, "r", encoding="utf-8-sig") as f:
            for line_no, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # 自动检测分隔符
                for sep in [",", "\t", " "]:
                    parts = [p.strip() for p in line.split(sep) if p.strip()]
                    if len(parts) >= 5:
                        break
                if len(parts) < 5:
                    print(f"[警告] 标定文件第 {line_no} 行格式不符，跳过: {line}")
                    continue
                try:
                    row = int(float(parts[0]))
                    col = int(float(parts[1]))
                    a = float(parts[2])
                    b = float(parts[3])
                    c = float(parts[4])
                    rsquare = float(parts[5]) if len(parts) > 5 else 0.0
                    self.params[(row, col)] = (a, b, c, rsquare)
                except (ValueError, IndexError) as e:
                    print(f"[警告] 标定文件第 {line_no} 行解析失败: {e}")

        print(f"[标定] 已加载 {len(self.params)} 组标定参数")

    def apply(self, channel_index: int, adc_value: int) -> float:
        """对单个通道的 ADC 值进行标定转换。

        channel_index: 0-47 的通道序号
        adc_value: 原始 ADC 值

        返回: 标定后的压力值 (kPa)，若无标定参数则返回原始值
        """
        # 通道映射到 (行, 列)
        region = channel_index // CHANNELS_PER_REGION  # 0-5
        point = channel_index % CHANNELS_PER_REGION    # 0-7

        # 尝试多种 key 格式（标定文件可能从 0 或 1 开始编号）
        for r_offset, c_offset in [(0, 0), (1, 1), (0, 1), (1, 0)]:
            key = (region + r_offset, point + c_offset)
            if key in self.params:
                a, b, c, _ = self.params[key]
                return self._compute(adc_value, a, b, c)

        # 没有找到标定参数，返回原始值
        return float(adc_value)

    @staticmethod
    def _compute(x: float, a: float, b: float, c: float) -> float:
        """f(x) = b * c * x / (c - a * x)"""
        # 约束：x 不超过 c
        if x > c:
            x = c
        denominator = c - a * x
        if denominator <= 0:
            # 分母 <= 0 时返回最大标定压力
            return b * c if b > 0 else 0.0
        return b * c * x / denominator


# ─────────────────────────────────────────────
#  串口通信模块
# ─────────────────────────────────────────────

class PhlexSerial:
    """与 PhlexSensor 手套的串口通信。"""

    def __init__(self, port: str, baudrate: int = BAUD_RATE, timeout: float = 1.0):
        self.port_name = port
        self.ser = serial.Serial(
            port=port,
            baudrate=baudrate,
            bytesize=serial.EIGHTBITS,
            stopbits=serial.STOPBITS_ONE,
            parity=serial.PARITY_NONE,
            timeout=timeout,
        )
        # 清空缓冲区
        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()
        print(f"[串口] 已连接 {port} @ {baudrate}bps")

    def request_frame(self) -> list[int] | None:
        """发送请求并读取一帧数据。

        返回: 48 个整数的列表（ADC 值），或 None（读取失败）
        """
        # 清空接收缓冲区，防止堆积
        self.ser.reset_input_buffer()

        # 发送请求命令
        self.ser.write(REQUEST_CMD)
        self.ser.flush()

        # 读取响应
        data = self.ser.read(FRAME_SIZE)
        if len(data) != FRAME_SIZE:
            return None

        # 解析：48 个 uint16，小端序
        channels = []
        for i in range(NUM_CHANNELS):
            value = struct.unpack_from("<H", data, i * 2)[0]
            channels.append(value)
        return channels

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(f"[串口] 已断开 {self.port_name}")


# ─────────────────────────────────────────────
#  数据保存模块
# ─────────────────────────────────────────────

class DataLogger:
    """将采集数据保存为 CSV 文件。"""

    def __init__(self, output_dir: str = "data"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filepath = self.output_dir / f"phlex_{timestamp}.csv"
        self.file = open(self.filepath, "w", newline="", encoding="utf-8")
        self.writer = csv.writer(self.file)

        # 写表头
        header = ["timestamp"]
        for region in REGION_NAMES_EN:
            for ch in range(1, 9):
                header.append(f"{region}_Ch{ch}")
        header.append("total")
        self.writer.writerow(header)
        self.count = 0
        print(f"[保存] 数据文件: {self.filepath}")

    def log(self, channels: list[float]):
        """记录一帧数据。"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        row = [timestamp] + [f"{v:.2f}" for v in channels] + [f"{sum(channels):.2f}"]
        self.writer.writerow(row)
        self.count += 1
        # 每 100 帧刷新一次磁盘
        if self.count % 100 == 0:
            self.file.flush()

    def close(self):
        self.file.flush()
        self.file.close()
        print(f"[保存] 共保存 {self.count} 帧数据到 {self.filepath}")


# ─────────────────────────────────────────────
#  终端可视化
# ─────────────────────────────────────────────

def print_frame_table(channels: list[float], frame_no: int, calibrated: bool = False):
    """在终端打印当前帧的数据表格。"""
    unit = "kPa" if calibrated else "ADC"
    print(f"\033[2J\033[H")  # 清屏
    print(f"╔══════════════════════════════════════════════════════════════════════╗")
    print(f"║  PhlexSensor Linux Reader  │  帧 #{frame_no:<6d}  │  单位: {unit:<5s}          ║")
    print(f"╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  区域    │ Ch1     Ch2     Ch3     Ch4     Ch5     Ch6     Ch7     Ch8    ║")
    print(f"╠══════════════════════════════════════════════════════════════════════╣")

    for i, name in enumerate(REGION_NAMES):
        start = i * 8
        vals = channels[start:start + 8]
        val_str = " ".join(f"{v:7.1f}" for v in vals)
        print(f"║  {name:<6s} │ {val_str} ║")

    total = sum(channels)
    print(f"╠══════════════════════════════════════════════════════════════════════╣")
    print(f"║  合计: {total:10.1f}                                                     ║")
    print(f"╚══════════════════════════════════════════════════════════════════════╝")
    print(f"  按 Ctrl+C 停止采集")


def print_terminal_heatmap(channels: list[float], max_val: float = 4096):
    """在终端用色块显示简易热力图。"""
    blocks = " ░▒▓█"
    print("\n  热力图 (颜色深度 = 压力大小):")
    for i, name in enumerate(REGION_NAMES):
        start = i * 8
        vals = channels[start:start + 8]
        bar = ""
        for v in vals:
            level = min(int(v / max_val * (len(blocks) - 1)), len(blocks) - 1)
            bar += blocks[level] * 2
        region_avg = sum(vals) / len(vals) if vals else 0
        print(f"  {name:<6s} [{bar}] avg={region_avg:.0f}")


# ─────────────────────────────────────────────
#  Matplotlib 实时绘图
# ─────────────────────────────────────────────

class RealtimePlotter:
    """使用 matplotlib 实时绘制曲线和热力图。"""

    def __init__(self, history_len: int = 300):
        self.history_len = history_len
        self.history = [[] for _ in range(NUM_CHANNELS)]

        plt.ion()
        self.fig, (self.ax_line, self.ax_heat) = plt.subplots(
            1, 2, figsize=(14, 6),
            gridspec_kw={"width_ratios": [2, 1]}
        )
        self.fig.suptitle("PhlexSensor Linux 实时监控", fontsize=14)

        # 折线图（默认显示大拇指 8 通道）
        self.current_region = 0
        self.lines = []
        colors = plt.cm.tab10(np.linspace(0, 1, 8))
        for ch in range(8):
            line, = self.ax_line.plot([], [], color=colors[ch],
                                      label=f"Ch{ch + 1}", linewidth=1)
            self.lines.append(line)
        self.ax_line.set_xlim(0, history_len)
        self.ax_line.set_ylim(0, 4096)
        self.ax_line.set_xlabel("采样点")
        self.ax_line.set_ylabel("ADC / kPa")
        self.ax_line.legend(loc="upper right", fontsize=8, ncol=4)
        self.ax_line.set_title(f"曲线: {REGION_NAMES[self.current_region]}")
        self.ax_line.grid(True, alpha=0.3)

        # 热力图 (6行 × 8列)
        self.heat_data = np.zeros((6, 8))
        self.heat_img = self.ax_heat.imshow(
            self.heat_data, cmap="jet", vmin=0, vmax=4096,
            aspect="auto", interpolation="bilinear"
        )
        self.ax_heat.set_yticks(range(6))
        self.ax_heat.set_yticklabels(REGION_NAMES)
        self.ax_heat.set_xticks(range(8))
        self.ax_heat.set_xticklabels([f"Ch{i+1}" for i in range(8)], fontsize=8)
        self.ax_heat.set_title("压力热力图")
        self.fig.colorbar(self.heat_img, ax=self.ax_heat, fraction=0.046)

        plt.tight_layout()
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(self, channels: list[float]):
        """更新图表。"""
        # 更新历史数据
        for i in range(NUM_CHANNELS):
            self.history[i].append(channels[i])
            if len(self.history[i]) > self.history_len:
                self.history[i].pop(0)

        # 更新折线图（当前区域的 8 个通道）
        base = self.current_region * 8
        for ch in range(8):
            data = self.history[base + ch]
            self.lines[ch].set_data(range(len(data)), data)

        y_max = max(max(self.history[base + ch]) if self.history[base + ch] else 1
                     for ch in range(8))
        self.ax_line.set_ylim(0, max(y_max * 1.1, 100))
        self.ax_line.set_xlim(0, max(len(self.history[base]), self.history_len))

        # 更新热力图
        for r in range(6):
            for c in range(8):
                self.heat_data[r][c] = channels[r * 8 + c]
        self.heat_img.set_data(self.heat_data)
        v_max = max(max(channels), 1)
        self.heat_img.set_clim(0, v_max)

        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def set_region(self, region_idx: int):
        self.current_region = region_idx % 6
        self.ax_line.set_title(f"曲线: {REGION_NAMES[self.current_region]}")


# ─────────────────────────────────────────────
#  模拟模式（无硬件时测试）
# ─────────────────────────────────────────────

class MockSerial:
    """模拟串口，用于无硬件时测试软件功能。"""

    def __init__(self):
        import random
        self.random = random
        self.frame_count = 0
        print("[模拟] 使用模拟数据（未连接真实设备）")

    def request_frame(self) -> list[int]:
        self.frame_count += 1
        channels = []
        for region in range(6):
            for ch in range(8):
                # 模拟不同区域有不同的基础压力
                base = [500, 800, 600, 400, 300, 700][region]
                noise = self.random.gauss(0, 50)
                # 模拟周期性变化
                wave = 200 * abs(
                    __import__("math").sin(self.frame_count * 0.05 + region * 0.5 + ch * 0.3)
                )
                val = max(0, min(4095, int(base + noise + wave)))
                channels.append(val)
        return channels

    def close(self):
        print("[模拟] 模拟设备已关闭")


# ─────────────────────────────────────────────
#  主程序
# ─────────────────────────────────────────────

def list_serial_ports():
    """列出所有可用串口。"""
    if not HAS_SERIAL:
        print("[错误] 请先安装 pyserial: pip install pyserial")
        return
    ports = serial.tools.list_ports.comports()
    if not ports:
        print("未发现可用串口。")
        print("提示：")
        print("  - 确认 USB 线已连接")
        print("  - 运行 dmesg | tail 查看内核日志")
        print("  - 检查是否需要 ch341 驱动（大多数 Linux 内核已内置）")
        print("  - 可能需要权限: sudo usermod -aG dialout $USER")
        return
    print(f"发现 {len(ports)} 个串口:\n")
    for p in ports:
        ch340_flag = " ← 可能是手套" if "CH340" in (p.description or "") or "ch341" in (p.driver or "").lower() else ""
        print(f"  {p.device}")
        print(f"    描述: {p.description}")
        print(f"    硬件ID: {p.hwid}")
        if p.manufacturer:
            print(f"    厂商: {p.manufacturer}")
        if ch340_flag:
            print(f"    {ch340_flag}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="PhlexSensor 触觉手套 Linux 数据采集工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  %(prog)s --list-ports                          # 列出串口
  %(prog)s -p /dev/ttyUSB0                       # 基础采集
  %(prog)s -p /dev/ttyUSB0 -c cal.txt --save     # 标定 + 保存
  %(prog)s -p /dev/ttyUSB0 --plot                 # 实时绘图
  %(prog)s --mock --plot                          # 模拟模式测试
        """,
    )
    parser.add_argument("--list-ports", action="store_true", help="列出可用串口并退出")
    parser.add_argument("-p", "--port", type=str, help="串口设备路径 (如 /dev/ttyUSB0)")
    parser.add_argument("-b", "--baud", type=int, default=BAUD_RATE, help=f"波特率 (默认 {BAUD_RATE})")
    parser.add_argument("-c", "--calibration", type=str, help="标定文件路径")
    parser.add_argument("--save", action="store_true", help="保存数据到 CSV")
    parser.add_argument("--save-dir", type=str, default="data", help="CSV 保存目录 (默认 data/)")
    parser.add_argument("--plot", action="store_true", help="启用 matplotlib 实时绘图")
    parser.add_argument("--once", action="store_true", help="只采集一帧后退出")
    parser.add_argument("--mock", action="store_true", help="模拟模式（无需硬件）")
    parser.add_argument("--rate", type=float, default=0.02, help="采集间隔秒数 (默认 0.02 = 50Hz)")
    parser.add_argument("--region", type=int, default=0, choices=range(6),
                        help="曲线显示区域: 0=大拇指 1=食指 2=中指 3=无名指 4=小指 5=手掌")

    args = parser.parse_args()

    # 列出串口
    if args.list_ports:
        list_serial_ports()
        return

    # 检查依赖
    if not args.mock and not HAS_SERIAL:
        print("[错误] 请安装 pyserial:")
        print("  pip install pyserial")
        sys.exit(1)

    if args.plot and not HAS_PLOT:
        print("[错误] 实时绘图需要 matplotlib 和 numpy:")
        print("  pip install matplotlib numpy")
        sys.exit(1)

    # 必须指定端口或模拟模式
    if not args.port and not args.mock:
        print("[错误] 请指定串口 (-p /dev/ttyUSB0) 或使用模拟模式 (--mock)")
        parser.print_help()
        sys.exit(1)

    # 初始化串口
    if args.mock:
        device = MockSerial()
    else:
        try:
            device = PhlexSerial(args.port, args.baud)
        except serial.SerialException as e:
            print(f"[错误] 无法打开串口 {args.port}: {e}")
            print("提示: 可能需要权限，运行 sudo usermod -aG dialout $USER 后重新登录")
            sys.exit(1)

    # 初始化标定
    calibration = None
    if args.calibration:
        try:
            calibration = CalibrationData(args.calibration)
        except FileNotFoundError:
            print(f"[错误] 标定文件不存在: {args.calibration}")
            device.close()
            sys.exit(1)

    # 初始化保存
    logger = None
    if args.save:
        logger = DataLogger(args.save_dir)

    # 初始化绘图
    plotter = None
    if args.plot:
        plotter = RealtimePlotter()
        plotter.set_region(args.region)

    # 零点数据（自动清零用）
    baseline = None

    print("\n[采集] 开始数据采集... (Ctrl+C 停止)\n")
    frame_no = 0

    try:
        while True:
            raw = device.request_frame()
            if raw is None:
                print("[警告] 读取超时，重试中...")
                time.sleep(0.1)
                continue

            frame_no += 1

            # 应用标定
            if calibration:
                values = [calibration.apply(i, raw[i]) for i in range(NUM_CHANNELS)]
            else:
                values = [float(v) for v in raw]

            # 应用零点偏移
            if baseline:
                values = [max(0.0, values[i] - baseline[i]) for i in range(NUM_CHANNELS)]

            # 保存
            if logger:
                logger.log(values)

            # 显示
            if plotter:
                plotter.update(values)
            else:
                print_frame_table(values, frame_no, calibrated=calibration is not None)
                print_terminal_heatmap(values, max_val=4096 if not calibration else 150)

            # 单次模式
            if args.once:
                break

            time.sleep(args.rate)

    except KeyboardInterrupt:
        print("\n\n[采集] 用户停止采集")

    finally:
        device.close()
        if logger:
            logger.close()
        if plotter:
            plt.ioff()
            plt.close("all")

    print("[完成] 程序退出")


if __name__ == "__main__":
    main()
