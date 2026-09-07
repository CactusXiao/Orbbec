import os
import csv
import time
import json
import shutil
import argparse
import subprocess

import cv2
import numpy as np
import serial


# ============================================================
# 串口协议
# ============================================================

HEADER = b"\xAA\x55\x03\x99"

RIGHT_HAND = 0x02


# ============================================================
# 右手传感器映射
#
# 来自说明书：
#   右手压力点-数组对应关系
#
# 注意：
#   说明书编号是 1-based
#   Python 数组是 0-based
#   所以读取时统一 index - 1
# ============================================================

RH_FINGERS = {

    "thumb": np.array([
        [240, 239, 238],
        [256, 255, 254],
        [16,   15,  14],
        [32,   31,  30],
    ], dtype=np.int32),

    "index": np.array([
        [237, 236, 235],
        [253, 252, 251],
        [13,   12,  11],
        [29,   28,  27],
    ], dtype=np.int32),

    "middle": np.array([
        [234, 233, 232],
        [250, 249, 248],
        [10,    9,   8],
        [26,   25,  24],
    ], dtype=np.int32),

    "ring": np.array([
        [231, 230, 229],
        [247, 246, 245],
        [7,     6,   5],
        [23,   22,  21],
    ], dtype=np.int32),

    "little": np.array([
        [228, 227, 226],
        [244, 243, 242],
        [4,     3,   2],
        [20,   19,  18],
    ], dtype=np.int32),
}


# 手掌 72 个压力点
RH_PALM = np.array([

     61,  60,  59,  58,  57,  56,  55,  54,  53,  52,  51,  50,

     80,  79,  78,  77,  76,  75,  74,  73,  72,  71,  70,  69,

     68,  67,  66,  96,  95,  94,  93,  92,  91,  90,  89,  88,

     87,  86,  85,  84,  83,  82, 112, 111, 110, 109, 108, 107,

    106, 105, 104, 103, 102, 101, 100,  99,  98, 128, 127, 126,

    125, 124, 123, 122, 121, 120, 119, 118, 117, 116, 115, 114,

], dtype=np.int32).reshape(6, 12)


# 弯折数据不是压力，因此不参与压力热图
RH_BENDS = {
    "thumb": 47,
    "index": 44,
    "middle": 41,
    "ring": 38,
    "little": 35,
}


# 用于计算 global max 的实际压力通道
RH_PRESSURE_INDICES = np.concatenate([
    RH_FINGERS["thumb"].reshape(-1),
    RH_FINGERS["index"].reshape(-1),
    RH_FINGERS["middle"].reshape(-1),
    RH_FINGERS["ring"].reshape(-1),
    RH_FINGERS["little"].reshape(-1),
    RH_PALM.reshape(-1),
]) - 1


# ============================================================
# 串口解析
# ============================================================

def read_tactile_frames(ser):
    """
    解析右手完整压力帧。

    第一包：
        AA 55 03 99
        packet=01
        sensor=02
        128 byte pressure

    第二包：
        AA 55 03 99
        packet=02
        sensor=02
        128 byte pressure
        16 byte IMU

    yield:
        pressure: np.uint8 [256]
        timestamp_monotonic
        timestamp_unix
    """

    buf = bytearray()
    first_packet = None

    while True:

        chunk = ser.read(4096)

        if chunk:
            buf.extend(chunk)

        while True:

            p1 = buf.find(HEADER)

            if p1 < 0:
                buf = buf[-3:]
                break

            p2 = buf.find(HEADER, p1 + 4)

            if p2 < 0:

                if p1 > 0:
                    del buf[:p1]

                break

            frame = bytes(buf[p1:p2])

            del buf[:p2]

            if len(frame) < 6:
                continue

            packet_id = frame[4]
            sensor_type = frame[5]

            # 这里只处理右手
            if sensor_type != RIGHT_HAND:
                continue

            # 第一包
            if packet_id == 0x01 and len(frame) == 134:

                first_packet = frame[6:134]

            # 第二包
            elif (
                packet_id == 0x02
                and len(frame) == 150
                and first_packet is not None
            ):

                second_packet = frame[6:134]

                pressure = np.frombuffer(
                    first_packet + second_packet,
                    dtype=np.uint8
                ).copy()

                first_packet = None

                if pressure.shape[0] != 256:
                    continue

                yield (
                    pressure,
                    time.perf_counter(),
                    time.time(),
                )


# ============================================================
# 可视化
# ============================================================

def value_to_color(value):
    """
    value 已经归一化到 0~255。
    """

    value = int(np.clip(value, 0, 255))

    img = np.array([[value]], dtype=np.uint8)

    color = cv2.applyColorMap(
        img,
        cv2.COLORMAP_JET
    )[0, 0]

    return (
        int(color[0]),
        int(color[1]),
        int(color[2]),
    )


def draw_sensor_grid(
    canvas,
    values,
    indices,
    origin,
    cell_size=24,
    gap=3,
):
    """
    一个传感器 = 一个小方格。
    """

    ox, oy = origin

    rows, cols = indices.shape

    for r in range(rows):

        for c in range(cols):

            sensor_index = indices[r, c] - 1

            value = values[sensor_index]

            color = value_to_color(value)

            x1 = ox + c * (cell_size + gap)
            y1 = oy + r * (cell_size + gap)

            x2 = x1 + cell_size
            y2 = y1 + cell_size

            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                color,
                thickness=-1,
            )

            # 单元边界
            cv2.rectangle(
                canvas,
                (x1, y1),
                (x2, y2),
                (40, 40, 40),
                thickness=1,
            )


def draw_label(canvas, text, xy):

    cv2.putText(
        canvas,
        text,
        xy,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )


def render_right_hand(
    normalized_pressure,
    relative_pressure,
    frame_id,
    time_s,
    global_max,
):
    """
    normalized_pressure:
        仅用于颜色，范围 0~255

    relative_pressure:
        原始相对 ADC，不做动态归一化
    """

    WIDTH = 760
    HEIGHT = 780

    canvas = np.zeros(
        (HEIGHT, WIDTH, 3),
        dtype=np.uint8
    )

    # --------------------------------------------------------
    # 标题
    # --------------------------------------------------------

    cv2.putText(
        canvas,
        "Right Hand Tactile Pressure",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


    # --------------------------------------------------------
    # 五指
    #
    # 右手掌面朝观察者：
    #
    # thumb   index middle ring little
    #
    # --------------------------------------------------------

    finger_positions = {

        "thumb":  (85, 305),

        "index":  (255, 155),

        "middle": (350, 115),

        "ring":   (445, 145),

        "little": (540, 200),
    }


    for name in [
        "thumb",
        "index",
        "middle",
        "ring",
        "little",
    ]:

        draw_sensor_grid(
            canvas,
            normalized_pressure,
            RH_FINGERS[name],
            finger_positions[name],
        )

        x, y = finger_positions[name]

        draw_label(
            canvas,
            name,
            (x, y - 12),
        )


    # --------------------------------------------------------
    # 手掌
    # --------------------------------------------------------

    palm_origin = (220, 410)

    draw_sensor_grid(
        canvas,
        normalized_pressure,
        RH_PALM,
        palm_origin,
    )

    draw_label(
        canvas,
        "palm",
        (
            palm_origin[0],
            palm_origin[1] - 12
        ),
    )


    # --------------------------------------------------------
    # 状态信息
    # --------------------------------------------------------

    mapped_relative = relative_pressure[
        RH_PRESSURE_INDICES
    ]

    current_max = float(
        np.max(mapped_relative)
    )

    current_mean = float(
        np.mean(mapped_relative)
    )

    info = [
        f"frame       : {frame_id}",
        f"time        : {time_s:.3f} s",
        f"frame max   : {current_max:.2f}",
        f"frame mean  : {current_mean:.2f}",
        f"global max  : {global_max:.2f}",
    ]

    y = 640

    for text in info:

        cv2.putText(
            canvas,
            text,
            (25, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        y += 25


    # --------------------------------------------------------
    # 全局统一色条
    # --------------------------------------------------------

    bar_x = 440
    bar_y = 665

    bar_w = 260
    bar_h = 28

    gradient = np.linspace(
        0,
        255,
        bar_w,
        dtype=np.uint8
    )[None, :]

    gradient = np.repeat(
        gradient,
        bar_h,
        axis=0
    )

    gradient = cv2.applyColorMap(
        gradient,
        cv2.COLORMAP_JET
    )

    canvas[
        bar_y:bar_y + bar_h,
        bar_x:bar_x + bar_w
    ] = gradient


    cv2.rectangle(
        canvas,
        (bar_x, bar_y),
        (bar_x + bar_w, bar_y + bar_h),
        (255, 255, 255),
        1,
    )


    # 左端
    cv2.putText(
        canvas,
        "0",
        (bar_x, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


    # 右端是真正的 global max relative ADC
    cv2.putText(
        canvas,
        f"{global_max:.1f}",
        (bar_x + bar_w - 55, bar_y - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


    cv2.putText(
        canvas,
        "relative ADC",
        (bar_x, bar_y + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


    return canvas


# ============================================================
# H265 MP4 编码
# ============================================================

def encode_video(
    relative,
    timestamps,
    output_path,
    global_max,
    fps,
):

    if shutil.which("ffmpeg") is None:
        raise RuntimeError(
            "找不到 ffmpeg，请先执行：sudo apt install ffmpeg"
        )

    WIDTH = 760
    HEIGHT = 780


    # --------------------------------------------------------
    # 视频时间轴
    # --------------------------------------------------------

    duration = timestamps[-1]

    video_times = np.arange(
        0,
        duration,
        1.0 / fps,
        dtype=np.float64,
    )

    print()
    print("Rendering video")
    print("------------------------------")
    print(f"record duration : {duration:.3f} s")
    print(f"video fps       : {fps}")
    print(f"video frames    : {len(video_times)}")
    print(f"global max      : {global_max:.3f}")
    print(f"output          : {output_path}")
    print()


    # --------------------------------------------------------
    # FFmpeg H265 / HEVC
    # --------------------------------------------------------

    cmd = [

        "ffmpeg",
        "-y",

        "-loglevel",
        "error",

        "-f",
        "rawvideo",

        "-vcodec",
        "rawvideo",

        "-pix_fmt",
        "bgr24",

        "-s",
        f"{WIDTH}x{HEIGHT}",

        "-r",
        str(fps),

        "-i",
        "-",

        "-an",

        # H265
        "-c:v",
        "libx265",

        "-preset",
        "medium",

        "-crf",
        "23",

        "-pix_fmt",
        "yuv420p",

        # MP4 中标记 HEVC，提高兼容性
        "-tag:v",
        "hvc1",

        "-movflags",
        "+faststart",

        output_path,
    ]


    process = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
    )


    # --------------------------------------------------------
    # 根据真实时间戳重采样
    # --------------------------------------------------------

    source_indices = np.searchsorted(
        timestamps,
        video_times,
        side="right"
    ) - 1

    source_indices = np.clip(
        source_indices,
        0,
        len(timestamps) - 1
    )


    for video_frame_id, src_idx in enumerate(source_indices):

        rel = relative[src_idx]


        # ----------------------------------------------------
        # 全局归一化
        #
        # 整个视频使用同一个 global_max。
        #
        # 注意：
        # 255 在这里只是 COLORMAP 的颜色编码范围，
        # 不代表压力最大值。
        # ----------------------------------------------------

        if global_max > 1e-8:

            normalized = (
                rel / global_max * 255.0
            )

        else:

            normalized = np.zeros_like(rel)


        normalized = np.clip(
            normalized,
            0,
            255
        ).astype(np.uint8)


        frame = render_right_hand(
            normalized_pressure=normalized,
            relative_pressure=rel,
            frame_id=src_idx,
            time_s=video_times[video_frame_id],
            global_max=global_max,
        )


        try:

            process.stdin.write(
                frame.tobytes()
            )

        except BrokenPipeError:

            raise RuntimeError(
                "FFmpeg 编码失败。"
            )


    process.stdin.close()

    return_code = process.wait()


    if return_code != 0:

        raise RuntimeError(
            f"FFmpeg 返回错误码 {return_code}"
        )


# ============================================================
# 保存 CSV
# ============================================================

def save_csv(
    path,
    data,
    timestamps,
):

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.writer(f)

        header = [
            "frame",
            "time_s",
        ] + [
            f"sensor_{i}"
            for i in range(1, 257)
        ]

        writer.writerow(header)


        for i in range(len(data)):

            writer.writerow(
                [
                    i,
                    f"{timestamps[i]:.6f}",
                ]
                + data[i].tolist()
            )


# ============================================================
# 主程序
# ============================================================

def main():

    parser = argparse.ArgumentParser()


    parser.add_argument(
        "--port",
        default="/dev/ttyACM0"
    )

    parser.add_argument(
        "--baud",
        type=int,
        default=921600
    )

    parser.add_argument(
        "--out_dir",
        required=True
    )

    parser.add_argument(
        "--duration",
        type=float,
        default=10
    )

    parser.add_argument(
        "--baseline_seconds",
        type=float,
        default=2.0
    )

    parser.add_argument(
        "--video_fps",
        type=float,
        default=30.0
    )


    args = parser.parse_args()


    os.makedirs(
        args.out_dir,
        exist_ok=True
    )


    # ========================================================
    # 连接串口
    # ========================================================

    ser = serial.Serial(
        args.port,
        args.baud,
        timeout=1
    )


    raw_frames = []
    timestamps = []
    unix_timestamps = []


    start_time = None


    print()
    print("Recording")
    print("------------------------------")
    print(f"port             : {args.port}")
    print(f"baud             : {args.baud}")
    print(f"duration         : {args.duration} s")
    print(f"baseline         : {args.baseline_seconds} s")
    print(f"output directory : {args.out_dir}")
    print()
    print(
        f"前 {args.baseline_seconds} 秒保持手套不受力。"
    )
    print()


    try:

        for (
            pressure,
            t_monotonic,
            t_unix,
        ) in read_tactile_frames(ser):


            if start_time is None:
                start_time = t_monotonic


            elapsed = (
                t_monotonic
                - start_time
            )


            raw_frames.append(
                pressure
            )

            timestamps.append(
                elapsed
            )

            unix_timestamps.append(
                t_unix
            )


            if len(raw_frames) % 100 == 0:

                print(
                    f"\r"
                    f"time={elapsed:7.2f}s "
                    f"frames={len(raw_frames):7d}",
                    end="",
                    flush=True
                )


            if (
                args.duration > 0
                and elapsed >= args.duration
            ):

                break


    except KeyboardInterrupt:

        print(
            "\nRecording stopped by Ctrl+C"
        )


    finally:

        ser.close()


    print()


    if len(raw_frames) == 0:

        raise RuntimeError(
            "没有采集到压力数据。"
        )


    # ========================================================
    # 转成数组
    # ========================================================

    raw = np.stack(
        raw_frames,
        axis=0
    ).astype(np.uint8)


    timestamps = np.asarray(
        timestamps,
        dtype=np.float64
    )


    unix_timestamps = np.asarray(
        unix_timestamps,
        dtype=np.float64
    )


    print()
    print(
        f"Captured {len(raw)} tactile frames."
    )

    print(
        f"Actual duration: "
        f"{timestamps[-1]:.3f} s"
    )


    # ========================================================
    # baseline
    # ========================================================

    baseline_mask = (
        timestamps
        <= args.baseline_seconds
    )


    if not np.any(baseline_mask):

        raise RuntimeError(
            "baseline 时间内没有采集到数据。"
        )


    baseline = raw[
        baseline_mask
    ].astype(
        np.float32
    ).mean(
        axis=0
    )


    # ========================================================
    # 相对压力
    #
    # relative = raw ADC - baseline
    # ========================================================

    relative = (
        raw.astype(np.float32)
        - baseline[None, :]
    )


    relative = np.maximum(
        relative,
        0
    )


    # ========================================================
    # 整段数据 global max
    #
    # 只统计真正映射到手上的压力点
    # 不把弯折数据/未使用通道混进来
    # ========================================================

    global_max = float(
        np.max(
            relative[
                :,
                RH_PRESSURE_INDICES
            ]
        )
    )


    print(
        f"Global maximum relative ADC: "
        f"{global_max:.3f}"
    )


    # ========================================================
    # 保存数据
    # ========================================================

    np.save(
        os.path.join(
            args.out_dir,
            "RH_raw.npy"
        ),
        raw
    )


    np.save(
        os.path.join(
            args.out_dir,
            "RH_relative.npy"
        ),
        relative
    )


    np.save(
        os.path.join(
            args.out_dir,
            "RH_baseline.npy"
        ),
        baseline
    )


    np.save(
        os.path.join(
            args.out_dir,
            "timestamps.npy"
        ),
        timestamps
    )


    # 一个文件保存全部数据，方便后续直接 np.load
    np.savez_compressed(

        os.path.join(
            args.out_dir,
            "RH_tactile.npz"
        ),

        raw=raw,

        relative=relative,

        baseline=baseline,

        timestamps=timestamps,

        unix_timestamps=unix_timestamps,

        global_max=np.float32(
            global_max
        ),
    )


    # CSV
    save_csv(
        os.path.join(
            args.out_dir,
            "RH_raw.csv"
        ),
        raw,
        timestamps
    )


    save_csv(
        os.path.join(
            args.out_dir,
            "RH_relative.csv"
        ),
        relative,
        timestamps
    )


    # ========================================================
    # metadata
    # ========================================================

    metadata = {

        "port":
            args.port,

        "baud":
            args.baud,

        "frames":
            int(len(raw)),

        "duration_seconds":
            float(timestamps[-1]),

        "baseline_seconds":
            args.baseline_seconds,

        "video_fps":
            args.video_fps,

        "global_max_relative_adc":
            global_max,

        "video_normalization":
            "relative_adc / global_max",

        "video_codec":
            "H.265 / HEVC",

        "video_container":
            "MP4",
    }


    with open(
        os.path.join(
            args.out_dir,
            "metadata.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            metadata,
            f,
            ensure_ascii=False,
            indent=2
        )


    # ========================================================
    # 视频
    # ========================================================

    output_video = os.path.join(
        args.out_dir,
        "RH_hand_heatmap.mp4"
    )


    encode_video(
        relative=relative,
        timestamps=timestamps,
        output_path=output_video,
        global_max=global_max,
        fps=args.video_fps,
    )


    print()
    print("Finished.")
    print()
    print(
        f"Video: {output_video}"
    )
    print(
        f"Data : "
        f"{os.path.join(args.out_dir, 'RH_tactile.npz')}"
    )


if __name__ == "__main__":
    main()