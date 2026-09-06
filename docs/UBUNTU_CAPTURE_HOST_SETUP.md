# Ubuntu 采集主机部署手册

本文档用于把 Orbbec 采集程序部署到多台全新 Ubuntu 主机。目标是让每台主机都能完成：

1. 安装系统依赖和 Python 环境。
2. 配置 Orbbec USB 权限、USB 缓冲和用户组。
3. 编译 `orbbec` 主程序和 `orbbec_probe` 探测工具。
4. 按本机真实硬件修改采集配置。
5. 完成相机枚举、启动、采集、回放或标注的基础验收。

仓库根目录以下用 `$ORBBEC_ROOT` 表示，例如：

```bash
export ORBBEC_ROOT="$HOME/Orbbec"
cd "$ORBBEC_ROOT"
```

## 1. 主机建议规格

推荐环境：

| 项目 | 建议 |
| --- | --- |
| 系统 | Ubuntu 22.04 LTS 或 24.04 LTS |
| 架构 | x86_64 |
| CMake | 3.22 或更高 |
| 内存 | 多相机采集建议 32 GB 以上 |
| 磁盘 | NVMe SSD，采集目录预留足够空间 |
| USB | 独立 USB3 控制器或质量可靠的带电源 USB3 Hub |
| 显示 | 主程序使用 OpenCV 图形窗口，需要桌面环境、显示器、VNC 或 X11 转发 |

注意：

- 仓库自带的 Orbbec SDK 动态库是 Linux x86_64 用的。如果主机是 ARM64，例如 Jetson/AGX，需要替换为对应架构的 Orbbec SDK。
- Ubuntu 20.04 默认 CMake 通常低于 3.22，不建议直接使用；如必须使用，需要先升级 CMake。

## 2. 首次检查

在新主机上先检查系统架构、CMake、Python 和 SDK 文件状态：

```bash
cd "$ORBBEC_ROOT"

uname -m
cmake --version || true
python3 --version || true

file lib/OrbbecSDK_v2.7.2/lib/libOrbbecSDK.so.2.7.2
ldd lib/OrbbecSDK_v2.7.2/lib/libOrbbecSDK.so.2.7.2 | grep "not found" || true
```

期望结果：

- `uname -m` 输出 `x86_64`。
- `cmake --version` 显示 `3.22` 或更高。
- `file ...libOrbbecSDK.so.2.7.2` 显示 `ELF 64-bit LSB shared object, x86-64`。
- `ldd ... | grep "not found"` 没有输出。

如果 SDK 文件不是 ELF，或者 `ldd` 有 `not found`，先不要继续构建，先确认 Git 拉取是否完整、SDK 是否被错误替换、文件是否损坏。

## 3. 安装系统依赖

执行：

```bash
sudo apt update

sudo apt install -y \
  build-essential git pkg-config cmake ninja-build \
  libopencv-dev libopencv-contrib-dev \
  libpcl-dev libeigen3-dev \
  libusb-1.0-0 libusb-1.0-0-dev libudev-dev \
  usbutils v4l-utils ffmpeg \
  python3 python3-venv python3-pip \
  fonts-noto-cjk xclip zenity yad \
  alsa-utils espeak speech-dispatcher
```

这些包的用途：

| 包 | 用途 |
| --- | --- |
| `build-essential`, `cmake` | 编译 C++ 主程序 |
| `libopencv-dev` | C++ 图像处理和 UI 窗口 |
| `libopencv-contrib-dev` | 推荐安装，支持更多 OpenCV 模块和中文 FreeType 渲染能力 |
| `libpcl-dev` | 点云处理，主程序直接依赖 |
| `libusb-1.0-0`, `libudev-dev` | Orbbec/USB 设备访问 |
| `v4l-utils` | 检查鱼眼或普通 UVC 摄像头 |
| `ffmpeg` | H.265 编码、解码、回放、后处理 |
| `fonts-noto-cjk` | 中文显示 |
| `xclip`, `zenity`, `yad` | Viewer/Collection 中的剪贴板和目录选择辅助 |
| `alsa-utils`, `espeak`, `speech-dispatcher` | 采集语音提示回退和声卡检查 |

中文采集播报使用 `zh-CN-XiaoxiaoNeural` 自然 neural TTS，默认语速为 `+10%`、音调为 `+0Hz`。仓库随附默认提示的 MP3，并把语音文件持久化在程序工作目录的 `voice/`；已有文件会直接播放，只有提示文案或音色参数变化、对应文件不存在时才会重新生成。因此重启程序不会重复生成，实际操作也不再承担在线生成带来的约 2 秒延迟。新的操作提示会取消并覆盖正在播放的旧提示，因此状态切换时不会积压过期语音。

```bash
python3 -m pip install --user edge-tts
```

如果 CMake 版本仍然低于 3.22，可用 snap 安装新版：

```bash
sudo snap install cmake --classic
hash -r
cmake --version
```

也可以使用 Kitware 官方 apt 源，但批量部署时 snap 通常更简单。

## 4. 配置用户权限和 USB 规则

安装 Orbbec udev 规则：

```bash
cd "$ORBBEC_ROOT"
sudo bash lib/OrbbecSDK_v2.7.2/shared/install_udev_rules.sh
```

把当前用户加入常用设备组：

```bash
for g in video render dialout plugdev; do
  getent group "$g" >/dev/null && sudo usermod -aG "$g" "$USER"
done
```

说明：

- `video`：Orbbec/V4L 摄像头设备访问。
- `render`：`/dev/dri/renderD*`，H.265 VAAPI/QSV 硬件编码可能需要。
- `dialout`：触觉串口设备可能需要。
- `plugdev`：部分 USB 设备规则可能使用。

执行后请重新登录，最稳妥是重启：

```bash
sudo reboot
```

重启后确认用户组：

```bash
id
```

## 5. 配置 USB 缓冲

多路相机采集容易受到 Linux USBFS 缓冲限制影响。运行前可临时设置：

```bash
echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

为了重启后保留，写入 modprobe 配置：

```bash
echo "options usbcore usbfs_memory_mb=128" | sudo tee /etc/modprobe.d/usbcore.conf
```

注意：该配置通常需要重启后完全生效。仓库里的 `run.sh` 也会在启动前临时设置一次。

## 6. 建立 Python 环境

主程序虽然是 C++，但默认采集流程会在开始前调用 `python3` 执行 AprilTag 外参健康检查。这个检查需要 Python 版 OpenCV 的 `aruco` 模块，所以必须安装 `opencv-contrib-python`。

建议在仓库根目录建立统一虚拟环境：

```bash
cd "$ORBBEC_ROOT"

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip

python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
python -m pip install numpy opencv-contrib-python pyyaml matplotlib Pillow imageio-ffmpeg
```

验证：

```bash
source "$ORBBEC_ROOT/.venv/bin/activate"
python - <<'PY'
import cv2
import numpy
print("cv2", cv2.__version__)
print("has aruco:", hasattr(cv2, "aruco"))
raise SystemExit(0 if hasattr(cv2, "aruco") else 1)
PY
```

如果后续需要启用手部关节自动处理脚本，再安装：

```bash
source "$ORBBEC_ROOT/.venv/bin/activate"
python -m pip install mediapipe
```

不要把 PICO streaming server 的 Python 环境和主程序环境混在一起。PICO server 可以使用它自己的 `egopj-main/outside/stream_server_ubuntu/.venv`。

## 7. 编译主程序

执行：

```bash
cd "$ORBBEC_ROOT"
source .venv/bin/activate

chmod +x build.sh run.sh
./build.sh
```

构建完成后应有：

| 文件 | 说明 |
| --- | --- |
| `bin/orbbec` | 主程序入口 |
| `build/orbbec_probe` | Orbbec 设备探测工具 |

检查动态库：

```bash
ldd bin/orbbec | grep "not found" || echo "bin/orbbec runtime libs OK"
ldd build/orbbec_probe | grep "not found" || echo "build/orbbec_probe runtime libs OK"
```

如果 CMake 找不到 SDK，可显式指定：

```bash
cmake -S . -B build -DOrbbecSDK_ROOT="$ORBBEC_ROOT/lib/OrbbecSDK_v2.7.2"
cmake --build build -j"$(nproc)"
mkdir -p bin
cp build/orbbec bin/orbbec
```

## 8. 探测本机硬件

接好 Orbbec 相机、鱼眼相机、触觉串口设备和 PICO 后，先做枚举。

### 8.1 Orbbec 相机

```bash
cd "$ORBBEC_ROOT"
source .venv/bin/activate

lsusb | grep -i -E "2bc5|orbbec" || true
./build/orbbec_probe
```

`orbbec_probe` 会输出：

- SDK 版本。
- 已连接 Orbbec 设备数量。
- 每台设备序列号。
- 支持的同步模式。
- RGB/Depth/IMU profile。
- 常见流启动测试结果，如果加 `--stream-test`。

更完整测试：

```bash
./build/orbbec_probe --stream-test
```

如果有两台以上 Orbbec，并需要同步测试：

```bash
./build/orbbec_probe --sync-test
```

### 8.2 鱼眼或普通 UVC 相机

```bash
ls -l /dev/v4l/by-id/ || true
v4l2-ctl --list-devices
```

配置文件中建议使用 `/dev/v4l/by-id/...`，不要使用易变的 `/dev/video0`、`/dev/video1`。

### 8.3 USB 拓扑

多相机卡顿或掉帧时先看 USB 拓扑：

```bash
lsusb -t
```

建议：

- 多台相机尽量分散到不同 USB 控制器。
- 使用 USB3 端口和优质线缆。
- 避免无源 Hub 和过长延长线。
- 多设备同步场景按硬件要求连接同步线或 Sync Hub。

### 8.4 PICO/Android 设备

如果启用 PICO ego streaming：

```bash
adb devices
```

如果没有 `adb`：

```bash
sudo apt install -y android-tools-adb
```

## 9. 每台主机的配置文件

默认配置是 `src/sync/config.json`。批量部署时，不建议每台机器都直接改这个文件，因为容易污染 Git。推荐复制一个本机配置，并仍然放在 `src/sync/` 目录下：

```bash
cd "$ORBBEC_ROOT"
cp src/sync/config.json "src/sync/config.$(hostname).json"
```

为什么要放在 `src/sync/`：

- 配置里的 `scriptPath: "extrinsic_health_check.py"` 是相对配置文件位置解析的。
- 配置里的 `cameraParamsPath: "../../camera_info/ego_camera_params.json"` 也依赖相对路径。
- 放在 `src/sync/` 下可以保持默认相对路径有效。

如果你把配置文件放到别的目录，必须同步改这些相对路径。

### 9.1 必改项

打开本机配置：

```bash
gedit "src/sync/config.$(hostname).json" &
```

或使用命令行编辑器：

```bash
nano "src/sync/config.$(hostname).json"
```

重点修改：

| 配置项 | 说明 |
| --- | --- |
| `devices[].sn` | 改成本机 `orbbec_probe` 输出的 Orbbec 序列号 |
| `devices[].index` | 数据保存时的相机编号，例如 `"00"`, `"01"`, `"02"` |
| `devices[].syncConfig.syncMode` | 主从同步模式，第一台通常是 `PRIMARY` |
| `devices[].streams[].enable` | 是否启用 RGB、Depth、Point Cloud |
| `fisheye.enabled` | 没接鱼眼先设为 `false` |
| `fisheye.cameras[].uniqueId` | 改成本机 `/dev/v4l/by-id/...` 路径 |
| `ego.enabled` | 没接 PICO 先设为 `false` |
| `extrinsicHealthCheck.enabled` | 没准备 AprilTag/外参检查时可先设为 `false` |
| `outputDir` | 默认输出根目录 |
| `save.rgbEncoding` | `image` 或 `h265` |
| `save.depthEncoding` | `png` 或 `ffv1` 等 |

### 9.2 最小 Orbbec-only 配置建议

如果只是先验证 Orbbec 主流程，建议临时：

```json
"ego": {
  "enabled": false
},
"fisheye": {
  "enabled": false
},
"extrinsicHealthCheck": {
  "enabled": false
}
```

然后只保留真实存在的 `devices`，并先启用一个低风险 stream 组合。

### 9.3 多设备同步配置

典型三台 Orbbec：

- 第 1 台：`OB_MULTI_DEVICE_SYNC_MODE_PRIMARY`，`triggerOutEnable: true`
- 第 2/3 台：`OB_MULTI_DEVICE_SYNC_MODE_SECONDARY_SYNCED`，`triggerOutEnable: false`

采集前确认：

```bash
./build/orbbec_probe --sync-test
```

如果硬件没有同步线或 Sync Hub，不要盲目启用强同步配置。先用普通采集确认每台相机都稳定，再上同步。

## 10. 运行

使用本机配置启动：

```bash
cd "$ORBBEC_ROOT"
source .venv/bin/activate

echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
bin/orbbec "src/sync/config.$(hostname).json"
```

如果使用默认配置，也可以运行：

```bash
./run.sh
```

但 `run.sh` 默认不传配置文件，会使用：

1. 当前目录下的 `config.json`，如果存在。
2. 否则使用 `src/sync/config.json`。

批量部署时建议明确传入本机配置，避免误用默认配置。

## 11. PICO streaming server 可选部署

如果需要 PICO ego 采集，进入 Unity 项目目录：

```bash
cd "$ORBBEC_ROOT/egopj-main"
```

安装 Ubuntu server 环境：

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip android-tools-adb ffmpeg

bash outside/stream_server_ubuntu/install_env.sh
source outside/stream_server_ubuntu/.venv/bin/activate
```

配置 ADB reverse：

```bash
bash outside/stream_server_ubuntu/setup_adb_reverse.sh --port 50051
```

启动 server：

```bash
python outside/stream_server_ubuntu/server.py \
  --host 127.0.0.1 \
  --port 50051 \
  --output-root outside/stream_server_ubuntu/sessions
```

server 控制台常用命令：

```text
start test_001
timecalibrate
status
stop
quit
```

如果主采集程序中启用 `ego.enabled: true`，还要确认 `src/sync/config.<host>.json` 里的：

- `ego.host`
- `ego.port`
- `ego.cameraParamsPath`
- `ego.timeCalibrate`

## 12. H.265 编码可选配置

默认配置里 RGB 保存方式可能是 `image`。如果改成 H.265：

```json
"save": {
  "rgbEncoding": "h265",
  "h265EncoderMode": "software",
  "h265Codec": "",
  "h265Crf": 23
}
```

先确认 FFmpeg 支持 H.265：

```bash
ffmpeg -hide_banner -encoders | grep -Ei "hevc|h265|265"
```

软件编码通常使用 `libx265`。硬件编码可按机器选择：

| 硬件 | 可能的 codec |
| --- | --- |
| NVIDIA | `hevc_nvenc` |
| Intel/VAAPI | `hevc_vaapi` |
| Intel/QSV | `hevc_qsv` |
| V4L2 M2M | `hevc_v4l2m2m` |

仓库有探测脚本：

```bash
python3 scripts/probe_h265_igpu.py
```

如果使用 `/dev/dri/renderD128`，确认权限：

```bash
ls -l /dev/dri/
id
```

当前用户通常需要在 `render` 组。

## 13. 标注和离线工具

### 13.1 关节标注工具

依赖：

```bash
source "$ORBBEC_ROOT/.venv/bin/activate"
python -m pip install Pillow
```

启动：

```bash
cd "$ORBBEC_ROOT"
python3 -m src.label.main
```

如中文字体异常：

```bash
sudo apt install -y fonts-noto-cjk
```

### 13.2 棋盘/AprilTag 位姿工具

建议使用独立虚拟环境：

```bash
cd "$ORBBEC_ROOT/cam_pose_via_checkboard"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

运行示例：

```bash
python3 -m src.main \
  --dataset_root /path/to/dataset_root \
  --config configs/default.yaml
```

## 14. 单机验收清单

每台主机部署完成后按下面顺序验收。

### 14.1 基础环境

```bash
cd "$ORBBEC_ROOT"
source .venv/bin/activate

uname -m
cmake --version
python - <<'PY'
import cv2
print(cv2.__version__, hasattr(cv2, "aruco"))
PY
ffmpeg -version | head -1
```

通过标准：

- 架构是 `x86_64`。
- CMake `>= 3.22`。
- `hasattr(cv2, "aruco")` 输出 `True`。
- FFmpeg 可运行。

### 14.2 SDK 和构建产物

```bash
file lib/OrbbecSDK_v2.7.2/lib/libOrbbecSDK.so.2.7.2
ldd bin/orbbec | grep "not found" || true
ldd build/orbbec_probe | grep "not found" || true
```

通过标准：

- SDK 是 ELF x86-64。
- 没有 `not found`。

### 14.3 设备枚举

```bash
lsusb | grep -i -E "2bc5|orbbec" || true
./build/orbbec_probe
ls -l /dev/v4l/by-id/ || true
```

通过标准：

- `orbbec_probe` 能看到预期数量的 Orbbec。
- 每台设备序列号与本机配置一致。
- 鱼眼相机路径与配置一致。

### 14.4 启动主程序

```bash
echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
bin/orbbec "src/sync/config.$(hostname).json"
```

通过标准：

- UI 窗口正常打开。
- 能进入需要的采集/查看页面。
- 启动相机时没有设备缺失、权限不足、profile 不支持等错误。

### 14.5 采集小样本

建议先采 5 秒或一个很短 episode，然后检查输出目录：

```bash
find ./collection_000 -maxdepth 3 -type f | head -50
```

通过标准：

- 输出目录存在。
- 各相机目录按 `index` 生成。
- RGB/Depth/PointCloud 文件数量符合预期。
- 如果启用 PICO，有 `ego` 相关文件。
- 如果启用鱼眼，有 `fisheye` 相关文件。

## 15. 常见问题

### 15.1 CMake 报 `CMake 3.22 or higher is required`

原因：系统 CMake 太旧。

处理：

```bash
sudo snap install cmake --classic
hash -r
cmake --version
```

### 15.2 CMake 找不到 `OrbbecSDK`

先检查 SDK 文件：

```bash
ls -lh lib/OrbbecSDK_v2.7.2/lib/
file lib/OrbbecSDK_v2.7.2/lib/libOrbbecSDK.so.2.7.2
```

再显式指定：

```bash
cmake -S . -B build -DOrbbecSDK_ROOT="$ORBBEC_ROOT/lib/OrbbecSDK_v2.7.2"
```

### 15.3 CMake 找不到 OpenCV 或 PCL

处理：

```bash
sudo apt update
sudo apt install -y libopencv-dev libopencv-contrib-dev libpcl-dev libeigen3-dev
rm -rf build
./build.sh
```

### 15.4 `orbbec_probe` 看不到相机

按顺序检查：

```bash
lsusb | grep -i -E "2bc5|orbbec" || true
ls -l /etc/udev/rules.d/99-obsensor-libusb.rules
id
cat /sys/module/usbcore/parameters/usbfs_memory_mb
```

处理：

1. 重新插拔相机。
2. 换 USB3 端口或线缆。
3. 重新安装 udev 规则。
4. 确认用户在 `video` 组。
5. 重启主机。

### 15.4.1 `lsusb` 能看到 Orbbec，但 SDK 显示 0 台

现象示例：

```text
Bus 006 Device 006: ID 2bc5:0807 Orbbec Gemini 336L
Bus 006 Device 007: ID 2bc5:0807 Orbbec Gemini 336L
Bus 006 Device 008: ID 2bc5:0807 Orbbec Gemini 336L
Orbbec SDK version: 2.7.2
Connected Orbbec devices: 0
```

这说明 Linux USB 层已经识别到设备，但 Orbbec SDK 还不能枚举。优先检查 udev 规则、设备节点权限、当前用户组、SDK 运行环境。

先确认仓库自带规则包含 Gemini 336L 的 `0807`：

```bash
cd "$ORBBEC_ROOT"
grep '0807' lib/OrbbecSDK_v2.7.2/shared/99-obsensor-libusb.rules
```

重新安装并触发 udev 规则：

```bash
cd "$ORBBEC_ROOT"
sudo bash lib/OrbbecSDK_v2.7.2/shared/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
```

然后必须重新插拔所有 Orbbec 相机。若仍然为 0，直接重启：

```bash
sudo reboot
```

重启后检查 USB 设备节点权限。以下命令会根据 `lsusb` 输出列出对应 `/dev/bus/usb/...`：

```bash
lsusb | awk '/2bc5:0807/ {
  bus=$2
  dev=$4
  gsub(":", "", dev)
  printf "/dev/bus/usb/%03d/%03d\n", bus, dev
}' | xargs -r ls -l
```

期望权限包含 `rw`，最好是类似：

```text
crw-rw-rw- 1 root video ... /dev/bus/usb/006/006
```

或者至少当前用户所在组有读写权限。检查当前用户组：

```bash
id
```

如用户不在 `video` 组：

```bash
sudo usermod -aG video "$USER"
sudo reboot
```

如果普通用户仍然枚举不到，用 `sudo` 做一次区分：

```bash
cd "$ORBBEC_ROOT"
sudo ./build/orbbec_probe
```

判断：

- `sudo ./build/orbbec_probe` 能看到相机，普通用户看不到：几乎确定是 udev/用户组/设备节点权限问题。
- `sudo ./build/orbbec_probe` 也看不到：继续检查 SDK 动态库、USB 拓扑、内核日志或设备固件状态。

继续检查 SDK 动态库：

```bash
cd "$ORBBEC_ROOT"
file lib/OrbbecSDK_v2.7.2/lib/libOrbbecSDK.so.2.7.2
ldd build/orbbec_probe | grep "not found" || true
```

检查内核是否有 USB 错误：

```bash
dmesg -T | grep -i -E "2bc5|orbbec|usb|uvc" | tail -120
lsusb -t
```

如果三台相机都在同一个 Bus/Hub 下并且 SDK 仍然枚举失败，先只插一台相机测试：

```bash
./build/orbbec_probe
```

一台可见、三台不可见时，优先处理 USB 供电、线缆、Hub、控制器带宽和 `usbfs_memory_mb`。

### 15.4.2 `sudo orbbec_probe` 能看到设备，但普通 collection 没帧

现象：

- `sudo ./build/orbbec_probe` 能看到 Orbbec。
- `./build/orbbec_probe` 普通用户看不到，或主程序进入 collection 后没有收到帧。
- `lsusb` 能看到 `2bc5:xxxx`。

这几乎一定是普通用户设备权限未生效。不要先改采集逻辑，先修权限。

检查普通用户能否枚举：

```bash
cd "$ORBBEC_ROOT"
./build/orbbec_probe
```

如果普通用户仍然是 0，按下面做：

```bash
cd "$ORBBEC_ROOT"
sudo bash lib/OrbbecSDK_v2.7.2/shared/install_udev_rules.sh
sudo udevadm control --reload-rules
sudo udevadm trigger
sudo usermod -aG video "$USER"
sudo usermod -aG render "$USER" 2>/dev/null || true
```

然后必须退出登录重新进入，推荐直接重启：

```bash
sudo reboot
```

重启后确认：

```bash
id
lsusb | awk '/2bc5:/ {
  bus=$2
  dev=$4
  gsub(":", "", dev)
  printf "/dev/bus/usb/%03d/%03d\n", bus, dev
}' | xargs -r ls -l
./build/orbbec_probe
```

只有当普通用户 `./build/orbbec_probe` 能看到设备后，再运行 collection。否则 collection 仍然会拿不到相机帧。

临时验证时可以用 root 运行主程序，但不建议作为长期方案：

```bash
cd "$ORBBEC_ROOT"
sudo -E env DISPLAY="$DISPLAY" XAUTHORITY="$XAUTHORITY" ./bin/orbbec "src/sync/config.$(hostname).json"
```

如果 root collection 有帧、普通用户没有帧，结论就是权限问题。

### 15.5 主程序提示 `Configured device not found`

原因：`src/sync/config.<host>.json` 里的 `devices[].sn` 和真实相机序列号不一致。

处理：

```bash
./build/orbbec_probe
```

把输出中的真实序列号填回配置。

### 15.6 鱼眼相机打不开

检查：

```bash
ls -l /dev/v4l/by-id/
v4l2-ctl --list-devices
```

处理：

- 更新 `fisheye.cameras[].uniqueId`。
- 暂时设置 `fisheye.enabled: false`，先验证 Orbbec 主流程。
- 换 USB 口，避免鱼眼和 Orbbec 挤在同一个 Hub。

### 15.7 外参健康检查失败：`OpenCV has no aruco module`

原因：Python 环境装了 `opencv-python` 或 `opencv-python-headless`，没有 `aruco`。

处理：

```bash
source "$ORBBEC_ROOT/.venv/bin/activate"
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
python -m pip install opencv-contrib-python numpy pyyaml matplotlib
python - <<'PY'
import cv2
print(hasattr(cv2, "aruco"))
PY
```

### 15.7.1 外参健康检查失败：`ModuleNotFoundError: No module named 'numpy'`

现象：

```text
[collection][extrinsic_check] status=(missing) ok=0 exit=1
summary=Traceback ...
  import numpy as np
ModuleNotFoundError: No module named 'numpy'
```

但手动执行下面命令又能成功：

```bash
source "$ORBBEC_ROOT/.venv/bin/activate"
python3 - <<'PY'
import cv2
import numpy
print("cv2", cv2.__version__)
print("has aruco:", hasattr(cv2, "aruco"))
PY
```

原因通常是 collection 实际调用的 Python 不是你刚刚测试的 Python。配置里的默认值是：

```json
"pythonExecutable": "python3"
```

如果主程序通过 `sudo` 启动，或启动时没有激活 `.venv`，这个 `python3` 很可能指向系统 Python 或 root 的 Python，因此没有 `numpy`。

最稳的修复方式：把本机配置中的 `extrinsicHealthCheck.pythonExecutable` 改成虚拟环境 Python 的绝对路径：

```json
"extrinsicHealthCheck": {
  "enabled": true,
  "pythonExecutable": "/home/user/Orbbec/.venv/bin/python",
  "scriptPath": "extrinsic_health_check.py"
}
```

把 `/home/user/Orbbec` 换成实际 `$ORBBEC_ROOT`。然后验证：

```bash
/home/user/Orbbec/.venv/bin/python - <<'PY'
import cv2
import numpy
print("python ok", cv2.__version__, hasattr(cv2, "aruco"))
PY
```

如果只是临时验证采集链路，也可以先关闭外参健康检查：

```json
"extrinsicHealthCheck": {
  "enabled": false
}
```

看到类似下面日志时，说明相机流已经在回调，问题不是“收不到相机帧”，而是录制开始前被健康检查阻塞：

```text
[collection] callback framesets=300 sn=... capturing=1 recording=0
```

`capturing=1` 表示相机正在出帧；`recording=0` 表示还没有进入正式保存阶段。

### 15.8 外参健康检查一直不通过

先确认是否真的需要它。没有 AprilTag、没有稳定外参、相机未全部就绪时，可以临时关闭：

```json
"extrinsicHealthCheck": {
  "enabled": false
}
```

如果需要保留检查：

- 确认 `init_extrinsic_path` 指向正确外参文件。
- 确认多台 RGB 相机都能看到足够 AprilTag。
- 确认 tag family 和尺寸，例如 `tag36h11`、`tagSizeM`。
- 检查采集输出目录下 `.extrinsic_health` 的 debug 结果。

### 15.9 UI 无法打开窗口

原因：没有图形环境或 DISPLAY 不可用。

检查：

```bash
echo "$DISPLAY"
```

处理：

- 使用本机显示器登录桌面。
- 使用 VNC/NoMachine。
- SSH 场景下配置 X11 转发，但多相机实时 UI 不推荐长期这么跑。

### 15.10 H.265 编码失败

检查：

```bash
ffmpeg -hide_banner -encoders | grep -Ei "hevc|h265|265"
python3 scripts/probe_h265_igpu.py
```

处理：

- 先改回 `rgbEncoding: "image"` 验证采集链路。
- 软件编码确认有 `libx265`。
- NVIDIA 硬件编码确认驱动和 `hevc_nvenc`。
- VAAPI/QSV 确认 `/dev/dri/renderD*` 权限和 `render` 用户组。

### 15.11 ALSA/aplay 语音提示报错

现象：

```text
ALSA lib pcm_dmix.c:1032:(snd_pcm_dmix_open) unable to open slave
aplay: main:831: 音乐打开错误： 没有那个文件或目录
```

这是采集语音提示播放失败，通常不影响相机取流。没有扬声器、默认声卡不存在、root/sudo 环境没有音频权限时都会出现。

自然语音会持久化在程序工作目录的 `voice/`。若完全没有播报或首次生成失败，安装 neural TTS 和播放器：

```bash
sudo apt install ffmpeg
python3 -m pip install --user edge-tts
```

如果不需要语音提示，直接在配置里关闭：

```json
"voiceFeedback": {
  "enabled": false
}
```

如果需要语音提示，先确认系统声卡：

```bash
aplay -l
speaker-test -t wav -c 2
```

再把 `voiceFeedback.speakerDevice` 改成可用设备，例如 `default`、`plughw:0,0` 或 `hw:1,0`。

## 16. 批量部署建议

### 16.1 统一准备

为每台主机记录：

| 字段 | 示例 |
| --- | --- |
| 主机名 | `orbbec-host-01` |
| Ubuntu 版本 | `22.04` |
| GPU/编码器 | `NVIDIA`, `Intel VAAPI`, `none` |
| Orbbec 序列号 | `AY3794F0038` |
| 相机 index | `00` |
| 鱼眼 by-id | `/dev/v4l/by-id/...` |
| PICO 是否启用 | `true/false` |
| 输出目录 | `/data/collection_000` |

### 16.2 建议流程

1. 先装系统依赖和 udev。
2. 重启。
3. 建 `.venv`。
4. 编译。
5. 跑 `orbbec_probe`。
6. 复制 `src/sync/config.json` 为 `src/sync/config.<hostname>.json`。
7. 写入本机序列号和设备路径。
8. 先关闭 PICO、鱼眼、外参健康检查，只验证 Orbbec。
9. 逐项打开鱼眼、PICO、外参健康检查、H.265。
10. 采集短样本并归档验收结果。

### 16.3 推荐保留的验收日志

每台主机部署完成后保存：

```bash
mkdir -p "$ORBBEC_ROOT/deploy_logs"

{
  date
  hostname
  uname -a
  cmake --version
  python3 --version
  id
  lsusb
  lsusb -t
  ls -l /dev/v4l/by-id/ 2>/dev/null || true
  cat /sys/module/usbcore/parameters/usbfs_memory_mb
  ffmpeg -version | head -1
} > "$ORBBEC_ROOT/deploy_logs/$(hostname)_env.txt" 2>&1

"$ORBBEC_ROOT/build/orbbec_probe" > "$ORBBEC_ROOT/deploy_logs/$(hostname)_orbbec_probe.txt" 2>&1
```

如需排查问题，把对应主机的：

- `deploy_logs/<host>_env.txt`
- `deploy_logs/<host>_orbbec_probe.txt`
- `src/sync/config.<host>.json`
- 采集输出目录中的错误日志或 `.extrinsic_health`

一起提供。

## 17. 新主机最短命令清单

下面是从零开始的压缩版，适合熟练人员复制执行。首次部署仍建议按完整文档逐项检查。

```bash
export ORBBEC_ROOT="$HOME/Orbbec"
cd "$ORBBEC_ROOT"

sudo apt update
sudo apt install -y \
  build-essential git pkg-config cmake ninja-build \
  libopencv-dev libopencv-contrib-dev \
  libpcl-dev libeigen3-dev \
  libusb-1.0-0 libusb-1.0-0-dev libudev-dev \
  usbutils v4l-utils ffmpeg \
  python3 python3-venv python3-pip \
  fonts-noto-cjk xclip zenity yad \
  alsa-utils espeak speech-dispatcher \
  android-tools-adb

sudo bash lib/OrbbecSDK_v2.7.2/shared/install_udev_rules.sh
for g in video render dialout plugdev; do
  getent group "$g" >/dev/null && sudo usermod -aG "$g" "$USER"
done
echo "options usbcore usbfs_memory_mb=128" | sudo tee /etc/modprobe.d/usbcore.conf

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python
python -m pip install numpy opencv-contrib-python pyyaml matplotlib Pillow imageio-ffmpeg

chmod +x build.sh run.sh
./build.sh

cp src/sync/config.json "src/sync/config.$(hostname).json"
./build/orbbec_probe
```

执行到这里后，先编辑 `src/sync/config.<hostname>.json`，再启动：

```bash
source "$ORBBEC_ROOT/.venv/bin/activate"
echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
bin/orbbec "src/sync/config.$(hostname).json"
```
