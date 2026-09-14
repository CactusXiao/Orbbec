# 新机器安装启动说明



## 1. 系统依赖

```bash
sudo apt update
sudo apt install -y \
  build-essential git pkg-config cmake \
  libopencv-dev libopencv-contrib-dev libpcl-dev libeigen3-dev \
  libusb-1.0-0 libusb-1.0-0-dev libudev-dev \
  usbutils v4l-utils ffmpeg cifs-utils tmux adb \
  python3 python3-venv python3-pip \
  fonts-noto-cjk xclip zenity yad alsa-utils espeak speech-dispatcher
```

基准版本：Ubuntu 22.04、Python 3.10.12、CMake 3.22.1、OpenCV 4.5.4、PCL 1.12.1、FFmpeg 4.4.2、Orbbec SDK `2.7.2`。

## 2. Python 依赖

```bash
cd /home/ubuntu/demo/Orbbec_demo
python3 -m pip install -U pip
python3 -m pip install numpy opencv-contrib-python pyyaml matplotlib imageio-ffmpeg requests
python3 -m pip install -r src/qc/requirements.txt
python3 -m pip install -r label/requirements.txt
python3 -m pip install --user edge-tts
```

基准机实际包：`numpy==1.21.5`、`opencv-python-headless==4.13.0.92`、`Pillow==9.0.1`、`PyYAML==5.4.1`、`matplotlib==3.5.1`、`requests==2.25.1`、`scipy==1.8.0`。

## 3. NAS
先确定nas地址，下面以`192.168.50.177`为例


```bash
sudo mkdir -p /mnt/nas
sudo mkdir -p /etc/samba
sudo install -m 600 /dev/null /etc/samba/nas.cred
```

写`/etc/samba/nas.cred`：

```ini
username=ego_collection_admin
password=<NAS_PASSWORD>
```

设置权限
```bash
sudo chmod 600 /etc/samba/nas.cred
```

设置开机自动挂载，写`/etc/fstab`：

```fstab
//192.168.50.177/ego /mnt/nas cifs credentials=/etc/samba/nas.cred,vers=3.0,sec=ntlmssp,uid=1000,gid=1000,file_mode=0664,dir_mode=0775,iocharset=utf8,nofail,x-systemd.automount,nounix,noserverino 0 0
```

NAS 相关字段对应关系：

| 含义 | `/etc/fstab` | 后端启动`.env` | 采集程序启动`src/sync/config.json` |
| --- | --- | --- | --- |
| NAS 地址 | `//192.168.50.177/ego` | 不填 | `taskBackend.nas.sharePath` |
| NAS IP | `192.168.50.177` | 不填 | `taskBackend.nas.serverIp` |
| 共享名 | `ego` | 不填 | `taskBackend.nas.shareName` |
| 本机挂载点 | `/mnt/nas` | `ORBBEC_NAS_ROOT=/mnt/nas`，用于 URI 解析/旧 uploader 备用 | `taskBackend.nas.mountPath`，采集端实际写入 |
| URI 前缀 | 不填 | `ORBBEC_NAS_URI_PREFIX=nas://ego` | `taskBackend.nas.uriPrefix` |
| URI 到路径映射 | 不填 | `ORBBEC_NAS_MOUNTS_JSON={"nas://ego":"/mnt/nas"}` | 由 `uriPrefix` + `mountPath` 生成给 Label/QC |
| 凭据文件 | `credentials=/etc/samba/nas.cred` | 不填 | 不填 |

必须一致：

- `fstab` 的挂载点 = `.env` 的 `ORBBEC_NAS_ROOT` = `config.json` 的 `mountPath`。
- `.env` 的 `ORBBEC_NAS_URI_PREFIX` = `config.json` 的 `uriPrefix`。
- `.env` 的 `ORBBEC_NAS_MOUNTS_JSON` 必须把同一个 URI 前缀映射到同一个挂载点。
- `config.json` 的 `sharePath` 只给前端显示/传配置用；真正复制数据由采集程序的 `mountPath` 决定。
- 后端只接收 `episode_uri=nas://...` 并派发后续任务；不读取采集机 `/data/local`。

测试

```bash
sudo systemctl daemon-reload
sudo mount -a
findmnt /mnt/nas
df -hT /data /mnt/nas
```



## 4. 设备和编译

```bash
cd /home/ubuntu/demo/Orbbec_demo
sudo bash lib/OrbbecSDK_v2.7.2/shared/install_udev_rules.sh
for g in video render dialout plugdev; do getent group "$g" >/dev/null && sudo usermod -aG "$g" "$USER"; done
echo 128 | sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb
./build.sh
./build/orbbec_probe
```

基准机硬件：6 台 Orbbec Gemini 336L、PICO 4 Enterprise、`/dev/dri/renderD128`。新采集机使用 AMD 核显编码，不依赖独显。改机器时同步更新 `src/sync/config.json` 里的 Orbbec SN、鱼眼 `uniqueId`、触觉串口。

AMD 核显/VAAPI 编码检测：

```bash
ls -l /dev/dri
id | grep -E 'video|render'
ffmpeg -hide_banner -encoders | grep -E 'hevc_vaapi|h265_vaapi'
vainfo 2>/dev/null | grep -E 'HEVC|VAProfile'
```

没有 `vainfo` 时安装：

```bash
sudo apt install -y vainfo mesa-va-drivers
```

期望：

- 存在 `/dev/dri/renderD128` 或其它 `renderD*`。
- 当前用户在 `render` 组
- FFmpeg 能看到 `hevc_vaapi`。
- `vainfo` 能看到 HEVC profile。

`src/sync/config.json` 编码配置：

```json
"save": {
  "rgbEncoding": "h265",
  "h265EncoderMode": "hardware",
  "h265Codec": "hevc_vaapi",
  "h265HwDevice": "/dev/dri/renderD128"
}
```

如果新机器的 render 设备不是 `/dev/dri/renderD128`，把 `h265HwDevice` 改成实际路径。

## 5. 确定相机编号
测试相机连接性：
```bash
cd /home/ubuntu/demo/Orbbec_demo
bash build.sh
./build/orbbec_probe
```

获取物理编号：
```bash
cd /home/ubuntu/demo/Orbbec_demo
bash build.sh
./build/orbbec_probe | grep -E '^\[[0-9]+\]|serial_number|uid|connection'
```


`src/sync/config.json` 填法：

同步器主设备：
```json
"devices": [
  {
    "sn": "CPCBC5300065",
    "index": "00",
    "syncConfig": {"syncMode": "OB_MULTI_DEVICE_SYNC_MODE_PRIMARY"}
  },
```
同步器同步设备：
```json
  {
    "sn": "CPCBC53000M1",
    "index": "01",
    "syncConfig": {"syncMode": "OB_MULTI_DEVICE_SYNC_MODE_SECONDARY"}
  }
]
```

鱼眼相机编号用稳定路径确认：

```bash
ls -l /dev/v4l/by-id
v4l2-ctl --list-devices
```

写入 `fisheye.cameras[].uniqueId`，格式如：

```json
"uniqueId": "v4l:/dev/v4l/by-id/usb-DCX-260417-KTY_usb_camera1_01.00.00-video-index0"
```

触觉手套编号确定串口路径：

```bash
ls -l /dev/serial/by-id
```

示例输出：

```text
usb-1a86_USB_Single_Serial_5565053881-if00 -> ../../ttyUSB0
usb-1a86_USB_Single_Serial_5565053747-if00 -> ../../ttyUSB1
```

把 `/dev/serial/by-id/` 加到前面，写入 `touch.devices[].portPath`，例如：

```json
"touch": {
  "devices": [
    {
      "id": "left",
      "handSide": "left",
      "sensorType": 1,
      "portPath": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5565053881-if00"
    },
    {
      "id": "right",
      "handSide": "right",
      "sensorType": 2,
      "portPath": "/dev/serial/by-id/usb-1a86_USB_Single_Serial_5565053747-if00"
    }
  ]
}
```

左右手确认：只插左手跑一次 `ls -l /dev/serial/by-id`，记录出现的那条；只插右手再跑一次，记录另一条。


## 6. 后端

复制并修改后端配置：

```bash
cd /home/ubuntu/demo/Orbbec_demo
cp .env.example .env
```

`.env` 关键项：

```.env
# 后端监听地址。
# 后端与 Collection 在同一台机器时用 127.0.0.1；需要让局域网内其它机器访问时改为 0.0.0.0。
ORBBEC_TASK_BACKEND_HOST=127.0.0.1

# 后端监听端口。修改后，浏览器地址和所有采集机 config.json 中的 taskBackend.baseUrl 也要改成相同端口。
ORBBEC_TASK_BACKEND_PORT=8765

# 后端自己的持久化状态目录，保存 instance、任务进度和 workflow.sqlite3，不是采集数据目录。
# 正式使用后不要随意换目录或删除内容。
ORBBEC_TASK_BACKEND_DATA_ROOT=./task_backend_state_nas4


# 旧实现环境配置，必须设为 0
ORBBEC_NAS_ENABLED=0

# NAS 在“运行后端的这台机器”上的本地挂载点。详见NAS部分说明
# 必须与 /etc/fstab 的挂载点、采集端 config.json 的 taskBackend.nas.mountPath 一致。
ORBBEC_NAS_ROOT=/mnt/nas

# NAS 的逻辑 URI 前缀，不是网络地址。采集端会用它上报 nas://ego/<subject>/<task>/episode<N>。
# 必须与采集端 config.json 的 taskBackend.nas.uriPrefix 一致；确定后不要随意改名，建议末尾不加 /。
ORBBEC_NAS_URI_PREFIX=nas://ego

# 把逻辑 URI 前缀映射为后端机器上的本地挂载点，供 Label、QC 和后续任务解析 nas:// URI。
# 值必须是单行的合法 JSON；左侧与 ORBBEC_NAS_URI_PREFIX 一致，右侧与 ORBBEC_NAS_ROOT 一致。
ORBBEC_NAS_MOUNTS_JSON={"nas://ego":"/mnt/nas"}

# 兼容旧配置，当前代码已忽略该开关：上传成功后始终会创建 auto_label 任务。
ORBBEC_AUTO_LABEL_AFTER_UPLOAD=1

# tools/virtual_workflow/orbbec_virtual_workflow.py 使用的 NAS 本地挂载点和 URI 前缀。
# 若在本机运行该工具，分别保持与 ORBBEC_NAS_ROOT、ORBBEC_NAS_URI_PREFIX 一致；
# 如果不运行 virtual_workflow 工具，这两项不会改变后端服务行为，可以省略。
ORBBEC_WORKFLOW_NAS_ROOT=/mnt/nas
ORBBEC_WORKFLOW_NAS_URI_PREFIX=nas://ego
```

启动后端前检查配置：

```bash
findmnt /mnt/nas
test -w /mnt/nas && echo "NAS writable"
python3 -m json.tool <<< '{"nas://ego":"/mnt/nas"}'
```

启动

```bash
cd /home/ubuntu/demo/Orbbec_demo
python3 task_backend/server.py
```

打开 `http://127.0.0.1:8765/`，选择任务文件和 instance。正式采集时需要固定一个instance不可更改。否则记录会丢失

后端在另一台同局域网机器时：

```env
ORBBEC_TASK_BACKEND_HOST=0.0.0.0
ORBBEC_TASK_BACKEND_PORT=8765
```

采集机 `baseUrl` 改成：

```json
"baseUrl": "http://<后端机器IP>:8765"
```



## 7. Collection

确认 `src/sync/config.json`：

```json
"taskBackend": {
  "enabled": true,
  "baseUrl": "http://127.0.0.1:8765",
  "nas": {
    "enabled": true,
    "serverIp": "192.168.50.177",
    "shareName": "ego",
    "sharePath": "//192.168.50.177/ego",
    "mountPath": "/mnt/nas",
    "uriPrefix": "nas://ego"
  }
}
```

```bash
cd /home/ubuntu/demo/Orbbec_demo
pico
./run.sh
```

真实采集流：

```text
Collection 设置 subject/save root
-> Load Tasks
-> 选 task
-> Start reserve episode
-> 本地写 /data/local/<subject>/<task>/episode_<N>
-> Confirm
-> 采集程序后台复制到 /mnt/nas/<subject>/<task>/episode<N>
-> 上传成功后采集程序向后端确认 episode_uri=nas://ego/<subject>/<task>/episode<N>
-> 后端登记 uploaded + nas_episode
-> 后端排 auto_label / qc / manual correction 后续任务
```

后端视角：采集系统等价于直接写 NAS。本地 `/data/local` 只是采集稳定性缓冲。

## 8. 后端工作流


主链路：

```text
uploaded -> auto_label -> qc -> finalized
```

返修链路：

```text
qc failed -> manual_correction_pending
-> Label UI 完成人工 2D
-> segment mano_opt
-> 所有 segment mano_succeeded
-> finalized
```

任务管理：后端页面选择固定 task-file instance；`auto_label`、`mano_opt`、`qc`、`manual_segment` 阶段保持 lease enabled。

人工 QC：打开后端 episode/detail 或 QC frontend，检查 `nas://ego/<subject>/<task>/episode<N>` 下视频、深度、标注和 3D 结果；通过则进入 finalized，失败则生成需纠偏 segment。UUID 只保存在后端映射和 episode 内 `.orbbec_upload_manifest.json`。

人工标注纠偏：Label UI 租约 `manual_segment`，读取 NAS episode，写回 `manual_2d/segments/<segment_id>`；完成后后端排 segment `mano_opt`，所有 segment 3D 成功后合并最终 3D。
