# 外包浏览器分支：本地草稿、批量提交与兼容入口

启用 `qc_dispatch` 后，网页 QC 的解码、MANO 渲染和视频编码按 episode 分配到采集机；浏览器直接从采集机 HTTPS 取媒体，Label 计算继续留在后端。部署与网络要求见 [QC 采集端节点说明](qc_worker_deploy/README.md)。无空闲节点时排队，不回退到后端 QC 计算。

本目录已实现 **独立浏览器本地草稿/批量提交工作台**、原版 Label/QC 的浏览器兼容入口，以及独立视频播放的性能实验。
新工作台的启动、结果格式、断线恢复和适用边界见 [BROWSER.md](BROWSER.md)。
**正式外包方向选择「浏览器本地 UI + 后端计算 + 独立预览视频」**。
整页远程应用保留为过渡/兼容入口，不作为低带宽场景的性能承诺。

原生浏览器已支持多视角标注、来源对照、后端骨架/MANO/跟踪、分段质检播放和本机草稿/批量提交。
外包交付包启动入口为 `http://127.0.0.1:18885/`，启动器通过 HTTPS 直接连接实验室 `10.162.241.5:18884`，不依赖 SSH。打包与部署见 [VPN 直连说明](vendor_gateway/README.md)。原本机 `18882` SSH 验证入口保留；当前使用独立账号、Label/QC 角色及 Task 范围授权。企业 VPN 的浏览器直接信任的 HTTPS 域名入口和跨进程并发调度尚未部署。
原版交互修正与账号说明以 [BROWSER.md](BROWSER.md)、[AUTHENTICATION.md](AUTHENTICATION.md) 为准；下文 noVNC 令牌只适用于旧兼容工具，不适用于 18882 工作台。
`benchmark.html` 是媒体实验页，不能提交标注或 QC 结果；不要将它当成完整产品。
实测及边界见 [VALIDATION.md](VALIDATION.md)。

## 兼容入口的处理边界

```text
外包机浏览器 ← 加密隧道 ← noVNC / websockify ← 独立虚拟显示器
      鼠标/键盘 →                             ↓
                                原版 label.main / src.qc.main
                                  ↓              ↓
                              工作流后端       NAS、FFmpeg、MANO、CoTracker
```

原始 H.265、深度视频、模型、NAS 文件路径解析、渲染、标注文件写入均在 Linux 工作主机。
浏览器解压画面更新并绘制，仍会消耗少量 CPU/GPU；不是“零计算”。
保留原版的界面和业务行为，包括按帧保存、可见性、QC 范围、租约和提交。

每个命名会话有独立 DISPLAY、Xauthority、单实例锁、缓存、QC 进度和 worker ID。
桌面启动未设置 `ORBBEC_FRONTEND_RUNTIME_DIR` 时，原单实例规则不变。
同一账号下的进程隔离只避免会话相互干扰，**不是外包租户的 OS 安全隔离**。

## 启动

Ubuntu 工作主机需要 `Xvfb`、`xauth`、`xdpyinfo`、`openbox`、`x11vnc`、
`websockify`、noVNC 和原项目的 GUI/渲染环境。常规安装：

```bash
sudo apt-get install xvfb xauth x11-utils openbox x11vnc python3-websockify fonts-noto-cjk
```

本次验证使用官方 [noVNC v1.6.0](https://github.com/novnc/noVNC/releases/tag/v1.6.0)。
官方源码归档 `https://codeload.github.com/novnc/noVNC/tar.gz/refs/tags/v1.6.0`
的 SHA-256 为 `5066103959ef4e9b10f37e5a148627360dd8414e4cf8a7db92bdbd022e728aaa`。
不将第三方代码或模型提交进本仓库。

从仓库根目录，在 Linux 主机启动一个会话：

```bash
python3 -m remote_frontend qc \
  --name vendor-a-qc-01 --operator vendor-a-qc-01 \
  --config /absolute/path/qc-config.json \
  --display 91 --web-port 16091 --vnc-port 15991 \
  --novnc-dir /absolute/path/noVNC-1.6.0 \
  --python /absolute/path/gui-python \
  --gateway-python /usr/bin/python3
```

标注使用 `label`，另分配 `--name`、`--operator`、DISPLAY 和端口。
`--python` 应使用已有追踪环境；mesh 的 Python、工具库和模型路径沿用原配置。
Label 的原始 mesh 参数仍遵循 `label/mesh_cache.py` 所读取的 `src/sync/config.json`。
会话中的 operator 必须唯一；Label 界面保留原版可编辑身份输入，不能充当生产身份认证。

两个监听端口都固定为 `127.0.0.1`。本机建立 SSH 隧道：

```bash
ssh -N -L 16091:127.0.0.1:16091 ego
```

在工作主机读取 `~/.local/state/orbbec/remote/vendor-a-qc-01/connection.json`，
在本机浏览器打开其中的 `url`。该文件是私有会话连接凭据，不提交进版本库。
随机令牌在 URL fragment 中，通过 WebSocket 握手传递；只有令牌映射的 VNC 目标可连接。
VNC 本身仅对工作主机 loopback 开放，剪贴板双向同步关闭，文件传输未启用。
同一会话不允许第二个观察者抢占控制。

关闭标签页可在默认 120 秒内重连。超时后 gateway 退出，监督器先向 GUI 发送 SIGTERM，
调用原程序退出清理，再关闭传输和虚拟显示。缓存删除，进度保留。
网络故障导致释放失败时，强制结束不会伪造释放成功，后端任务等待租约到期。
日志在命名会话目录；缓存和 Xauthority 在私有 `/tmp/orbbec-remote-*` 中。
不应把该服务直接监听公网。面向真实供应商部署需要独立 OS 身份/容器、受限 NAS 权限、
工作流身份鉴权和 HTTPS/WSS 入口，不能仅依赖隐藏桌面菜单。

## 复现隔离验证

`validation.py` 创建独立工作流数据库和 QC/Label 测试任务，不连接生产后端，
也不启动 Publisher / NAS 状态同步。视频仅通过只读用途的链接引用原采集数据；
标定和 MANO 数据复制到测试目录，所有标注/QC 输出写测试目录。
原始视频链接不是 OS 层的只读挂载，生产部署仍需真正的权限隔离。

```bash
python3 -m remote_frontend.validation \
  --episode /mnt/nas/xjz/task15/episode1 \
  --state-dir /absolute/new/validation-directory --frames 180 --port 18765
```

使用生成的 `config.json` 启动兼容入口，完成真实解码和 MANO 渲染后，读取
`connection.json` 的 `runtime_dir`，在缓存仍存在时编码预览：

```bash
python3 -m remote_frontend.encode_preview \
  --cache /tmp/orbbec-remote-SESSION/cache/qc/pilot-qc \
  --output /absolute/path/media/preview.mp4 --frames 180
python3 -m remote_frontend.preview_server --preview /absolute/path/media/preview.mp4
```

另建立 `18880` 端口的 SSH 隧道，在本机浏览器打开：

- `http://127.0.0.1:18880/?mode=video`：普通网络。
- `http://127.0.0.1:18880/?mode=video&profile=wan`：每个媒体响应限速 4 Mbps，增加 100 ms 请求等待。

该限速是应用层实验，不模拟丢包、抖动、TCP 拥塞或多个并发 Range 共享链路。
首次点击“开始测量”会重新请求视频；播放结束记录显示帧率、掉帧和出画时间。
“跳至第 90 帧”同时记录目标是否已缓冲，避免把缓存内跳转冒充网络跳转延迟。
预览服务只提供固定派生视频和实验页，不提供 NAS 路径或任意文件下载。

如需记录远程画面流量，将 `benchmark.html` 复制到 noVNC 的静态资源目录，
打开 `/benchmark.html#token=连接URL中的令牌`。开始测量后在原版 QC 点击播放。
页面的更新计数是 framebuffer 更新次数，**不代表原视频帧率**。

## 正式浏览器分支的设计决定

1. 按现有 Label/QC 的布局和操作语义实现本地 DOM/Canvas UI；任务选择、按钮、进度条、
   拖点和可见性切换本地立即反馈。不要将这些交互锁在视频/远程桌面的往返链路上。
2. 后端完成原始解码、同步、MANO 投影和画面合成。已录制 Episode 优先缓存派生 H.264
   预览，支持 Range/短分片预取。真正随交互实时变化的服务端渲染再评估 WebRTC，
   不把点播全部变成实时推流。
3. 播放可以降低预览分辨率/码率，慢网络时降低清晰度或明确缓冲；不能承诺任意网络下 30 FPS。
   暂停、逐帧和保存时必须获取精确原始帧对应的高清静态图及关节点。
4. 编辑只更新本机草稿；明确提交时携带会话、版本、提交编号及所有帧/相机结果。
   后端校验租约、相机、帧范围及版本；重复提交幂等，旧画面回复不覆盖新帧。
   浏览器 `video.currentTime` 只用于预览定位，不能独自决定标注保存的源帧编号。
5. 拖点的本地反馈与“后端已保存”分开显示。断网保留待提交编辑，恢复后校验租约和版本，
   不把未确认的数据显示为已提交。视图缩放不改变原始像素坐标。
6. 缓存键包含 Episode 内容/标定/MANO 版本；会话复用只读派生媒体，每个 worker 保有自己的
   写入状态。限制同时渲染/追踪任务数量，再按实测 CPU/GPU/内存决定每机并发数。

建议以 30 FPS 视频、拖点本地反馈 p95 < 50 ms 为设计目标，而非当前已保证指标。
正式验收应补充：长 Episode、4/8/20 Mbps、50/100/200 ms RTT、丢包、低配设备、
同时多人作业、冷跳帧 p95、高清截图可辨识度、错帧/乱序/断线恢复和全量 UI 功能对照。
