# QC 采集端计算节点

网页继续只登录现有工作台。后端负责账号、工作流租约、episode 调度、状态和结果提交；采集端完成原视频解码、MANO 渲染、分片编码和媒体读取。Label 的 `BrowserMedia`、`BrowserCompute` 和任务分配不变。

启用 `qc_dispatch` 后，所有网页 QC 会话（含旧会话）都走采集节点，后端不复用本地 QC 缓存，也不在 worker 不可用时回退到本地计算。后端仍需要读取 NAS 元数据并传输压缩输入文件，所以不是零 CPU / 零网络占用。

## 协议与生命周期

- 后端按 episode ID + 已有源版本生成计算编号；同一版本的观看者共享一个分配。
- worker 每 2 秒通过 HTTPS 携带独立集群凭据上报状态并领取分配。该凭据与 Tailscale 入网 Auth Key 完全独立；不能把入网 key 当作应用层密码。
- 每个 worker 同时计算一个 episode。编码完成后立即可以接下一项，同时继续提供先前视频。没有空闲 worker 时排队。
- worker 心跳 30 秒失联、进程启动身份变化或后端重启后，分配的 generation 更新。迟到结果不能覆盖新分配；旧 worker 在失去租约后停止计算和媒体服务。
- 网页每 2 秒更新状态，已完成的视频也续期。观看引用超过 180 秒未更新会释放；最后一个观看者离开后，worker 停止计算并清理本次输入/派生缓存。
- 直接媒体 URL 带绑定计算编号和 generation 的短时 HMAC 令牌。worker 同时检查在线租约和资源范围。前端不拿到集群凭据、NAS 路径或入网 key。
- CORS 与 CSP 只允许配置中列出的 HTTPS origin；媒体支持 Range。重分配后网页重建播放器，Label URL 不变。
- 后端调度状态原子落盘到 `session/dispatch/qc-dispatch.json`；业务 QC 结果仍只由原后端提交。

## 后端配置

在现有 `browser_server --config` 文件中增加，保留已有 Label / NAS 参数：

```json
{
  "qc_dispatch": {
    "secret_file": "/absolute/private/worker.secret",
    "workers": {
      "capture-01": "https://CAPTURE-NAME.TAILNET.ts.net"
    },
    "heartbeat_timeout": 30,
    "viewer_ttl": 180
  }
}
```

凭据使用随机生成的至少 32 字符内容，文件权限 0600。新 worker ID 和 origin 必须由管理员加入此列表；不信任客户端自报任意媒体域名。修改列表后重启现有网页后端，不另起一套业务后端。

## 采集机部署（Ubuntu / Python 3.10）

1. 用管理员提供的 Tailscale Auth Key 入网，完成 `tailscale status` 检查。Auth Key 仅在部署时安全输入，不写入代码、日志、服务配置或命令行参数。已入网的设备不需重复入网。
2. 在独立目录（示例 `/home/user/qc-worker`）部署代码 `remote_frontend/`、`label/`、`mano/`、`src/qc/`、`task_backend/`，不覆盖采集软件目录。无需下载 Label 跟踪模型。
3. 创建独立 venv，不共享采集软件的 Python 包。安装 CPU PyTorch 和本目录依赖：

```bash
python3 -m venv /home/user/qc-worker/venv
/home/user/qc-worker/venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
/home/user/qc-worker/venv/bin/pip install -r requirements.txt
/home/user/qc-worker/venv/bin/pip check
```

4. 复制已有且获授权的 `ckpt/mano/`、`optimizer/mano_wrapper.py`、`optimizer/rotation.py` 到专用 `toolkit/`。专用 toolkit 的 `optimizer/__init__.py` 留空，避免加载完整优化训练环境；不要修改原模型工具库。
5. 参考 `config.example.json` 设置 worker ID、目录、后端 origin、允许的网页 origin，安装管理员发放的独立 worker.secret。
6. `tailscale serve --bg --yes 18900` 为本机 loopback 媒体服务提供 tailnet 内 HTTPS。首次申请证书需等待成功。
7. 安装并按实际目录调整本目录的用户服务：

```bash
mkdir -p ~/.config/systemd/user
cp orbbec-qc-worker.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now orbbec-qc-worker.service
```

用户服务随该用户登录启动。若需注销后或无人登录时也运行，由管理员执行 `sudo loginctl enable-linger user`。Tailscale 系统服务独立运行。

## 网络要求

必须在 tailnet 访问规则中分别允许：

- 采集节点 → 工作台 TCP 443，用于拉取分配和压缩输入。
- 获授权的网页前端 → 采集节点 TCP 443，用于直接读取视频及逐帧图。

仅允许前端访问原工作台 443 的旧策略不足以支持此架构。不要因此开放采集机 SSH、后端管理端口或全部端口。每加入一台采集机，需要同时登记其 worker origin 和相应的网络规则。Tailscale 入网 key 本身不会自动赋予这些访问权限。

## 采集优先与运维

示例值：每机 1 个 episode，子进程限制在 8 个逻辑核，2 个 mesh worker，CPUWeight/IOWeight=10、Nice=10、MemoryHigh=6 GB、MemoryMax=8 GB。进程级 CPU affinity 同时约束 ffmpeg、Mesa 和 BLAS 子进程。无 NVIDIA 时使用 CPU H.264 编码；保留 `detail960` 画面布局。

空闲磁盘不足 30 GB、可用内存不足 4 GB，或一分钟负载超过“总逻辑核数减 QC 核数”（32 核示例为 24）时不领新任务；计算期间保留 30 GB 磁盘余量。输入包有 20 GB 上限，并校验归档路径、文件类型和解包大小。建立 `state/pause` 可停止领取新任务，删除后恢复；这不打断已有任务。

```bash
systemctl --user status orbbec-qc-worker
journalctl --user -u orbbec-qc-worker
systemctl --user restart orbbec-qc-worker
```

单任务的 `state/<id>-<generation>/status.json`、`worker.log` 提供诊断，失败日志保留在 `state/errors/`。源视频和业务结果不会被修改。资源限额不能替代真实采集压测；还需考虑 USB 配置、相机数量、编码方式及 I/O 性能。

## 验证与回滚

```bash
python3 -m unittest tests.test_qc_dispatch
node tests/browser_media_routing.mjs
node tests/browser_player_downloads.mjs
node tests/browser_player_pause.mjs
node tests/browser_player_preparation.mjs
node tests/browser_label_prefetch.mjs
```

恢复旧代码和旧配置可回滚架构，但旧版本会重新在后端运行 QC。生产故障时优先暂停 QC 领取或恢复 worker，不应悄悄移回后端计算。现有 NAS、Label、账户和工作流数据库不需要迁移。
