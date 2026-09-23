# 浏览器 Label / QC 分支

启用 `qc_dispatch` 后，网页 QC 的解码、MANO 渲染和视频编码按 episode 分配到采集机；浏览器直接从采集机 HTTPS 取媒体，Label 计算继续留在后端。部署与网络要求见 [QC 采集端节点说明](qc_worker_deploy/README.md)。无空闲节点时排队，不回退到后端 QC 计算。

本分支以原 `label/app.py`、`label/canvas_view.py`、`src/qc/app.py` 为操作规范。
任务层级、按钮顺序、画布手势、确认、播放、坏帧、返回和提交遵循原桌面程序。
不得为了网页实现另设一套业务流程。原 Label/QC 的启动、代码、配置及 API 保持独立。

## 原版交互

Label 保留双栏 Task / Episode 选择、单机位默认视图及底部操作栏。左侧默认显示该 Episode 的所有标注区间、帧范围、主要错误机位和各区间确认进度；可独立选择区间。逐帧、时间轴和跟踪限于所选区间，末帧确认后停留，由用户选择下一区间。相邻或重叠的 QC 区间不合并。
数字 0 为五机位加 Ego 总览，数字键及上一/下一机位切换单机位；总览只读。存在 Ego RGB 的 Episode 将 Ego 作为正常可编辑机位，支持关节点、可见性、跟踪和最终保存。Ego RGB 按同步时间戳映射到参考帧，骨架投影使用鱼眼内参及逐帧外参。原始视角、含可见性的原始视角、修改后视角按原顺序轮换。
关节拖动、双击图像关节切换可见性、手型图单击切换可见性、双击选择跟踪、右键选择和右键重新定位、
长按 380 ms 框选切换可见性、滚轮缩放、右键平移、撤销、忽略视角沿用原语义。
Show Skeleton 使用当前视角来源重建；Show MANO 使用原 MANO。重建、渲染、CoTracker 均在服务器运行。

“确认并继续”保存当前帧所有机位的**修改后数据**到本机确认快照，包括当前正在查看原始预览的情况。
确认后在当前区间内跟踪到下一帧，并保留当前相机画布的缩放与平移；只有跟踪实际更新下一帧时才取消该帧的确认。显示 Skeleton 时不能确认。
所有区间的全部帧确认后才可提交。已有草稿加入 Ego 时保留原有修改，清除旧确认进度以便重新核对新增机位；升级前已固定的待提交正文保持原样重试。返回任务先询问是否放弃未确认修改，并释放任务。

QC 保留 Task → Episode → 后端准备 → 六视图质检的步骤。
播放时禁用逐帧、十帧及“该帧不通过”；暂停后进入独立坏帧模式，起止点初始为空。
该模式最多回看进入点之前 9 帧，手部 Pose 和 EgoPose 分别确认，按原 5 帧间隔规则合并。
撤销回到进入坏帧模式时的位置。实际播放到末帧后允许提交，随后回看不撤销完成状态；
仅跳到末帧不算完成，不要求逐帧勾选。Episode 异常经原有确认后直接提交。
返回 Episode 列表释放租约并保留本机进度，再进入时重新领取并恢复。

网页账号栏用于登录和权限管理，工作流内操作员由登录身份决定，不能自行冒用另一个 worker。
浏览器不提供桌面调试用途的本机 JSONL 路径选择；外包人员通过已分配的后端任务进入。

## 计算和草稿

原始 H.265/深度解码、时间同步、MANO 投影、三维重建、跟踪及预览编码在实验室主机。
浏览器仍需解码派生 JPEG/H.264 并绘制轻量关节点和 UI，这是普通媒体显示所需的计算。
QC 每 90 帧生成一段 H.264 视频（1536×1728、30 FPS、目标 6 Mbps）。同一视频帧上下两平面
分别保存六机位原图及 MANO 合成图；浏览器裁切同一解码帧并本地混合，不依赖两路播放器同步。
不透明度 0% 显示原图，100% 显示原 MANO 合成效果；滑块只保存到本机草稿，不改变提交内容。
暂停和逐帧直接显示视频帧。源帧由固定帧率的顺序索引映射回 manifest.frames，显示目标帧前禁用坏帧起止点。
支持 requestVideoFrameCallback 的浏览器确认实际呈现帧；兼容回退使用完成 seek 后的媒体时间。
“高清检查”是主动读取原始分辨率 JPEG 与 MANO 合成 JPEG 的入口，同样支持透明度；播放时退出高清检查。
预览缩小且有重编码损失，不能把预览当作原始分辨率数据。

MSE 前方预取 60 秒、保留后方 15 秒，按 3 秒片段对齐；窗口边界最多多留一个片段。
播放与耗尽后的恢复要求前方至少 6 秒，靠近片尾按剩余时长计算。冷跳转先取目标片段，后续取前方及回看片段。
不支持 MSE 时等待完整 MP4，缓冲范围由浏览器管理，不能承诺相同的 60/15 秒窗口。
MSE 视频缓冲是当前播放器内存，退出时释放；视频没有整段写入 IndexedDB。

Label 当前机位优先预取后 20 帧、前 2 帧，随后其他机位当前帧和后 2 帧；总览预取后 4 帧。
后台最多 2 路预取，全部图片请求最多 4 路；切换帧/机位取消旧队列。
图片以 Blob 存入 IndexedDB，按最近访问淘汰，目标上限 512 MiB；配额失败时降低目标，继续在线显示。
对象 URL 上限 64 个，退出/登出释放。数据库升级仅清理旧版无容量索引的图片缓存，保留 drafts。
这些容量是应用目标，浏览器仍可能因磁盘/内存压力提前回收缓存，不能保证断网时所有帧可用。

修改和已确认快照自动保存在 IndexedDB；点击最终提交才上传结果 JSON，视频不回传。
拖动不需要等待网络，计算请求及未缓存媒体仍需要后端。当前页面断网后可继续处理已缓存内容；
重新打开或刷新页面需要联网核验登录，不能用离线缓存绕过身份认证。
本机草稿按操作员筛选，浏览器编辑锁阻止同一草稿同时编辑。清理网站数据会删除草稿；
同一操作系统账号能读取浏览器本地文件，IndexedDB 不构成加密隔离，应使用个人系统/浏览器配置文件。

提交前持久化固定提交编号与正文，失败后重试同一请求；收到后端回执才标记成功。
提交不明时锁定这份结果，避免重试不同内容。服务器检查账号权限、任务归属、租约、源版本、
帧/机位集合、数组尺寸、有限坐标、确认进度。重复提交返回原回执。
释放后恢复必须重新取得该任务；已被别人接走或源版本变化时不能覆盖，旧草稿保留为冲突记录。

## 与原系统隔离

服务仅为现有数据库增加 `browser_sessions`、`browser_receipts`，不调用 `WorkflowStore.initialize()`，
不重置流程开关、索引或 outbox，不启动上传、Publisher、NAS 同步和下游 worker。
领取和提交通过现有 `JobService` 执行，与桌面使用同一工作队列和租约规则。
账号库单独位于 state-dir，不改原工作流账号或桌面配置。

Label 输出仍为 `manual_2d/segments/<job>/<camera>/<frame>.npy` 和独立可见性目录。
旧目标仓库不支持独立可见性 artifact 时，兼容隐藏坐标 `(-1,-1)` 并保留可见性旁文件。
QC 沿用 `qc/qc_report.json` 与 `ego/ego_pose_qc.json`，Ego 范围不混入手部修复段。
文件写入或业务提交抛出异常会恢复文件并回滚数据库；文件系统与 SQLite 之间尚无断电发布日志，
不能保证进程被强杀或断电时跨介质原子提交。

## 启动与登录

```bash
python3 -m remote_frontend.browser_server \
  --database /absolute/path/workflow.sqlite3 \
  --config /absolute/path/config.json \
  --state-dir /absolute/path/browser-state \
  --operator bootstrap-legacy-worker \
  --port 18882
```

使用已有 numpy、Pillow、FFmpeg 环境；`nas_mounts`、`mesh_renderer_python`、`mano_toolkit_root`、
`mano_model_dir`、`tracking_python` 为服务器实际路径。现有工作流数据库必须存在，服务不迁移 schema。
`--operator` 只绑定首次创建的管理员身份，便于接续旧草稿；后续用户各有独立不可编辑的 operator。
同一个服务可管理多个账号，不再每个操作员分发秘密链接。
原后端已注册账号通过 `backend_accounts_file` 关联并授予外包权限，继续使用原账号密码；具体配置见 AUTHENTICATION.md。

服务只监听 127.0.0.1。当前 VPN 验证沿用本机 SSH 转发：

```bash
ssh -N -L 18882:127.0.0.1:18882 ubuntu@LAB_HOST
```

直接访问 `http://127.0.0.1:18882/`，任何浏览器先显示账号登录页。
首次管理员临时密码写入 state-dir 的 `initial-admin.json`（0600），首次登录必须修改。
旧 `#key`、共享访问密钥和旧 cookie 不再有登录权限。不要将状态目录或临时密码提交到 Git。
正式供应商入口按用户选择使用企业 VPN + 企业 HTTPS 反向代理，所有人使用同一个普通地址、各自账号。
代理配置与限制见 [AUTHENTICATION.md](AUTHENTICATION.md)。当前本机地址不是供应商通用部署地址。

## 验证和边界

```bash
python3 -m unittest tests.test_browser_accounts tests.test_browser_batch \
  tests.test_browser_parity tests.test_browser_media_layers tests.test_remote_frontend \
  tests.test_frontend_lifecycle tests.test_label_client_smoke \
  tests.test_qc_worker_smoke tests.test_qc_playback_completion
```

浏览器状态对照直接调用原 QC 方法。实际视频、交互、恢复和提交在独立 NAS 副本及测试库验证，
不提交真实业务任务。记录见 [VALIDATION.md](VALIDATION.md)。
当前限制同时准备两个媒体任务、执行两个计算；仍需长视频、低配 Safari、丢包及多人并发验收。
短片验证不构成稳定帧率或延迟 SLA。字体和窗口装饰由浏览器/操作系统绘制，尚未完成逐像素截图回归。
