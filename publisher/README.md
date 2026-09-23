# 采集、自动标注、质检纠偏与再标注异步流水线

NAS 后端发布流程与轻量质检子集的完整说明见 [NAS_PUBLISH_AND_QUALITY_CHECK.md](NAS_PUBLISH_AND_QUALITY_CHECK.md)。

本系统将群晖 NAS 中采集完成的标准 episode 经对象存储中转站发送到企业内网专用标注服务器。服务器异步完成手部 pose、关节点可见性和 ego 外参标注，再将联合结果回传 NAS；采集侧随后独立管理质检与人工纠偏，必要时仅上传 `manual_2d/segments/` 进行再标注。`hand_mask/`、`hand_detection/` 和其它中间结果始终留在标注服务器，不通过中转站回传。

## 系统流程图

```text
┌────────────────────────── 模块一：NAS 与采集主机 ──────────────────────────┐
│                                                                          │
│  [采集程序] → [NAS episode 文件树] → [固定 publish API]                  │
│                                           │                              │
│                                           ▼                              │
│  [统一 SQLite 状态中心] ←────────→ [nas-uploader serve]                  │
│   · episode 生命周期                 · 异步上传/下载                      │
│   · manual generation               · 校验/重试/ACK                      │
│   · quality_queue                   · 中转对象清理                        │
│          │                                                               │
│          ▼                                                               │
│  [质检与人工纠偏模块]                                                    │
│   · 内部自行管理领取、质检和人工标注状态                                 │
│   · 通过 → quality_passed                                                │
│   · 未通过 → 写 manual_2d → publish --manual-2d → correction_pending     │
│                                                                          │
│  [NAS Job Controller]                [monitor + health/report]           │
│   · 读取队列压力/调整计算实例          · 健康检查/邮件报警/日报            │
│   · 恢复远端常驻服务                  · 汇总 serve/SQLite/调度状态         │
└──────────────────────────────────────────────────────────────────────────┘
        │ ① 原始 episode                    ▲ ④ 首次联合标注结果
        │ ⑤ manual_2d                       ▲ ⑧ 稀疏再标注结果
        ▼                                   │
┌────────────────────────── 模块二：对象存储中转站 ─────────────────────────┐
│  [原始数据区]     payload + manifest + ready/ACK                          │
│  [首次结果区]     optimized_pose + joints_vis + ego_pose.json             │
│  [人工纠偏区]     manual_2d/segments，按 generation 隔离                  │
│  [再标注结果区]   仅目标帧 optimized_pose                                 │
│  [控制对象区]     ready / ACK / worker 心跳 / 队列压力快照                │
│                                                                          │
│  只负责可恢复中转；不执行标注计算，也不维护业务生命周期。                 │
└──────────────────────────────────────────────────────────────────────────┘
        │ ② 完整 episode / ⑥ 人工纠偏 generation
        ▼
┌────────────────────── 模块三：标注服务器 ─────────────────────┐
│                                                                          │
│  [data-transfer 常驻服务] → [统一标注状态库] → [episode/view 队列]        │
│   · 下载/校验/状态同步       · 传输生命周期       · 状态/租约/失败隔离     │
│   · 结果上传/ACK            · pose/ego 结果汇合                          │
│                                                                          │
│  首次自动标注（异步）：                                                  │
│    [Mask 分割：多单卡] → [关节点检测与可见性：单卡] → [Pose 优化：单卡]  │
│                                                            │             │
│    [Ego 外参：CPU 常驻] ────────────────────────────────────┤             │
│                                                            ▼             │
│                         [首次结果汇合：pose 与 ego 均完成]                │
│                                      │                                   │
│                                      └→ ③ 上传首次联合标注结果           │
│                                                                          │
│  人工纠偏再标注（异步）：                                                │
│    [correction_pending] → [人工纠偏优化：单卡，仅修改错误帧]             │
│                                      │                                   │
│                                      └→ ⑦ 上传稀疏再标注结果             │
│                                                                          │
│  控制流：data-transfer → 心跳/队列压力 → 中转站 → NAS Job Controller     │
│          NAS Job Controller → 创建/回收标注实例，恢复常驻服务            │
└──────────────────────────────────────────────────────────────────────────┘

完整闭环：
  采集 → 发布 → 中转 → 首次自动标注 → 结果回传 → 质检
       ├─ 通过 → quality_passed
       └─ 未通过 → 人工纠偏 → 中转 → 再标注 → 稀疏结果回传 → relabeled
```

编号 ①—④ 表示首次自动标注闭环，⑤—⑧ 表示人工纠偏再标注闭环；图底部的“控制流”描述实例调度、健康监测和进程恢复。中转站只保存可恢复传输所需的 payload、manifest、索引和 ACK，不承担标注计算或业务状态管理。

## 1. 文件、路径与所在环境

开始使用前，先区分采集主机、NAS、容器和专用标注服务器路径。相同数据在不同环境中的路径可能不同。

### 1.1 采集主机 Ubuntu

- `/mnt/nas`：NAS 共享目录在采集主机上的挂载点。采集程序把 episode 写到这里。
- `/home/ubuntu/WorkSpace/wuchao/upload`：NAS producer 源码和主机侧配置模板目录；当前不在这里运行 `serve`。
- `/home/ubuntu/WorkSpace/wuchao/upload/nas-config.yaml`：主机侧 NAS 容器运行配置模板，包括上传下载并发、健康检查和 Job 调度参数；修改后复制到 NAS。
- `/home/ubuntu/WorkSpace/wuchao/upload/email.yaml`：主机侧邮件配置模板。邮箱账号、SMTP 授权码、收件人和日报时间在这里填写，再复制到 NAS。
- `/home/ubuntu/.local/bin/coscli`：主机侧 COSCLI，仅用于手工单文件或普通文件夹传输。
- `/home/ubuntu/.cos.yaml`：主机侧 COSCLI 凭据配置。
- `~/.ssh/config` 中的 `synology`：采集主机连接 NAS 的 SSH 别名，使用 `nas-deploy` 公钥认证。

### 1.2 群晖 NAS 宿主机

- `/volume1/ego`：NAS 上的 episode 根目录，与采集主机 `/mnt/nas` 是同一份数据。
- `/volume1/ego/.nas-uploader/build`：容器镜像构建目录和 producer 代码。
- `/volume1/ego/.nas-uploader/build/compose.yaml`：`nas-uploader`、`nas-uploader-monitor` 和可选 `nas-job-controller` 的编排配置。
- `/volume1/ego/.nas-uploader/config/config.yaml`：NAS producer、并发、健康检查和创智 Job 调度配置。
- `/volume1/ego/.nas-uploader/config/email.yaml`：NAS 运行时邮件配置。权限为 `600`，由主机侧模板复制而来。
- `/volume1/ego/.nas-uploader/config/.cos.yaml`：NAS COSCLI 凭据配置。
- `/volume1/ego/.nas-uploader/state/episodes.sqlite3`：原始上传、结果下载、人工纠偏上传、质检子集、状态事件和日报发送状态的统一持久数据库。
- `/volume1/ego/.nas-uploader/state/transfer.lock`：publish、ready 和 cleanup 操作锁。
- `/volume1/ego/.nas-uploader/state/serve-health.json`：`serve` 每 60 秒更新的本地健康心跳。
- `/volume1/ego/.nas-uploader/state/job-controller.json`：Job 逻辑槽位、优先级切换阶段和最近完成记录；控制器重启后据此恢复，不能手工编辑。
- `/volume1/ego/.nas-uploader/state/job-status.json`：分割、检测、自动优化、人工纠偏、ego 外参压力、Inspire GPU Job 与 `wc-dev` 常驻服务的标准化健康快照，供 health、报警、日报和即时报告读取。
- `/volume1/ego/.nas-uploader/inspire-cli`：提供给 Job 控制器和即时报告使用的 Inspire CLI 环境。
- `/volume1/ego/.nas-uploader/inspire-home`：Inspire CLI 的持久账号与会话目录。
- `/usr/local/sbin/nas-uploader-publish`：固定 publish SSH API 包装器。
- `/usr/local/sbin/nas-uploader-check`：后端 A 查询待质检子集并同步质检生命周期的固定 SSH API 包装器。
- `/usr/local/sbin/nas-uploader-health`：固定 health SSH API 包装器。
- `/usr/local/sbin/nas-uploader-report`：固定即时邮件汇报 SSH API 包装器。

### 1.3 NAS 容器内部

- `nas-uploader`：常驻运行双向 `serve`，同时负责原始 episode 与人工纠偏结果上传，以及普通 pose/ego 或 shape-calibration 结果的下载、恢复、ACK 与 COS 清理。
- `nas-uploader-monitor`：独立健康监控进程，每 10 分钟检查一次并负责邮件。
- `nas-job-controller`：常驻控制进程；根据共享队列压力创建或回收单卡 GPU Job，并每 60 秒检查和恢复 `wc-dev` 中的 data-transfer 与 ego 外参 worker。
- `/mnt/nas`：容器内可写的 NAS episode 根目录，对应宿主机 `/volume1/ego`；结果下载器写 episode 下的 `optimized_pose/`、`joints_vis/<view>/` 与 `ego_pose.json`，人工纠偏上传器读取 `manual_2d/segments/`。
- `/app/config/config.yaml`：容器内只读的 producer 配置映射。
- `/app/config/email.yaml`：容器内只读的邮件配置映射。
- `/app/state`：容器内可写状态目录，对应 NAS 的 `.nas-uploader/state`。
- `/app/run.sh`：容器命令入口，支持 `serve`、`publish`、`health`、`monitor`、`report`、`list`、`job-controller` 和 `job-inspect`。

### 1.4 企业内网专用标注服务器（当前部署实例：`wc-dev`）

- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits`：接收模块项目目录。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits/scripts/data_transfer.sh`：创智接收器入口。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits/configs/data_transfer.yaml`：创智接收配置。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/data`：下载完成的 episode 根目录。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/transfer_state/episodes.sqlite3`：创智下载、标注、结果回传和人工纠偏接收生命周期的统一状态库。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/optimizer_state/queue.sqlite3`：pose 的分割/检测/优化与独立 ego 外参分支共用的 episode/view 队列及租约数据库；不与数据传输状态混用。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/optimizer_state/submissions/*.sqlite3`：旧版 submission 计算断点；新 worker 首次启动时只读迁移，原文件保留。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/transfer_state/data_transfer.log`：当前后台接收日志，最大 `10 MiB`。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/transfer_state/data_transfer.log.1` 至 `.3`：自动轮转的历史接收日志，最多保留 3 份。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/transfer_state/data_transfer.lock`：创智传输服务的进程锁；防止 NAS 控制器重复启动进程，不能手工删除。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits/configs/async_optimizer.yaml`：所有单卡角色 Job 共享的模型、优化、队列路径与租约配置。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits/scripts/run_async_optimizer.sh`：GPU Job 内部入口，控制器注入 `--instance` 与 `--role segment|detection|optimizer`。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits/ego_extrinsics/`：CPU ego 外参模块；`geometry.py` 只计算，`io.py` 负责 episode 文件读写，`pipeline.py` 编排 AprilTag 重建。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits/configs/ego_extrinsics.yaml`：外参输入文件、AprilTag、PnP、插值和租约配置。
- `/inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits/scripts/run_ego_extrinsics.sh`：`wc-dev` 中常驻 ego CPU worker 的内部入口；常驻模式日志自动轮转。
- `<episode>/ego_pose.json`：按参考帧逐帧保存的 ego 外参；同时明确保存 `T_ego_from_reference` 与逆变换 `T_reference_from_ego`。

### 1.5 COS 对象

- `cos://sii-transfer/episode-transfer/payload/`：标准 episode 文件中转目录。
- `cos://sii-transfer/episode-transfer/manifests/`：标准 episode manifest。
- `cos://sii-transfer/episode-transfer/control/ready.json`：NAS 发布的完整 episode 列表。
- `cos://sii-transfer/episode-transfer/control/acks.json`：创智发布的持久接收确认。
- `cos://sii-transfer/episode-transfer/manual_2d/`：按 episode 和人工纠偏 generation 隔离的 `segments/` payload 与 manifest。
- `cos://sii-transfer/episode-transfer/control/manual_2d/ready.json`：NAS 发布的待接收人工纠偏 generation 索引。
- `cos://sii-transfer/episode-transfer/control/manual_2d/acks.json`：创智发布的人工纠偏落盘确认，NAS 据此清理中转对象。
- `cos://sii-transfer/episode-transfer/control/workers/wc-dev.json`：创智接收器心跳。
- `cos://sii-transfer/episode-transfer/control/scheduler/status.json`：创智传输服务发布的 v2 共享队列快照，包括 active/unadmitted episode、各 view 状态及 `segment/detection/optimizer/ego` 的 pending/leased 数；NAS 控制器只消费未过期快照。
- `cos://sii-transfer/episode-transfer/results/`：按 episode 和 `generation` 隔离的 `optimized_pose/ + joints_vis/ + ego_pose.json` 联合 payload 与 manifest。
- `cos://sii-transfer/episode-transfer/control/results/ready.json`：创智发布的待回传结果索引。
- `cos://sii-transfer/episode-transfer/control/results/acks.json`：NAS 发布的结果落盘确认，创智据此清理结果中转对象。
- `cos://sii-transfer/manual/`：手工单文件或普通文件夹传输前缀，不进入标准 episode 协议。

## 2. 一条 episode 的首次标注生命周期

```text
采集完成
  → NAS pending/uploading/ready
  → COS 原始 episode 中转
  → 创智 downloading/available
  → pose GPU 流水线与 ego CPU 流水线异步领取（标定 episode 仅走分割/检测/shape-calibration）
  → labeling（两分支独立状态）
  → pose complete + ego complete 原子汇合
  → result_pending/result_uploading/result_ready
  → COS optimized_pose + joints_vis + ego_pose.json 中转（标定为 pose_mesh + pose_2d + shape + scale）
  → NAS result_downloading/labeled 或 shape_calibrated
  → 结果 ACK
  → 创智 result_cleaning/labeled 并清理 COS 结果对象
```

1. 采集程序把完整且不再修改的 episode 写入 NAS 的 `<subject>/<task>/<episode>`，随后通过固定 publish API 在 NAS SQLite 中登记 `pending`。publish 不计算哈希、不等待上传，通常约 1 秒返回。
2. NAS `serve.upload` 计算文件清单与 SHA-256，将原始目录和 manifest 上传到 COS，完成后置为 `ready` 并更新 `control/ready.json`。失败任务保留在 SQLite 中，由常驻服务重试。
3. 创智 data_transfer 下载并逐文件校验，完整落盘后置为 `available`，发布源 ACK，并把共享优化队列的阶段压力快照写入 COS。
4. NAS 收到匹配的源 ACK 后删除该 episode 的 COS 原始 payload/manifest，并把 NAS 生命周期置为 `cleaned`；NAS 原始目录不删除。
5. NAS `nas-job-controller` 按角色压力创建独立 GPU workload：每个组最多 3 个分割 Job、1 个检测 Job，自动 pose 优化器和人工纠偏优化器各最多 1 个 Job。ego 外参由 `CPU资源空间 / CPU资源-2` 中现有的 `wc-dev` 常驻 CPU worker 独立领取；它与 data-transfer 共用正确的交互式实例环境，但进程、租约和任务状态彼此独立。
6. 单卡 Job 以固定 role 启动。分割/检测领取最老 episode 的最老可执行 view；优化器只领取所有 view 已完成检测的最老 episode。领取使用 owner、随机 token、过期时间和续租，实例被抢占后任务会自动回到原阶段。
7. pose 分支依次完成分割、检测和优化，最终同步落盘 `optimized_pose/`。ego 分支从固定相机多视角 AprilTag 建图，再在鱼眼 ego 帧上执行 PnP；直接估计缺口用旋转 SLERP 与平移插值补齐，每一参考帧写入 `ego_pose.json`。两支分别记录 `pending/complete/failed`，一支失败不会终止另一支或后续 episode。
8. 只有 pose 与 ego 都为 `complete` 时，创智生命周期才原子进入 `result_pending`。两支都已终止且至少一支失败时才进入 `label_failed`；失败邮件会携带 `pose/...` 或 `ego/...` 阶段和错误摘要。
9. 创智 data_transfer 对普通 episode 将 `optimized_pose/`、`joints_vis/<view>/<frame>.npy` 和 `ego_pose.json` 写入 v2 manifest；对标定 episode 则写入 v4 manifest 中的 `pose_mesh.png`、`pose_2d.png`、`shape.npy` 和 `scale.npy`，状态依次进入 `result_uploading`、`result_ready`。
10. NAS `serve.download` 按 manifest 类型精确校验并落盘：普通结果写回原 episode 的 pose/ego 文件并进入 `labeled`；标定结果写入 `<subject>/shape_calibration_result/` 并进入 `shape_calibrated`，两者均发布 ACK。创智收到匹配 ACK 后清理 COS 结果 payload/manifest。

11. NAS 在同一事务中把 `labeled` episode 加入轻量 `quality_queue`。后端 A 查询后调用 `take`，NAS 将主状态改为 `quality_labeling` 并从子集删除；具体质检与人工标注调度完全由后端 A 管理。
12. 质检通过时，后端 A 回报 `passed`，NAS 状态进入 `quality_passed`。质检未通过时继续保持 `quality_labeling`；人工纠偏完整落盘后，后端 A 调用 `publish --manual-2d`，NAS 状态进入 `correction_pending` 并登记异步上传。

人工纠偏回传创智使用同一套常驻服务和同一份两侧 SQLite，但使用独立的 `manual_transfers` 表和 COS 控制索引。创智完整校验 `manual_2d/segments/` 后进入 `correction_pending`；独立单卡纠偏 Job 按最早 generation 领取任务，合并相邻错误帧并向两侧扩展正确 pose 锚点。每个 `.npy` 必须是 `(2,21,2)`，手性与关节顺序直接沿用 `hand_detection`；完整坐标对 `(-1,-1)` 作为不可见点写成置信度 0，不参与纠偏优化，其他负坐标直接报错。人工坐标只在内存中覆盖本轮优化输入，绝不改写 `hand_detection/`。优化器沿用自动 pose 的损失与参数，只更新人工文件对应的错误帧，窗口内正确帧保持不变。创智随后用 v3 manifest 仅回传这些帧的 `optimized_pose/<frame>.npy`，NAS 校验后原子覆盖对应帧并进入 `relabeled`，其余 pose、ego 外参和中间结果均不变。

源 episode 的不可变版本由传输层的 `episode_id + manifest_sha256` 保证，回传轮次由递增 `generation` 和 publish token 保证幂等。优化 worker 本身不解析源 manifest 或“版本号”，只读取 data_transfer 已确认完整的不可变目录和共享队列 checkpoint。同一路径下直接替换原始采集文件不会被识别为新版本；原始数据修订必须使用新的 episode ID，人工纠偏则使用独立的 `manual_2d` generation。

## 3. 命令在哪台主机执行

本文不要求登录 NAS 后手工操作。

- `ssh synology ...` 命令：在采集主机 Ubuntu 上执行，由采集主机远程调用 NAS 固定 API 或 Docker。
- `inspire notebook ...` 命令：在已安装并登录 Inspire CLI 的控制主机上执行，用于远程控制 `wc-dev`。
- 普通文件路径 `/mnt/nas/...` 和 `/home/ubuntu/...` 的命令：在采集主机 Ubuntu 上执行。

## 4. 启动和停止 NAS 服务

以下命令全部在采集主机执行。

### 启动 serve 和 monitor

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker start \
  nas-uploader nas-uploader-monitor
```

### 停止 serve 和 monitor

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker stop \
  nas-uploader-monitor nas-uploader
```

### 重启

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker restart \
  nas-uploader nas-uploader-monitor
```

### 查看容器状态

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker ps \
  --filter name=nas-uploader
```

### 查看 serve 日志

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker logs \
  --tail 100 -f nas-uploader
```

### 查看 monitor 日志

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker logs \
  --tail 100 -f nas-uploader-monitor
```

两个容器均使用 `restart: unless-stopped`。NAS 或 Docker 服务重启后会自动恢复；状态数据库位于 NAS 持久目录，不随容器删除。

启用 Job 调度后，第三个 `nas-job-controller` 容器同样使用 `restart: unless-stopped`。它的创建、启动、停止和查看命令见第 10 节；未完成 Job 调度环境配置时不要把 `job_scheduler.enabled` 改为 `true`。

## 5. 发布标准 episode 与人工纠偏结果

### 5.1 发布采集完成的标准 episode

episode ID 必须严格为三段相对路径：

```text
subject/task/episode
```

例如：

```text
采集主机目录：/mnt/nas/xjz/task12/episode-001
NAS 目录：     /volume1/ego/xjz/task12/episode-001
episode ID：   xjz/task12/episode-001
```

发布前必须保证目录已经采集完成，之后不再写入；目录内至少包含一个普通文件，且不包含符号链接文件。

在采集主机执行：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-publish xjz/task12/episode-001
```

`publish` 只完成路径校验和 NAS SQLite `pending` 登记，通常约 1 秒返回。SHA-256、COS 上传和失败恢复由常驻 `serve` 异步完成。

发布受试者手型标定 episode 时增加 `--shape-calibration`：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-publish --shape-calibration \
  xjz/hand_shape_calibration/episode-001
```

标定标志会进入不可变 manifest。创智仍执行全部视角的分割和检测，但跳过 ego 与普通 pose 结果分支；初始化使用零 MANO shape 和单位 scale，随后把整个 episode 共享的一组 shape/scale 作为可训练参数。完成后结果写入 `<data_root>/shapes/<subject>/shape.npy` 和同目录的 `scale.npy`，并生成 `pose_mesh.png`、`pose_2d.png` 两张 3×2 六视角组合图。四个文件经独立 generation 回传，NAS 在 `<subject>/shape_calibration_result/` 下创建目录并落盘，随后进入 `shape_calibrated`，不会进入质检队列。`--shape-calibration` 不能与 `--manual-2d` 同时使用。

重复发布行为：

- `pending`、`uploading`、`ready`：幂等返回，不增加重复任务；
- `cleaned`：输出 warning，退出码为 0，不重新上传；
- 已开始上传后内容变化：manifest 不一致时立即报错。

查看队列仍然从采集主机执行：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker exec \
  nas-uploader /app/run.sh list
```

状态含义：

| 状态 | 含义 |
| --- | --- |
| `pending` | publish 已登记，等待上传 |
| `uploading` | 正在上传或等待中断恢复 |
| `ready` | payload 与 manifest 已完整上传，等待创智 ACK |
| `cleaning` | 已收到匹配 ACK，正在清理 COS |
| `cleaned` | 原始 episode 的 COS 对象已清理，等待标注结果 |
| `result_downloading` | NAS 正在下载或恢复下载联合自动标注结果 |
| `labeled` | `optimized_pose/`、`joints_vis/` 与 `ego_pose.json` 均已在 NAS 完整校验并落盘，同时进入待质检子集 |
| `shape_calibrated` | 标定结果组合图、`shape.npy` 和 `scale.npy` 已落盘到 `<subject>/shape_calibration_result/`，不进入质检队列 |
| `quality_labeling` | 后端 A 已接管，正在管理质检以及可能发生的人工标注 |
| `quality_passed` | 后端 A 回报质检通过 |
| `correction_pending` | 人工纠偏已登记，等待上传、创智纠偏优化或稀疏结果回传 |
| `relabeled` | 纠偏后的 pose 帧已从创智回传并在 NAS 覆盖落盘 |

创智统一状态库中的主要状态：

| 状态 | 含义 |
| --- | --- |
| `downloading` | 正在从 COS 下载或覆盖恢复原始 episode |
| `available` | 原始 episode 已完整校验，等待加入共享优化队列 |
| `labeling` | pose 与 ego 两个独立分支至少一个仍在等待或执行 |
| `label_failed` | 两分支均已终止且至少一个失败；错误按 component 记录并已隔离 |
| `result_pending` | pose/ego 均完成，或标定参数与组合图已生成，等待 data_transfer 上传结果 |
| `result_uploading` | 正在上传或恢复上传本轮 `optimized_pose/ + joints_vis/ + ego_pose.json` |
| `result_ready` | 结果已完整上传，等待 NAS 落盘 ACK |
| `result_cleaning` | 已收到匹配的 NAS ACK，正在清理 COS 结果对象 |
| `labeled` | NAS 已确认普通结果落盘，COS 结果中转已清理 |
| `shape_calibrated` | NAS 已确认标定组合图、shape 和 scale 落盘，COS 结果中转已清理 |
| `correction_pending` | 人工纠偏文件已完整校验，等待纠偏 Job |
| `correction_optimizing` | 单卡纠偏 Job 持有租约并执行窗口优化 |
| `correction_failed` | 本轮纠偏数据或算法失败，错误已隔离记录 |
| `correction_result_pending/uploading/ready/cleaning` | 稀疏纠偏 pose 正在等待上传、上传、等待 NAS ACK 或清理 COS |
| `relabeled` | NAS 已确认稀疏纠偏 pose 落盘，COS 中转已清理 |

两侧状态列都不使用 SQLite 固定枚举约束，并保留每次转移事件。NAS 质检相关新流程固定为 `labeled -> quality_labeling -> quality_passed/relabeled`，不再生成 `quality_checked / reuploaded / rechecked`；历史记录不会被删除。每轮传输使用递增 `generation`，同一 publish token 重试幂等，同一 episode 不允许多个 generation 并行回传。

创智 `label_components` 为每个 `labeling` episode 维护 `pose_state` 与 `ego_state`，取值为 `pending / complete / failed`，并分别保存 fenced token、更新时间和错误。主状态不增加容易失控的组合枚举，而由这两个正交分支原子汇合；共享计算队列则使用 `POSE_DONE/POSE_FAILED` 与 `EGO_PENDING/EGO_RUNNING/EGO_DONE/EGO_FAILED` 保存各自断点和租约。

shape calibration episode 的 ego 分支登记为已跳过，优化器完成共享 shape/scale 与两张六视角组合图后通过独立 generation 回传；NAS 落盘到 `<subject>/shape_calibration_result/` 后使用 `shape_calibrated` 终态，不生成 `optimized_pose` 或 `ego_pose.json`。

### 5.2 发布人工纠偏结果

人工纠偏文件必须位于：

```text
<episode>/manual_2d/segments/<segment_id>/<camera>/<frame:05d>.npy
```

例如 `xjz/task12/episode-001/manual_2d/segments/segment-01/camera-01/00042.npy`。每一帧可以没有纠偏文件，因此文件数可以小于 episode 帧数；但待发布目录必须至少包含一个文件，且所有文件必须严格是三层相对结构和五位数字 `.npy` 文件名。

在能够 SSH 到 NAS 的主机执行：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-publish --manual-2d \
  xjz/task12/episode-001
```

该命令表示后端 A 已经完成人工标注：NAS 主状态更新为 `correction_pending`，同时登记人工纠偏上传请求。哈希、上传、失败重试、创智下载、纠偏优化、稀疏结果回传、ACK 和 COS 清理由常驻服务与独立纠偏 Job 异步完成；正常新流程要求 episode 已由后端 A 接管并处于 `quality_labeling`。

人工纠偏传输状态单独保存在同一 SQLite 的 `manual_transfers` 表：

| NAS 状态 | 含义 |
| --- | --- |
| `manual_pending` | 请求已登记，等待 NAS serve 上传 |
| `manual_uploading` | 正在上传，或等待中断恢复 |
| `manual_ready` | 本 generation 已完整进入 COS，等待创智 ACK |
| `manual_cleaning` | 已收到匹配 ACK，正在清理 COS 中转对象 |
| `manual_cleaned` | 创智已确认人工文件落盘，NAS 已清理这一方向的 COS 对象 |
| `correction_complete` | 创智纠偏结果已回传并在 NAS 对应帧覆盖落盘 |

| 创智状态 | 含义 |
| --- | --- |
| `manual_downloading` | 正在下载或覆盖恢复 `manual_2d/segments/` |
| `correction_pending` | 文件集合、大小和 SHA-256 已完整校验，等待纠偏 Job |
| `correction_running` | 纠偏 Job 持有带 fencing token 的可续租任务 |
| `correction_result_pending/uploading/ready/cleaning` | 稀疏 pose 结果的回传阶段 |
| `correction_complete` | NAS 已确认稀疏结果落盘 |

同一人工 generation 在活动状态下重复发布是幂等操作；上一 generation 已完成纠偏后再次发布，会创建递增的新 generation。若第一次上传已经开始后源文件发生变化，manifest 校验会报错，必须先处理该异常，系统不会把变化后的文件静默混入同一 generation。

### 5.3 后端 A 的质检生命周期接口

查询最早进入待质检子集的 episode：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-check list --limit 100
```

告知 NAS 后端 A 已经接管一个或多个 episode：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-check take \
  xjz/task12/episode-001 xjz/task12/episode-002
```

质检通过：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-check result passed \
  xjz/task12/episode-001
```

质检未通过、需要人工标注：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-check result needs-labeling \
  xjz/task12/episode-002
```

`quality_queue` 只保存尚未被后端 A 接管的 `labeled` episode。`take` 后该项从子集删除，NAS 不维护 worker、租约、超时回池和人工标注任务；这些状态全部由后端 A 持久化。人工标注完成后使用 5.2 节的 `publish --manual-2d` 接口。

## 6. 健康检查 API

在采集主机执行：

```bash
ssh synology sudo /usr/local/sbin/nas-uploader-health
```

该命令输出 JSON，包含：

- `serve`：双向 serve 心跳，以及独立的 `serve.upload`、`serve.download` 最近循环结果；两者都健康时 serve 才健康；
- `publish`：NAS 根目录可读性、状态目录可写性、锁目录可写性、SQLite 写事务能力；
- `download`：NAS 根目录可写性和结果落盘所需的 SQLite 能力；
- `email`：邮件配置、SMTP 网络连接和账号认证结果，不发送测试邮件；
- `scheduler`：控制器心跳、五个角色的 pending/leased 压力、active/unadmitted episode、等待/运行 workload 数、各 workload 角色、优先级和平台状态；调度未启用时明确显示 `disabled`；
- `recent_uploads`：最近 5 个进入 `ready` 状态的 episode ID 和上传完成时间；
- `recent_labels`：最近 5 个进入 `labeled` 状态的 episode ID 和时间；
- `healthy`：只有双向 `serve`、`publish`、`download`、`email` 和已启用的 Job 调度全部健康时才为 `true`。

`nas-uploader-monitor` 每 600 秒执行同一套检查。serve 每 60 秒写一次心跳，心跳超过 180 秒即判定为异常。`nas-job-controller` 每 60 秒探测一次 `wc-dev`；单次 Jupyter/TLS 控制链路失败只记录为 `degraded`，不会令 scheduler 异常或发送故障邮件。连续 3 次失败或超过 180 秒没有成功探测才确认异常；任意一次成功会立即清零计数。降级期间仍会校验 COS 调度快照的新鲜度，手动 report 会显示“探测降级”及连续失败次数。

健康事件表从本监控功能部署后开始记录，因此部署前已经完成的旧 episode 不会出现在 `recent_uploads` 或日报历史统计中。

## 7. 配置报警邮箱和日报

邮箱账号和收件人填写在采集主机的：

```text
/home/ubuntu/WorkSpace/wuchao/upload/email.yaml
```

文件格式：

```yaml
enabled: true

smtp:
  host: smtp.example.com
  port: 465
  security: ssl
  username: uploader@example.com
  password: SMTP授权码

sender: uploader@example.com
recipients:
  - receiver@example.com

daily_report:
  time: "09:00"
  timezone: Asia/Shanghai
```

字段说明：

- `enabled`：设为 `true` 才会发送报警、日报和即时报告；
- `smtp.host`：邮件服务商 SMTP 地址；
- `smtp.port`：`ssl` 通常使用服务商提供的 SSL 端口，`starttls` 使用 STARTTLS 端口；
- `smtp.security`：只允许 `ssl` 或 `starttls`；
- `smtp.username`：SMTP 登录账号；
- `smtp.password`：SMTP 授权码或应用密码，不建议填写网页登录密码；
- `sender`：发件邮箱，通常与 SMTP 账号相同；
- `recipients`：一个或多个收件邮箱；
- `daily_report.time`：日报发送时间，24 小时制 `HH:MM`；
- `daily_report.timezone`：日报日期边界和发送时间所使用的 IANA 时区。

填写完成后，在采集主机执行：

```bash
chmod 600 /home/ubuntu/WorkSpace/wuchao/upload/email.yaml

scp /home/ubuntu/WorkSpace/wuchao/upload/email.yaml \
  synology:/volume1/ego/.nas-uploader/config/email.yaml

ssh synology chmod 600 /volume1/ego/.nas-uploader/config/email.yaml

ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker restart \
  nas-uploader-monitor
```

然后执行健康检查：

```bash
ssh synology sudo /usr/local/sbin/nas-uploader-health
```

`email.healthy` 应为 `true`，信息应显示 SMTP 连接与认证成功。若邮件本身发生故障，系统无法通过同一个故障邮箱发送报警，但 health API 和 monitor 日志会明确显示邮件诊断失败。

## 8. 邮件行为与统计口径

### 异常报警

每 10 分钟检查一次。健康状态首次变为异常时，发送经过解析的简洁文本报告，不直接发送原始 JSON。后续连续检查仍异常时不会重复发送；只有至少检查到一次恢复正常后，下一次异常才会重新告警。最近一次正常/异常状态保存在现有 `episodes.sqlite3` 中，monitor 容器重启不会重置去重状态。若告警邮件发送失败，则不记录为已发送，下一轮会继续尝试。

### 每日固定日报

默认每天 `Asia/Shanghai 09:00` 发送昨日统计：

- 上传数量：昨日从 `uploading` 进入 `ready` 的 episode 数；
- 消费数量：昨日收到创智 ACK 并进入 `cleaned` 的 episode 数。
- 标注结果数量：昨日完整下载 `optimized_pose/ + joints_vis/ + ego_pose.json` 并进入 `labeled` 的 episode 数；
- 当前所有生命周期状态数量，包括下载、标注、回传和后续扩展状态；
- 当前各状态数量相较昨日同一日报周期快照的变化量；
- NAS 数据卷的总量、已用量、可用量和可用比例；
- 当前 active/unadmitted episode 数及分割、检测、自动优化、人工纠偏、ego 外参的 pending/leased 数；
- 当前等待/运行 Job 总数、上限、角色、LOW/HIGH 优先级、平台状态和 fallback 开关。

即使上传和消费数量都是 0，也会发送日报。日报使用简洁文本，不发送原始 JSON。日报成功发送后会在 SQLite 中记录日期，monitor 容器重启不会造成同一天重复发送。首次启用状态快照时没有昨日基准，变化量显示为 `N/A`，从下一个日报周期起显示正负变化值。

### 随时发送系统状态邮件

在采集主机执行：

```bash
ssh synology sudo /usr/local/sbin/nas-uploader-report
```

该命令现场查询 COS 角色压力快照和 Inspire Job 列表后，立即发送一封简洁文本邮件。主体与日报格式一致，但上传、消费和收到标注结果的数量统计“今天 00:00 至命令执行时刻”，episode 状态、变化量、NAS 余量、共享队列和 Job 状态也在执行时重新读取；邮件末尾追加当前 `serve.upload / serve.download / publish / download / email / scheduler` 监测结果。发送失败时命令返回非零，不会伪装成功。

## 9. 企业内网专用标注服务器双向传输服务

data_transfer 是独立的低优先级 CPU Notebook 服务，与优化 GPU Job 异步工作。NAS 的 `nas-job-controller` 每 60 秒检查一次 `wc-dev` 和接收进程；Notebook 为 `RUNNING` 但进程不存在时，控制器通过 Inspire CLI 的 Jupyter Terminal 通道请求后台启动，不依赖 SSH 连接缓存。一次 Jupyter/TLS 超时只会进入 `degraded`，连续 3 次或 180 秒未成功才确认异常。启动命令使用共享盘上的 `data_transfer.lock` 加独占锁，同一时刻最多存在一个服务进程；Notebook 被抢占后文件锁自动释放，新容器运行后无需人工进入容器，下一次检查会恢复服务。

以下命令都在安装了 Inspire CLI 的控制主机执行。检查接收器和日志轮转器；正常时应看到一个 `data_transfer.run` 和一个 `rotate_log.py`：

```bash
inspire notebook exec wc-dev \
  "pgrep -af 'data_transfer.run|rotate_log.py'"
```

查看 NAS 控制器最近一次检查结果；输出中的 `data_transfer_recovery` 会给出 Notebook 状态、检查时间、健康性和最近动作：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker exec \
  nas-job-controller /app/run.sh job-inspect
```

直接从控制主机查看最近日志：

```bash
inspire notebook exec wc-dev \
  "tail -n 100 /inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/transfer_state/data_transfer.log"
```

常驻入口自行写入并轮转 `data_transfer.log`：当前日志达到 `10 MiB` 后依次轮转为 `.1`、`.2`、`.3`，总日志空间上限约为 `40 MiB`。Notebook 被平台 kill 后，SQLite、COS 请求和共享盘文件仍保留，NAS 控制器会在新容器进入 `RUNNING` 后自动恢复进程。

查看创智侧可用 episode：

```bash
inspire notebook exec wc-dev \
  "cd /inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits && \
   bash scripts/data_transfer.sh --list-available"
```

服务下载原始 episode，逐文件校验大小与 SHA-256，写入 `available` 后才发布 ACK；同时接收 NAS 发布的人工纠偏 generation，并异步上传已经汇合的普通 pose/ego 结果或标定的 `pose_mesh + pose_2d + shape + scale` 结果。人工纠偏只替换目标 episode 下的 `manual_2d/segments/`；两种方向都在对端 ACK 后清理各自 COS 中转对象。Notebook 中断时 COS 数据与 SQLite 请求仍保留，新实例会覆盖未完成内容并重试。

共享队列开始处理 episode 时把生命周期置为 `labeling`。pose worker 与 ego CPU worker 分别使用 job ID 作为幂等 token登记本分支完成；只有 `optimized_pose/`、所有视角的 `joints_vis/` 和逐帧 `ego_pose.json` 都同步落盘后，事务汇合才写入 `result_pending`。实际 SHA-256 和 COS 上传由常驻 data_transfer 服务异步完成。

需要手工补发已经同时落盘的 `optimized_pose/`、`joints_vis/` 与 `ego_pose.json` 时，在安装 Inspire CLI 的控制主机执行：

```bash
inspire notebook exec wc-dev \
  "cd /inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits && \
   bash scripts/data_transfer.sh publish xjz/task12/episode-001"
```

手工 publish token 默认组合两个结果的修改时间；缺少任一结果会直接拒绝，未修改结果的重复请求幂等。正常标注流程不需要执行此命令。

## 10. 企业内网专用标注服务器自动标注 workload 调度

### 10.1 工作方式

`wc-dev` 的 data_transfer 发布 v2 调度快照，内容是共享队列中 active/unadmitted episode 数、view 状态数，以及 `segment / detection / optimizer / ego / correction` 五个角色的 pending/leased 数。NAS `nas-job-controller` 只读取未过期快照并查询 Inspire 当前 GPU Job；不再创建或分配 submission，也不再创建 HPC。

`max_groups=G` 时动态 GPU 逻辑槽位为 `wc-label-segment-1..3G`、`wc-label-detection-1..G`、`wc-label-optimizer-1` 和 `wc-label-correction-1`，GPU Job 硬上限是 `4G+2`。控制器按实时压力申请槽位：分割最多 `3G`，检测最多 `G`，自动 pose 优化器和人工纠偏优化器各最多 1 个；存在尚未入共享队列的 episode 时至少预留 1 个分割 Job 完成入队。ego 不占动态 Job 槽位，固定由 `wc-dev` 中唯一的 `wc-label-ego-1` worker 处理。

GPU分支由NAS容器执行结构化 Inspire CLI 调用：先读Job配额与所有合法计算组的实时余量，选择对应优先级余量最大的组，再通过 `inspire --json job create` 创建 `1,10,100` 单卡Job。ego分支不提交平台workload；控制器通过Notebook Terminal每60秒检查 `wc-dev`，以 `setsid + flock` 幂等恢复data-transfer和常驻ego worker。ego入口不设置CUDA设备，也不加载GPU模型。

每个GPU Job只加载自己角色的模型并循环领取共享队列任务。只要该角色仍有pending或leased工作，空闲worker不主动退出；该角色两者都为0时worker正常退出，控制器删除成功Job。ego worker使用常驻模式，队列为空时保留进程并低频轮询，因此新episode进入ego队列后无需等待新实例创建。新episode可以随时加入共享队列；调度分母直接来自当前episode/view表，无需扩展固定submission。

控制器同时计算平台上 `PENDING / CREATING / QUEUING / RUNNING` 的GPU Job与尚未出现在平台列表中的持久launch reservation，合计始终不超过 `4G+2`。`wc-dev` 是独立的既有交互式实例，不计入动态Job上限。同一轮可并行创建多个独立角色workload，且每个固定槽位最多只有一个实例。

GPU Job使用当前配置的优先级与fallback逻辑。`wc-dev`停止时，Notebook常驻服务健康检查失败并报警；实例恢复为 `RUNNING` 后，控制器在下一次检查中自动恢复data-transfer和ego worker。ego执行中断时租约到期，任务回到ego pending。单episode的算法/数据错误只把对应分支标为failed，并继续处理其它episode；GPU槽位或Notebook服务异常会进入健康检查与邮件报警。

### 10.2 配置

先在采集主机编辑 `/home/ubuntu/WorkSpace/wuchao/upload/nas-config.yaml` 的 `job_scheduler`，再复制到 NAS 的 `/volume1/ego/.nas-uploader/config/config.yaml`。关键字段如下：

- `enabled`：环境配置完成后设为 `true`；默认 `false`；
- `max_groups`：GPU流水组数，范围 `1..6`，默认 `1`；加上自动pose optimizer和纠偏optimizer后，动态GPU Job硬上限为 `4 * max_groups + 2`；
- `allow_high_priority_fallback`：是否允许 LOW 等待超时后切换 HIGH；
- `fallback_pending_seconds`：LOW 处于等待状态多久后触发 fallback；
- `submission_prefix`：历史字段名保留作为 Job 名称前缀，当前为 `wc-label`，不再表示 submission；
- `command`：Job 内执行的单角色入口，必须且只能各包含一次 `{instance}` 和 `{role}`；
- `data_transfer_recovery.enabled`：是否由常驻控制器恢复 `wc-dev` 的data-transfer与ego worker；
- `data_transfer_recovery.check_seconds`：恢复检查间隔，当前为 `60` 秒；
- `data_transfer_recovery.notebook / workspace`：被检查的 CPU Notebook，当前为 `wc-dev / CPU资源空间`；
- `data_transfer_recovery.transport`：远程命令通道；当前 `wc-dev` 禁止新建 SSH 连接，因此固定为 `jupyter`；
- `data_transfer_recovery.process_pattern / start_command`：确认并幂等启动data-transfer的精确模式与后台命令；
- `data_transfer_recovery.ego_process_pattern / ego_start_command`：确认并幂等启动常驻ego worker的精确模式与后台命令；
- `low_priority`：LOW 的 `workspace / project / quota / image` 和可选共享内存、最长时间，当前单卡规格为 `1,10,100`、共享内存 `100 GiB`，`priority` 固定为 `3`；
- `high_priority`：仅 fallback 开启时必填，使用相同单卡规格，`priority` 固定为 `6`。`group` 仅保留配置兼容，实际提交前会在所有支持该 quota 的计算组中选择实时余量最大的一个。

NAS 上还必须准备 `/volume1/ego/.nas-uploader/inspire-cli` 和 `/volume1/ego/.nas-uploader/inspire-home`，让容器内 `/opt/inspire-cli/.venv/bin/inspire` 能使用一个已登录且有权创建、停止和删除上述 Job、访问 `wc-dev` 的 Inspire 账号。`nas-job-controller` 通过 Compose 中的 `INSPIRE_*_PROXY=http://sii-proxy:7890` 访问 SII API 和 Jupyter Terminal。账号凭据不能写入镜像或提交到代码库。

GPU Job 的镜像和共享项目环境必须满足原异步 pipeline 的运行依赖，并能访问共享队列、模型与 TensorRT plan。至少应在正式启用前确认项目 Python 可以导入 `tensorrt`、`smplx`、`ultralytics`，且 `configs/async_optimizer.yaml` 中的 MANO、WiLoR、SAM/TensorRT 资产路径存在。模型初始化失败属于服务故障，不会把尚未领取的 episode 错标为 `label_failed`。

从旧版升级时，必须先停止并删除仍存在的平台 `wc-label-N` 四卡 Job，再启动新版控制器；控制器发现仍存在旧 Job 会明确报错，不会同时运行两套 pipeline。新版角色 worker 会把 `optimizer_state/submissions/*.sqlite3` 中的 view 数组和 init/pose 断点一次性导入 `optimizer_state/queue.sqlite3`，源数据库不删除；旧控制器 JSON 中已经不存在平台 Job 的槽位会记为 `MIGRATED` 后释放。

升级激活顺序固定为：先从主机停止 `nas-job-controller` 和 `nas-uploader-monitor`，再通过主机 Inspire CLI 停止并删除旧 `wc-label-N` 四卡 Job，然后结束 `wc-dev` 中旧 data-transfer 进程；最后复制新版 `nas-config.yaml` 并重建 `nas-uploader-monitor nas-job-controller`。新版控制器会在 1 分钟内恢复 `wc-dev` data-transfer；它发布 v2 快照后才会创建单卡角色 Job。不要让旧控制器消费 v2 快照，也不要让新控制器长期消费旧 v1 快照。

将配置复制到 NAS：

```bash
scp /home/ubuntu/WorkSpace/wuchao/upload/nas-config.yaml \
  synology:/volume1/ego/.nas-uploader/config/config.yaml
```

首次创建或代码更新后，从采集主机重建 monitor 和 Job 控制器：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker compose \
  -f /volume1/ego/.nas-uploader/build/compose.yaml \
  --profile job-scheduler up -d --build \
  nas-uploader-monitor nas-job-controller
```

此命令会启动真实调度；只有配置和 Inspire 账号都完成后才能执行。静态检查代码时不要运行该命令。

### 10.3 状态查看

从采集主机实时读取共享队列压力与 Job 状态，不创建或删除 Job：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker exec \
  nas-job-controller /app/run.sh job-inspect
```

日报、手动 report 和异常报警都会包含相同的角色压力/Job 摘要。手动 report 会先执行一次上述实时查询；定时邮件读取控制器按 `poll_seconds` 持续刷新的状态文件。

## 11. 手工传输单个文件

手工传输不进入 episode 状态、ready、ACK 和自动清理协议。以下命令均在采集主机执行。

上传：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log cp \
  /mnt/nas/path/to/example.bin \
  cos://sii-transfer/manual/example.bin \
  --process-log=false --fail-output=false
```

下载到采集主机指定目录：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log cp \
  cos://sii-transfer/manual/example.bin \
  /path/to/destination/example.bin \
  --process-log=false --fail-output=false
```

确认 SHA-256 后清理精确对象：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log rm \
  cos://sii-transfer/manual/example.bin \
  --force --fail-output=false
```

## 12. 手工传输普通文件夹

上传目录：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log sync \
  /mnt/nas/path/to/folder/ \
  cos://sii-transfer/manual/folder-name/ \
  --recursive --routines 3 --thread-num 8 \
  --process-log=false --fail-output=false
```

下载目录：

```bash
mkdir -p /path/to/destination/folder-name

/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log sync \
  cos://sii-transfer/manual/folder-name/ \
  /path/to/destination/folder-name/ \
  --recursive --routines 3 --thread-num 8 \
  --process-log=false --fail-output=false
```

确认完整后清理精确前缀：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log rm \
  cos://sii-transfer/manual/folder-name/ \
  --recursive --force --fail-output=false
```

禁止对 bucket 根目录或共享协议前缀执行递归删除。

## 13. 运维边界

- 不要在采集主机运行旧的 `./run.sh serve`；唯一 producer 是 NAS 的 `nas-uploader` 容器。
- `publish` 不等待创智 Notebook 在线。创智长期离线时，ready episode 会持续占用 COS，应监控容量和费用。
- 不要修改已经进入原始上传流程的采集数据；源数据修订使用新的 episode ID。`manual_2d/segments/` 通过 `--manual-2d` 作为独立 generation 发布，不会改变原始 manifest。
- 邮件配置和 COS 配置包含凭据，权限必须保持为 `600`，不得提交到代码库或写入日志。
- 健康检查只验证 SMTP 连接和认证，不发送测试邮件；使用 `nas-uploader-report` 验证实际投递。

## 14. 命令速查

以下命令均在采集主机或安装了 Inspire CLI 的控制主机执行，不需要登录 NAS 后手工操作。

### NAS 服务管理

启动 NAS 上传服务和健康监控服务：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker start \
  nas-uploader nas-uploader-monitor
```

停止 NAS 健康监控服务和上传服务：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker stop \
  nas-uploader-monitor nas-uploader
```

重启 NAS 上传服务和健康监控服务：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker restart \
  nas-uploader nas-uploader-monitor
```

查看两个 NAS 容器是否正在运行：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker ps \
  --filter name=nas-uploader
```

持续查看上传服务最近 100 行日志：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker logs \
  --tail 100 -f nas-uploader
```

持续查看健康监控服务最近 100 行日志：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker logs \
  --tail 100 -f nas-uploader-monitor
```

### 自动标注 workload 调度

将主机侧 Job 调度配置复制到 NAS；该文件会覆盖 NAS 当前运行配置：

```bash
scp /home/ubuntu/WorkSpace/wuchao/upload/nas-config.yaml \
  synology:/volume1/ego/.nas-uploader/config/config.yaml
```

首次创建或代码更新后，重建并启动 monitor 与 Job 控制器；该命令会开始真实调度：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker compose \
  -f /volume1/ego/.nas-uploader/build/compose.yaml \
  --profile job-scheduler up -d --build \
  nas-uploader-monitor nas-job-controller
```

停止 Job 控制器；已经提交到创智的平台 Job 不会被此命令删除：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker stop \
  nas-job-controller
```

重新启动已经创建的 Job 控制器容器：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker start \
  nas-job-controller
```

持续查看 Job 控制器最近 100 行日志：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker logs \
  --tail 100 -f nas-job-controller
```

实时查询共享队列压力与 Job 状态，不执行创建、停止或删除：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker exec \
  nas-job-controller /app/run.sh job-inspect
```

### Episode 发布与查询

将指定 episode 登记到 NAS 异步上传队列：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-publish xjz/task12/episode-001
```

将指定 episode 登记为手型标定任务：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-publish --shape-calibration \
  xjz/hand_shape_calibration/episode-001
```

将指定 episode 的 `manual_2d/segments/` 登记到 NAS 人工纠偏异步上传队列：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-publish --manual-2d \
  xjz/task12/episode-001
```

查看全部 episode 的主生命周期，以及当前人工纠偏 generation 的传输状态：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker exec \
  nas-uploader /app/run.sh list
```

查询等待后端 A 接管的 episode，按进入 `labeled` 的时间从早到晚返回：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-check list --limit 100
```

告知 NAS 后端 A 已接管指定 episode，将其置为 `quality_labeling` 并移出待办子集：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-check take \
  xjz/task12/episode-001 xjz/task12/episode-002
```

告知 NAS 指定 episode 质检通过，将其置为 `quality_passed`：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-check result passed \
  xjz/task12/episode-001
```

告知 NAS 指定 episode 质检未通过；NAS 保持 `quality_labeling`，人工标注调度由后端 A 继续管理：

```bash
ssh synology \
  sudo /usr/local/sbin/nas-uploader-check result needs-labeling \
  xjz/task12/episode-002
```

### 健康检查与邮件

检查双向 serve、publish、download、邮件、Job 调度及最近上传和标注记录：

```bash
ssh synology sudo /usr/local/sbin/nas-uploader-health
```

立即现场刷新角色压力/Job 状态，并发送今日实时统计、episode 状态变化、NAS 余量和系统健康结果：

```bash
ssh synology sudo /usr/local/sbin/nas-uploader-report
```

限制采集主机邮件配置文件只能由当前用户读写：

```bash
chmod 600 /home/ubuntu/WorkSpace/wuchao/upload/email.yaml
```

将采集主机邮件配置复制到 NAS：

```bash
scp /home/ubuntu/WorkSpace/wuchao/upload/email.yaml \
  synology:/volume1/ego/.nas-uploader/config/email.yaml
```

限制 NAS 邮件配置文件只能由文件所有者读写：

```bash
ssh synology chmod 600 /volume1/ego/.nas-uploader/config/email.yaml
```

重启监控容器，使新的邮件配置生效：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker restart \
  nas-uploader-monitor
```

### 创智接收器

检查 `wc-dev` 的接收进程和日志轮转器：

```bash
inspire notebook exec wc-dev \
  "pgrep -af 'data_transfer.run|rotate_log.py'"
```

查看NAS常驻控制器的角色压力、GPU Job以及 `wc-dev` 中data-transfer/ego worker的自动恢复状态；该命令只读，不会主动启动或删除Job：

```bash
ssh -t synology sudo /var/packages/ContainerManager/target/usr/bin/docker exec \
  nas-job-controller /app/run.sh job-inspect
```

`nas-job-controller` 每 60 秒检查一次。若 `wc-dev` 被抢占，控制器先等待新容器进入 `RUNNING`，然后在下一轮幂等恢复；不需要手工登录 Notebook 启动。若要长期停用自动恢复，应将 NAS 配置中的 `job_scheduler.data_transfer_recovery.enabled` 改为 `false` 后重建控制器，不能只结束创智进程，否则一分钟内会再次启动。

从控制主机查看接收器最近 100 行日志：

```bash
inspire notebook exec wc-dev \
  "tail -n 100 /inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/transfer_state/data_transfer.log"
```

查看创智侧已经完整下载并可用的 episode：

```bash
inspire notebook exec wc-dev \
  "cd /inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits && \
   bash scripts/data_transfer.sh --list-available"
```

手工把一个已经同步落盘的 `optimized_pose/ + joints_vis/ + ego_pose.json` 登记到创智异步回传队列；正常标注流程会自动执行，不需要手工调用：

```bash
inspire notebook exec wc-dev \
  "cd /inspire/hdd/project/feelingai/chenwenming-25012/jxs/wc/opt_toolkits && \
   bash scripts/data_transfer.sh publish xjz/task12/episode-001"
```

### 手工传输单个文件

将采集主机上的单个文件上传到 COS `manual` 前缀：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log cp \
  /mnt/nas/path/to/example.bin \
  cos://sii-transfer/manual/example.bin \
  --process-log=false --fail-output=false
```

将 COS 中的单个文件下载到采集主机指定路径：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log cp \
  cos://sii-transfer/manual/example.bin \
  /path/to/destination/example.bin \
  --process-log=false --fail-output=false
```

删除 COS 中已经确认不再需要的单个文件：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log rm \
  cos://sii-transfer/manual/example.bin \
  --force --fail-output=false
```

### 手工传输普通文件夹

将采集主机上的普通文件夹递归上传到 COS `manual` 前缀：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log sync \
  /mnt/nas/path/to/folder/ \
  cos://sii-transfer/manual/folder-name/ \
  --recursive --routines 3 --thread-num 8 \
  --process-log=false --fail-output=false
```

创建采集主机上的文件夹下载目标目录：

```bash
mkdir -p /path/to/destination/folder-name
```

将 COS 中的普通文件夹递归下载到采集主机：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log sync \
  cos://sii-transfer/manual/folder-name/ \
  /path/to/destination/folder-name/ \
  --recursive --routines 3 --thread-num 8 \
  --process-log=false --fail-output=false
```

删除 COS 中已经确认不再需要的普通文件夹前缀：

```bash
/home/ubuntu/.local/bin/coscli -c /home/ubuntu/.cos.yaml --disable-log rm \
  cos://sii-transfer/manual/folder-name/ \
  --recursive --force --fail-output=false
```
