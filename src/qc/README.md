# 人工 QC Worker

这是 Workflow `qc` 阶段的人工质检前端，只检查 episode-level MANO 3D 结果是否可接受。

## 启动

在 `Orbbec` 目录下执行：

```bash
python3 -m src.qc.main
```

## .env

QC Worker 读取后台共享的 NAS mount 配置和自己的 `QC_*` 配置，常用项：

```bash
ORBBEC_NAS_MOUNTS_JSON={"nas://ego":"/mnt/nas"}
QC_BACKEND_URL=http://127.0.0.1:8765
QC_SAMPLE_INTERVAL=10
QC_DEFAULT_LEASE_MINUTES=10
QC_CRASH_LEASE_EXTENSION_MINUTES=10
QC_TMP_DIR=./tmp
QC_STATE_DIR=./qc_state
QC_WORKER_MACHINE_ID=qc_machine_01
QC_RANGE_MERGE_GAP_FRAMES=5
```

## 行为

- Task / Episode 列表来自后端 `qc` stage 的可租 job。
- 双击 Episode 后才会正式租借。
- 本机保留的有效进度会合并显示，并显示剩余租期倒计时。
- RGB H.265 会解码到 `QC_TMP_DIR/<episode_id>/<camera>/<frame>.png`。
- QC 提交会先写 `<episode>/qc/qc_report.json`，再调用后端 complete。
- “Episode 异常”提交为 `result_type=bad_episode`，不会创建人工返修 segment。
