# 人工 QC Worker

这是 Workflow `qc` 阶段的人工质检前端，只检查 episode-level MANO 3D 结果是否可接受。

## 启动

在 `Orbbec` 目录下执行：

```bash
python3 -m src.qc.main --config /tmp/orbbec_qc_frontend_config.json
```

## 配置

主程序点击 `QC` 时会根据 `src/sync/config.json` 里的 `frontends.qc` 和 NAS 配置生成启动配置，并通过
`--config` 传给 QC Worker。直接本地调试时也可以手动提供同样结构的 JSON：

```json
{
  "backend_url": "http://127.0.0.1:8765",
  "operator_id": "qc_operator_01",
  "worker_machine_id": "qc_machine_01",
  "sample_interval": 10,
  "default_lease_minutes": 10,
  "crash_lease_extension_minutes": 10,
  "tmp_dir": "./tmp",
  "state_dir": "./qc_state",
  "range_merge_gap_frames": 5,
  "request_timeout_seconds": 10.0,
  "playback_fps": 30.0,
  "mesh_renderer_python": "/home/ubuntu/WorkSpace/zhenghao/opt_toolkits/.venv/bin/python",
  "mano_toolkit_root": "/home/ubuntu/WorkSpace/zhenghao/opt_toolkits",
  "mano_model_dir": "/home/ubuntu/WorkSpace/zhenghao/opt_toolkits/ckpt/mano",
  "mesh_render_factor": 1.0,
  "nas_mounts": {"nas://ego": "/mnt/nas"}
}
```

`sample_interval` 仅为旧进度和 QC 报告协议保留，播放界面不再按采样点检查。mesh renderer 使用独立 Python，需能导入优化工具包、PyTorch、OpenCV、`trimesh` 和 `pyrender`。

## 行为

- Task / Episode 列表来自后端 `qc` stage 的可租 job。
- 双击 Episode 后才会正式租借。
- 本机保留的有效进度会合并显示，并显示剩余租期倒计时。
- RGB H.265 会解码到配置里的 `tmp_dir/<episode_id>/<camera>/<frame>.png`。
- `<episode>/optimized_pose/<frame>.npy` 会按 `mano/pose_mesh.py` 的方式生成 MANO 表面，并预渲染到 `tmp_dir/<episode_id>/mesh/<camera>/<frame>.jpg`。
- 六个视角使用同一个播放帧游标，不裁剪画面；播放时可暂停，暂停后可前后移动 1 帧或 10 帧。
- 暂停后可把当前帧标记为不通过并选择坏帧区间；确认后的区间在播放进度条中显示为红色。
- 正常结果必须播放到末帧后才能提交；“Episode 异常”仍可直接提交。
- QC 提交会先写 `<episode>/qc/qc_report.json`，再调用后端 complete。
- “Episode 异常”提交为 `result_type=bad_episode`，不会创建人工返修 segment。
