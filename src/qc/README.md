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
  "mesh_render_factor": 0.5,
  "mesh_render_workers": 6,
  "mesh_prebuffer_frames": 30,
  "mesh_prefer_integrated_gpu": true,
  "nas_mounts": {"nas://ego": "/mnt/nas"}
}
```

`sample_interval` 仅为旧进度和 QC 报告协议保留，播放界面不再按采样点检查。mesh renderer 使用独立 Python，需能导入优化工具包、PyTorch、OpenCV、`trimesh` 和 `pyrender`。00/02/03/05 四路 RGB 和 Pico Ego 视频会边解码、边渲染；前 30 个完整五视图 mesh 帧准备好后即可播放，RGB 与 mesh 图片会保留到离开当前 Episode。针对 9950X 的播放并发实测配置为 6 个渲染进程、0.5 倍 mesh 图层，并优先选择 AMD 核显 EGL 设备；不存在可用 OpenGL 时会自动回退 CPU。

Pico 视图读取 `<episode>/ego/camera_params.json`、`<episode>/ego_pose.json` 和带有 `frame_index`/`ego_frame_index` 的同步时间戳表。投影遵循 `mano/ego_pose.py`：使用逐帧 `T_ego_from_reference` 变换 MANO 顶点，按原始 OpenCV fisheye 标定扭曲透明 mesh 图层，再叠加到不做几何变换的 Pico RGB 上。

## 行为

- Task / Episode 列表来自后端 `qc` stage 的可租 job。
- 双击 Episode 后才会正式租借。
- 本机保留的有效进度会合并显示，并显示剩余租期倒计时。
- 00/02/03/05 和 Pico Ego RGB H.265 会并行解码为高质量 JPEG 缓存，旧 PNG 缓存仍可继续使用。
- `<episode>/optimized_pose/<frame>.npy` 会生成 MANO 表面，并预渲染到 `tmp_dir/<episode_id>/mesh/<camera>/<frame>.jpg`。
- 五个视图使用同一个播放帧游标，不裁剪画面；进度条支持点击和拖动，暂停后也可前后移动 1 帧或 10 帧。
- 暂停后可把当前帧标记为不通过并选择坏帧区间，随后二选一确认为“手部 Pose 不准”或“EgoPose 外参不准”。
- 手部 Pose 区间沿用进度条内的红色块，并进入原 QC/人工返修流程；EgoPose 区间以紧贴进度条下方的红色实线显示，只写 `<episode>/ego/ego_pose_qc.json`，不改变后端 QC 结果和后续流程。
- 正常结果必须播放到末帧后才能提交；“Episode 异常”仍可直接提交。
- QC 提交会先写 `<episode>/qc/qc_report.json`，再调用后端 complete。
- “Episode 异常”提交为 `result_type=bad_episode`，不会创建人工返修 segment。
