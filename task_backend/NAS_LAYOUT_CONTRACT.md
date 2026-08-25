# NAS File Layout Contract

本文档定义采集、自动标注、3D/MANO、QC、人工返修共用的 NAS 文件结构。所有 worker 必须只在本文规定目录写入。

## 1. URI 与本地路径

后端用 UUID 作为 episode 主键，但 UUID 不进入 NAS 目录名。预约时后端同时分配：

```text
episode_id:   <uuid>
storage_name: episode<episode_index>
```

后端 `episodes` 表用 `(subject_id, task_name, storage_name)` 唯一定位 NAS 目录，并维护
`episode_id -> subject_id/task_name/storage_name -> episode_uri` 映射。`episode_index` 一旦分配就不复用
（包括预约释放后），避免旧 NAS 数据被新 episode 覆盖。

后端记录 episode 根 URI：

```text
nas://<dataset>/<subject_id>/<task_name>/<storage_name>
```

worker 必须将其映射到本机路径：

```text
<NAS_MOUNT>/<subject_id>/<task_name>/<storage_name>
```

示例：

```text
episode_id:   550e8400-e29b-41d4-a716-446655440000
storage_name: episode12
URI:          nas://ego/xiaojiazhou/task-clean-the-bowl/episode12
Path:         /mnt/nas/xiaojiazhou/task-clean-the-bowl/episode12
```

后端返回 `payload.resolved_data_path` 时优先使用该路径。

每个 episode 根目录必须包含 `.orbbec_upload_manifest.json`，至少记录：

```json
{
  "episode_id": "550e8400-e29b-41d4-a716-446655440000",
  "episode_uuid": "550e8400-e29b-41d4-a716-446655440000",
  "subject_id": "xiaojiazhou",
  "task_name": "task-clean-the-bowl",
  "episode_index": 12,
  "storage_name": "episode12"
}
```

## 2. Episode 根目录

标准结构：

```text
<episode>/
  camera_params.json
  extrinsics.json
  timestamps.csv
  <camera>/
    camera_params.json
    RGB/
    Depth/
  pred_2d/
  manual_2d/
  mano/
  qc/
  workflow/
```

`camera_params.json`、`extrinsics.json` 缺失时部分工具可回退到工作目录同名文件，但生产数据必须放在 episode 内。

## 3. 原始采集数据

每个标注相机目录：

```text
<episode>/<camera>/
  RGB/
    rgb.h265
    rgb.h265.timestamps.csv
    或 <frame:05d>.<ext>
  Depth/
    depth.mkv
    depth.mkv.timestamps.csv
```

约定：

- `<camera>` 使用后端 payload 中的 camera ID，例如 `00`、`01`。
- RGB 帧默认路径模板：`{camera}/RGB/{frame:05d}.png`。
- 如果保存为视频，worker 读取 `episode_media` 或 `camera_params.json` 中的 storage/timestamp 描述。
- 原始采集目录为只读区。自动标注、3D、QC、人工返修不得改写原始文件。

## 4. 自动标注输出

目录：

```text
<episode>/pred_2d/<camera>/<frame:05d>.npy
```

文件要求：

- dtype：`float32`
- shape：`(2, 21, 2)`
- 坐标：RGB 像素坐标
- 不可见点：`[-1.0, -1.0]`
- 无手帧也必须写文件，内容全 `-1`

对应后端 artifact：

```json
{
  "kind": "pred_2d",
  "uri": "nas://.../<episode>/pred_2d"
}
```

## 5. Episode 级 3D/MANO 输出

目录：

```text
<episode>/mano/episode/
  joints_3d.npy
  mano_episode.json
```

`joints_3d.npy`：

- dtype：`float32`
- shape：`(N, 2, 21, 3)`
- 帧顺序必须与 job payload `frames` 一致

`mano_episode.json` 最小字段：

```json
{
  "schema_version": 1,
  "kind": "orbbec_mano_3d_episode",
  "frames": [0, 1, 2],
  "cameras": ["00", "01"],
  "joints_3d_file": "joints_3d.npy",
  "coordinate_system": "declared_by_worker",
  "model": "mano_v1"
}
```

对应后端 artifact：

```json
{
  "kind": "mano_episode",
  "uri": "nas://.../<episode>/mano/episode"
}
```

## 6. QC 输出

目录：

```text
<episode>/qc/
  qc_report.json
```

`qc_report.json` 最小字段：

```json
{
  "schema_version": 1,
  "kind": "orbbec_qc_report",
  "episode_id": "<episode_id>",
  "passed": true,
  "score": 0.98,
  "segments": [],
  "metrics": {},
  "created_at": "2026-08-05T00:00:00Z"
}
```

对应后端 artifact：

```json
{
  "kind": "qc_report",
  "uri": "nas://.../<episode>/qc/qc_report.json"
}
```

QC 失败时，`segments` 使用：

```json
[
  {
    "start_frame": 10,
    "end_frame": 24,
    "reason": "projection_error",
    "score": 0.23
  }
]
```

## 7. 人工返修 2D 输出

目录：

```text
<episode>/manual_2d/segments/<segment_id>/<camera>/<frame:05d>.npy
```

文件要求同 `pred_2d`：

- dtype：`float32`
- shape：`(2, 21, 2)`
- 不可见点：`[-1.0, -1.0]`

对应后端 artifact：

```json
{
  "kind": "manual_2d",
  "uri": "nas://.../<episode>/manual_2d/segments/<segment_id>"
}
```

## 8. Segment 级 3D Patch 输出

目录：

```text
<episode>/mano/segments/<segment_id>/
  joints_3d.npy
  mano_patch.json
```

`joints_3d.npy`：

- dtype：`float32`
- shape：`(N, 2, 21, 3)`
- 只包含该 segment 的 `frames`

`mano_patch.json` 最小字段：

```json
{
  "schema_version": 1,
  "kind": "orbbec_mano_3d_segment_patch",
  "segment_id": "<segment_id>",
  "frames": [10, 11, 12],
  "cameras": ["00", "01"],
  "joints_3d_file": "joints_3d.npy",
  "coordinate_system": "declared_by_worker",
  "model": "mano_v1"
}
```

对应后端 artifact：

```json
{
  "kind": "mano_segment_patch",
  "uri": "nas://.../<episode>/mano/segments/<segment_id>"
}
```

## 9. 后端 Workflow 文件

后端维护：

```text
<episode>/workflow/final_3d_sources.json
```

worker 不得写该文件。它用于声明最终 3D 结果来源：

- 默认使用 `mano/episode`
- 对 QC 失败并返修成功的帧段，使用 `mano/segments/<segment_id>`

## 10. 写入规则

- 所有 worker 输出必须写在对应 stage 目录下。
- 写入大文件必须先写临时文件，再原子 rename 到目标文件。
- complete API 必须在文件全部落盘后调用。
- URI 必须指向最终稳定目录或文件，不能指向临时路径。
- 同一 job 重试时允许覆盖本 job 的未完成输出，不得覆盖其他 job 或其他 segment 输出。
- 文件名中的帧号固定为 5 位十进制：`00000.npy`。
- 目录名、camera ID、segment ID 必须与后端 payload 保持一致。

## 11. 禁止事项

- 禁止修改或删除 `<camera>/RGB`、`<camera>/Depth` 原始采集数据。
- 禁止 worker 写 `workflow/final_3d_sources.json`。
- 禁止把自动标注结果写入 `manual_2d`。
- 禁止把人工返修结果写入 `pred_2d`。
- 禁止不同 segment 共用同一 `mano/segments/<segment_id>` 目录。
- 禁止 complete 后继续修改已登记 artifact。

## 12. 最小验收

一个完整通过 QC 的 episode 至少包含：

```text
<episode>/
  <camera>/RGB/...
  pred_2d/<camera>/<frame:05d>.npy
  mano/episode/joints_3d.npy
  mano/episode/mano_episode.json
  qc/qc_report.json
  workflow/final_3d_sources.json
```

一个 QC 失败后返修完成的 episode 还必须包含：

```text
<episode>/
  manual_2d/segments/<segment_id>/<camera>/<frame:05d>.npy
  mano/segments/<segment_id>/joints_3d.npy
  mano/segments/<segment_id>/mano_patch.json
```
