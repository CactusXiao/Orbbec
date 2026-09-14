# Orbbec Workflow Integration Contract

本文档面向真实自动标注、3D/MANO 优化、质检 worker 的接入方。当前后端只负责任务状态、租约、产物登记和状态推进，不运行模型，也不强制校验模型产物文件内容。真实服务应作为外部 HTTP worker 接入。

## 1. Backend Boundary

- 默认服务地址：`http://127.0.0.1:8765`
- 协议：HTTP JSON，`Content-Type: application/json`
- 鉴权：当前无鉴权层，部署到共享网络前需要由反向代理、内网 ACL 或 VPN 兜底。
- 后端状态：`progress_state.json` 记录采集任务进度，`workflow.sqlite3` 记录 workflow episode/job/artifact/segment。
- URI 类型：后端记录 `local://...`、`nas://...` 等抽象 URI；worker 需要能把 `payload.data_uri` 或 `payload.resolved_data_path` 映射成本机可读写路径。
- 推荐使用返回里的顶层 `payload`，不要只读 `job.payload`。顶层 `payload` 已由后端补齐 episode、artifact、camera、frame、media 和本地路径解析信息。

## 2. End-to-End State Machine

主路径：

```text
captured
  -> upload job succeeded
uploaded
  -> POST /api/v1/workflow/episodes/push-auto-label
auto_label job queued
  -> auto_label leased/running/succeeded
auto_labeled
  -> backend queues episode-level mano_opt
mano_opt job queued
  -> mano_opt leased/running/succeeded
mano_optimized
  -> backend queues qc
qc job queued
  -> qc passed
finalized
```

QC 失败返修路径：

```text
qc job succeeded with passed=false
qc_failed -> manual_correction_pending
  -> backend creates manual segments from qc result.segments
manual_segment leased/completed by Label UI
  -> backend queues segment-level mano_opt
segment mano_opt succeeded
  -> all failed segments mano_succeeded
finalized
```

阶段租约默认是暂停的。真实 worker 上线前必须开启：

```bash
curl -s -X POST "$BASE/api/v1/workflow/stages/auto_label/enable" \
  -H 'Content-Type: application/json' -d '{"updated_by":"real_worker"}'
curl -s -X POST "$BASE/api/v1/workflow/stages/mano_opt/enable" \
  -H 'Content-Type: application/json' -d '{"updated_by":"real_worker"}'
curl -s -X POST "$BASE/api/v1/workflow/stages/qc/enable" \
  -H 'Content-Type: application/json' -d '{"updated_by":"real_worker"}'
curl -s -X POST "$BASE/api/v1/workflow/stages/manual_segment/enable" \
  -H 'Content-Type: application/json' -d '{"updated_by":"label_ui"}'
```

`upload` 不受阶段开关控制；`manual_segment` 是人工返修分段阶段，不是普通 `jobs.type`。

## 3. Shared Job API

### 3.1 Lease

```http
POST /api/v1/jobs/lease
```

请求：

```json
{
  "type": "auto_label",
  "worker_id": "auto_label_worker_01",
  "lease_seconds": 300,
  "task_name": "optional_filter",
  "subject_id": "optional_filter"
}
```

`type` 可为 `upload`、`auto_label`、`mano_opt`、`qc`、`review`、`manual_label`。真实自动链路只需要 `auto_label`、`mano_opt`、`qc`。如果没有可租 job，返回 `404 {"error":"no queued ... job is available"}`；如果阶段未开启，返回 `409 {"error":"leasing disabled for job type: ..."}`。

响应通用结构：

```json
{
  "job": {
    "job_id": "auto_label_<episode>_episode",
    "type": "auto_label",
    "status": "leased",
    "episode_id": "<episode_id>",
    "payload": {},
    "result": {},
    "lease_owner": "auto_label_worker_01",
    "lease_until": "2026-08-05T10:00:00Z",
    "attempt": 0,
    "created_at": "2026-08-05T09:55:00Z",
    "updated_at": "2026-08-05T09:55:00Z"
  },
  "episode": {
    "episode_id": "<episode_id>",
    "subject_id": "S001",
    "task_name": "pick_object",
    "episode_index": 1,
    "status": "auto_labeling",
    "data_uri": "nas://ego/S001/pick_object/episode_001",
    "local_capture_path": "/capture/local/path",
    "frame_count": 120,
    "cameras": ["00", "01"],
    "metadata": {},
    "created_at": "...",
    "updated_at": "..."
  },
  "artifacts": [],
  "payload": {
    "job_id": "...",
    "episode_id": "...",
    "subject_id": "S001",
    "task_name": "pick_object",
    "data_uri": "nas://...",
    "resolved_data_path": "/mounted/nas/S001/pick_object/episode_001",
    "cameras": ["00", "01"],
    "frames": [0, 1, 2],
    "scope": "episode",
    "rgb_path_template": "{camera}/RGB/{frame:05d}.png",
    "episode_media": {
      "schema_version": 1,
      "kind": "orbbec_episode_media",
      "requires_rgb_video_decode": true,
      "rgb_frame_files_required": false,
      "cameras": {}
    }
  }
}
```

### 3.2 Heartbeat

```http
POST /api/v1/jobs/{job_id}/heartbeat
```

请求：

```json
{
  "worker_id": "auto_label_worker_01",
  "lease_seconds": 300,
  "status": "running"
}
```

`status` 只能是 `leased` 或 `running`。建议长任务每 `lease_seconds / 3` 续约一次。

### 3.3 Complete

```http
POST /api/v1/jobs/{job_id}/complete
```

请求通用结构：

```json
{
  "result": {
    "ok": true,
    "worker_id": "worker_01",
    "model": "model_name_or_version"
  },
  "artifacts": [
    {
      "kind": "pred_2d",
      "uri": "nas://.../pred_2d",
      "metadata": {
        "worker_id": "worker_01"
      }
    }
  ]
}
```

`artifacts` 也可用单个 `artifact` 对象代替。后端强制要求每个 artifact 有 `kind` 和 `uri`；`metadata` 必须是对象。重复 complete 已成功 job 会返回 `changed=false`，不会再次推进状态。

### 3.4 Fail And Release

```http
POST /api/v1/jobs/{job_id}/fail
POST /api/v1/jobs/{job_id}/release
```

fail 请求：

```json
{
  "error": "model input missing",
  "result": {
    "worker_id": "worker_01",
    "cleanup_manifest": {
      "attempt_outputs": ["nas://.../tmp/attempt_001"]
    }
  }
}
```

release 请求：

```json
{
  "reason": "worker shutting down"
}
```

`release` 会把非终态 job 放回 `queued`。`fail` 会把普通 job 标记为 `failed`；segment-level `mano_opt` 失败时，后端会保留人工 2D，并自动创建新的 segment MANO retry job。

## 4. Stage And Episode Control API

### 4.1 Stage Status

```http
GET /api/v1/workflow/stages/{job_type}
POST /api/v1/workflow/stages/{job_type}/enable
POST /api/v1/workflow/stages/{job_type}/disable
```

`job_type` 可为 `auto_label`、`mano_opt`、`qc`、`manual_segment`。GET 返回 `control`、`stats`、`active`、`queued`、`completed`。

### 4.2 Push Uploaded Episodes To Auto Label

```http
POST /api/v1/workflow/episodes/push-auto-label
```

按单个 episode：

```json
{
  "episode_id": "<episode_id>",
  "pushed_by": "operator_or_service"
}
```

按任务：

```json
{
  "task_name": "pick_object",
  "subject_id": "optional_subject_filter",
  "pushed_by": "operator_or_service"
}
```

全部 eligible episode：

```json
{
  "scope": "all",
  "pushed_by": "operator_or_service"
}
```

只有 `status=uploaded` 且有 `nas_uri`/`data_uri` 的 episode 会被推送。重复推送同一个 episode 是幂等的：已有 `auto_label` job 时不会再创建。

### 4.3 Episode Workflow Status

```http
GET /api/v1/episodes/{episode_id}/upload
GET /api/v1/collection/episodes/{episode_id}/upload
```

返回 episode 当前 workflow、upload、全部 artifacts、segments 和 compact jobs。Dashboard 页面同样可看：

```text
/
/workflow/stages/auto_label
/workflow/stages/mano_opt
/workflow/stages/qc
/episodes/<episode_id>
```

## 5. Auto Label Worker Contract

### 5.1 Input Payload

`auto_label` lease 的关键字段：

```json
{
  "job_id": "auto_label_<episode>_episode",
  "episode_id": "<episode_id>",
  "subject_id": "S001",
  "task_name": "pick_object",
  "data_uri": "nas://.../episode_001",
  "resolved_data_path": "/mounted/nas/.../episode_001",
  "cameras": ["00", "01"],
  "frames": [0, 1, 2],
  "scope": "episode",
  "label_scope": "episode",
  "rgb_path_template": "{camera}/RGB/{frame:05d}.png",
  "prediction_dir": "pred_2d",
  "correction_dir": "corrected_2d",
  "episode_media": {}
}
```

RGB 输入优先级：

1. 如果 `episode_media.cameras[cam].rgb.path` 可用，读该视频或帧源；`encoding` 当前常见为 `h265`。
2. 否则按 `resolved_data_path / rgb_path_template.format(camera=cam, frame=frame)` 找帧图。
3. 如果只存在视频，worker 需要自行解码并保证输出帧号与 `frames` 对齐。

### 5.2 Required Output Files

推荐写入：

```text
<episode>/pred_2d/<camera>/<frame:05d>.npy
```

每个 `.npy`：

- dtype：`float32`
- shape：`(2, 21, 2)`
- 坐标单位：对应 RGB 图像像素坐标
- 不可见点：`[-1.0, -1.0]`
- hand 维度：`0/1` 固定两个手槽；无法检测的手整手填 `-1`

可选 visibility sidecar 当前工具可读，但不是后端强制项；没有 sidecar 时会根据 `[-1,-1]` 推断可见性。

### 5.3 Complete Request

```json
{
  "result": {
    "ok": true,
    "model": "real_auto_label_v1",
    "worker_id": "auto_label_worker_01",
    "frames": [0, 1, 2],
    "frames_predicted": [0, 1, 2],
    "prediction_dir": "pred_2d"
  },
  "artifacts": [
    {
      "kind": "pred_2d",
      "uri": "nas://.../episode_001/pred_2d",
      "metadata": {
        "worker_id": "auto_label_worker_01",
        "model": "real_auto_label_v1"
      }
    }
  ]
}
```

后端收到 `auto_label` 成功后：

- 如果 artifact 未显式给 `pred_2d`/`auto_2d`，会默认登记 `data_uri/pred_2d`。
- 当该 episode 的所有 `auto_label` job 都成功后，episode 变为 `auto_labeled`。
- 后端自动创建 episode-level `mano_opt` job。

## 6. 3D/MANO Optimization Worker Contract

`mano_opt` 有两种 scope：`episode` 和 `segment`。

### 6.1 Episode-Level Input

```json
{
  "job_id": "mano_opt_<episode>_episode",
  "episode_id": "<episode_id>",
  "data_uri": "nas://.../episode_001",
  "resolved_data_path": "/mounted/nas/.../episode_001",
  "cameras": ["00", "01"],
  "frames": [0, 1, 2],
  "scope": "episode",
  "mano_scope": "episode",
  "input_2d_uri": "nas://.../episode_001/pred_2d",
  "pred_uri": "nas://.../episode_001/pred_2d",
  "output_uri": "nas://.../episode_001/mano/episode",
  "mano_output_dir": "mano/episode",
  "prediction_dir": "pred_2d",
  "rgb_path_template": "{camera}/RGB/{frame:05d}.png"
}
```

读取 2D 输入：

```text
<episode>/<prediction_dir>/<camera>/<frame:05d>.npy
```

推荐输出：

```text
<episode>/mano/episode/joints_3d.npy
<episode>/mano/episode/mano_episode.json
```

`joints_3d.npy`：

- dtype：`float32`
- shape：`(N, 2, 21, 3)`，`N == len(frames)`
- 坐标系：接入方必须在 `mano_episode.json` 里声明；当前 UI 会把该 3D 按 camera 参数投影显示。

`mano_episode.json` 推荐：

```json
{
  "schema_version": 1,
  "kind": "orbbec_mano_3d_episode",
  "frames": [0, 1, 2],
  "cameras": ["00", "01"],
  "joints_3d_file": "joints_3d.npy",
  "coordinate_system": "world_or_camera_declared_by_worker",
  "model": "real_mano_v1"
}
```

Complete：

```json
{
  "result": {
    "ok": true,
    "scope": "episode",
    "worker_id": "mano_worker_01",
    "frames": [0, 1, 2],
    "frames_optimized": [0, 1, 2],
    "output_uri": "nas://.../episode_001/mano/episode"
  },
  "artifacts": [
    {
      "kind": "mano_episode",
      "uri": "nas://.../episode_001/mano/episode",
      "metadata": {
        "worker_id": "mano_worker_01",
        "scope": "episode"
      }
    }
  ]
}
```

后端收到 episode-level `mano_opt` 成功后：

- 如果未显式给 `mano_episode` artifact，会默认登记 `result.output_uri` 或 `data_uri/mano/episode`。
- episode 变为 `mano_optimized`。
- 后端写/刷新 `<episode>/workflow/final_3d_sources.json`。
- 后端自动创建 `qc` job。

### 6.2 Segment-Level Input

QC 失败后，人工 UI 完成一个 segment 会自动创建 segment-level `mano_opt`：

```json
{
  "job_id": "mano_opt_<segment_id>",
  "episode_id": "<episode_id>",
  "segment_id": "<segment_id>",
  "data_uri": "nas://.../episode_001",
  "resolved_data_path": "/mounted/nas/.../episode_001",
  "cameras": ["00", "01"],
  "frames": [10, 11, 12],
  "start_frame": 10,
  "end_frame": 12,
  "scope": "segment",
  "mano_scope": "segment",
  "input_2d_uri": "nas://.../manual_2d/segments/<segment_id>",
  "manual_2d_uri": "nas://.../manual_2d/segments/<segment_id>",
  "pred_uri": "nas://.../pred_2d",
  "mano_episode_uri": "nas://.../mano/episode",
  "output_uri": "nas://.../mano/segments/<segment_id>",
  "manual_2d_dir": "manual_2d/segments/<segment_id>",
  "mano_output_dir": "mano/segments/<segment_id>",
  "mano_episode_dir": "mano/episode"
}
```

读取人工 2D：

```text
<episode>/manual_2d/segments/<segment_id>/<camera>/<frame:05d>.npy
```

人工 2D 的 shape 为 `(2, 21, 2)`，hand order 固定为 `left, right`，joint order 固定为
`mano/mano(1).py` 中的 `SMPLX_MANO_JOINT_NAMES`。MANO 投影、Label 画布和 artifact 使用同一索引，不做二次重排。

推荐输出：

```text
<episode>/mano/segments/<segment_id>/joints_3d.npy
<episode>/mano/segments/<segment_id>/mano_patch.json
```

`mano_patch.json` 推荐：

```json
{
  "schema_version": 1,
  "kind": "orbbec_mano_3d_segment_patch",
  "segment_id": "<segment_id>",
  "frames": [10, 11, 12],
  "cameras": ["00", "01"],
  "joints_3d_file": "joints_3d.npy",
  "model": "real_mano_v1"
}
```

Complete：

```json
{
  "result": {
    "ok": true,
    "scope": "segment",
    "segment_id": "<segment_id>",
    "worker_id": "mano_worker_01",
    "output_uri": "nas://.../episode_001/mano/segments/<segment_id>"
  },
  "artifacts": [
    {
      "kind": "mano_segment_patch",
      "uri": "nas://.../episode_001/mano/segments/<segment_id>",
      "metadata": {
        "worker_id": "mano_worker_01",
        "scope": "segment",
        "segment_id": "<segment_id>"
      }
    }
  ]
}
```

后端收到 segment-level `mano_opt` 成功后：

- segment 变为 `mano_succeeded`。
- 后端刷新 `workflow/final_3d_sources.json`。
- 如果该 episode 的所有 segments 都是 `mano_succeeded`，episode 变为 `finalized`。

## 7. QC Worker Contract

### 7.1 Input Payload

```json
{
  "job_id": "qc_<episode>",
  "episode_id": "<episode_id>",
  "data_uri": "nas://.../episode_001",
  "resolved_data_path": "/mounted/nas/.../episode_001",
  "cameras": ["00", "01"],
  "frames": [0, 1, 2],
  "pred_uri": "nas://.../pred_2d",
  "mano_episode_uri": "nas://.../mano/episode",
  "mano_episode_dir": "mano/episode",
  "qc_report_uri": "nas://.../qc/qc_report.json",
  "prediction_dir": "pred_2d",
  "rgb_path_template": "{camera}/RGB/{frame:05d}.png"
}
```

QC 应检查：

- 2D 预测是否完整、shape 是否正确。
- 3D/MANO episode 产物是否存在，推荐检查 `mano/episode/joints_3d.npy` 和 `mano_episode.json`。
- 关键帧投影、时序跳变、可见性、左右手一致性、缺失率等真实质量指标。

### 7.2 Passed Complete

```json
{
  "result": {
    "passed": true,
    "qc_passed": true,
    "score": 0.98,
    "reason": "qc_passed",
    "worker_id": "qc_worker_01",
    "mano_3d_uri": "nas://.../mano/episode",
    "mano_3d_checked": true
  },
  "artifacts": [
    {
      "kind": "qc_report",
      "uri": "nas://.../episode_001/qc/qc_report.json",
      "metadata": {
        "passed": true,
        "worker_id": "qc_worker_01"
      }
    }
  ]
}
```

后端收到 `passed=true` 后，episode 变为 `finalized`，并刷新 `workflow/final_3d_sources.json`。

### 7.3 Failed Complete

```json
{
  "result": {
    "passed": false,
    "qc_passed": false,
    "score": 0.41,
    "reason": "projection_error",
    "worker_id": "qc_worker_01",
    "segments": [
      {
        "start_frame": 10,
        "end_frame": 24,
        "reason": "temporal_jump",
        "score": 0.23
      },
      {
        "start_frame": 60,
        "end_frame": 75,
        "reason": "missing_hand",
        "score": 0.18
      }
    ]
  },
  "artifacts": [
    {
      "kind": "qc_report",
      "uri": "nas://.../episode_001/qc/qc_report.json",
      "metadata": {
        "passed": false,
        "worker_id": "qc_worker_01"
      }
    }
  ]
}
```

`segments` 支持字段别名：

- segment 列表字段可叫 `segments`、`failed_segments`、`qc_failed_segments` 或 `failure_segments`。
- 每段可用 `start_frame`/`end_frame`，也支持 `start`/`end`、`first_frame`/`last_frame`。
- 如未给任何合法 segment，后端会把整个 episode frame 范围作为一个失败段。

后端收到 `passed=false` 后：

- episode 先变 `qc_failed`，随后变 `manual_correction_pending`。
- 为每个失败段创建一个 `pending_manual` segment。
- 人工 Label UI 从 `/api/v1/label/segments/lease` 租这些 segment。

### 7.4 QC Report File

推荐写入：

```text
<episode>/qc/qc_report.json
```

推荐内容：

```json
{
  "schema_version": 1,
  "kind": "orbbec_qc_report",
  "episode_id": "<episode_id>",
  "passed": false,
  "score": 0.41,
  "segments": [],
  "metrics": {},
  "model_versions": {},
  "created_at": "2026-08-05T00:00:00Z"
}
```

## 8. Manual Segment API For QC Failure

真实自动链路一般不需要实现人工 UI，但需要理解它如何回流 3D 优化。

### 8.1 Queue

```http
GET /api/v1/label/tasks
GET /api/v1/label/tasks/{task_name}/episodes
```

### 8.2 Lease Segment

```http
POST /api/v1/label/segments/lease
```

请求：

```json
{
  "operator_id": "labeler_01",
  "lease_seconds": 600,
  "task_name": "optional",
  "episode_id": "optional"
}
```

响应也是 `{segment, episode, artifacts, payload}`。关键 `payload` 字段：

```json
{
  "segment_id": "<segment_id>",
  "episode_id": "<episode_id>",
  "data_uri": "nas://...",
  "resolved_data_path": "/mounted/nas/...",
  "start_frame": 10,
  "end_frame": 24,
  "frames": [10, 11, 12],
  "cameras": ["00", "01"],
  "pred_2d_uri": "nas://.../pred_2d",
  "mano_episode_uri": "nas://.../mano/episode",
  "manual_2d_output_uri": "nas://.../manual_2d/segments/<segment_id>",
  "correction_dir": "manual_2d/segments/<segment_id>"
}
```

### 8.3 Complete Segment

```http
POST /api/v1/label/segments/{segment_id}/complete
```

请求：

```json
{
  "result": {
    "operator_id": "labeler_01",
    "frames_completed": [10, 11, 12]
  },
  "artifacts": [
    {
      "kind": "manual_2d",
      "uri": "nas://.../manual_2d/segments/<segment_id>",
      "metadata": {
        "segment_id": "<segment_id>",
        "operator_id": "labeler_01"
      }
    }
  ]
}
```

后端会把 segment 标记为 `manual_labeled`，然后自动创建 segment-level `mano_opt` job。

## 9. Artifact Kinds And Default Locations

| Stage | Artifact kind | Default URI/path |
| --- | --- | --- |
| Upload | `nas_episode` | `nas://.../<episode>` |
| Auto label | `pred_2d` or `auto_2d` | `<episode>/pred_2d` |
| Episode 3D | `mano_episode` | `<episode>/mano/episode` |
| QC | `qc_report` | `<episode>/qc/qc_report.json` |
| Manual segment | `manual_2d` | `<episode>/manual_2d/segments/<segment_id>` |
| Segment 3D | `mano_segment_patch` | `<episode>/mano/segments/<segment_id>` |
| Final source map | backend-written file | `<episode>/workflow/final_3d_sources.json` |

`workflow/final_3d_sources.json` 由后端写入，包含 base episode 3D 和人工修补 segment 覆盖规则。消费者应按：

```text
默认使用 base_3d；
当 frame 落入 ready override 的 [start_frame, end_frame] 时，使用对应 mano_segment_patch。
```

## 10. Error Contract

所有 API 成功当前返回 HTTP 200 和 JSON object。错误格式：

```json
{
  "error": "human readable message"
}
```

常见状态码：

- `400`：请求体字段缺失、字段类型错误、job/stage/status 不支持。
- `404`：job、episode、segment 不存在，或当前没有可租 job。
- `409`：阶段租约暂停、租约 owner 冲突、job 已终态、采集 reservation 冲突。
- `503`：后端进程尚未在 setup 页面启动 task instance 时访问 collection API。
- `500`：后端未预期异常。

worker 建议策略：

- `404 no queued ...`：正常空队列，等待后重试。
- `409 leasing disabled`：阶段未开放，报警但不要退出整个服务。
- `409 leased by another owner`：检查是否重复 worker_id 或租约过期逻辑。
- 产物写失败后调用 `/fail`，并在 `cleanup_manifest` 中列出临时输出，便于后端清理能解析到的本地路径。
- worker 自己崩溃时不调用 complete；租约过期后 job 可被其他 worker 重新 lease。

## 11. Minimal Worker Loop

```python
while True:
    leased = post("/api/v1/jobs/lease", {
        "type": "auto_label",
        "worker_id": WORKER_ID,
        "lease_seconds": 300,
    })
    payload = leased["payload"]
    job_id = leased["job"]["job_id"]

    post(f"/api/v1/jobs/{job_id}/heartbeat", {
        "worker_id": WORKER_ID,
        "lease_seconds": 300,
        "status": "running",
    })

    # Read payload["resolved_data_path"] or resolve payload["data_uri"].
    # Write outputs under the episode directory.

    post(f"/api/v1/jobs/{job_id}/complete", {
        "result": {"ok": True, "worker_id": WORKER_ID},
        "artifacts": [{"kind": "pred_2d", "uri": payload["data_uri"] + "/pred_2d"}],
    })
```

生产实现需要在长任务内部定时 heartbeat，并确保写文件采用临时目录/临时文件后原子 rename，避免后端或下游读到半成品。

## 12. Pre-Integration Checklist

1. 后端已启动并在 setup 页面选择了 task file instance。
2. `ORBBEC_NAS_MOUNTS_JSON` 能解析 `nas://...`。
3. `auto_label`、`mano_opt`、`qc`、需要时 `manual_segment` 阶段已 enable。
4. 上传完成的 episode 已通过 `/api/v1/workflow/episodes/push-auto-label` 推入自动标注队列。
5. 自动标注输出 `pred_2d/<camera>/<frame:05d>.npy`，shape 为 `(2,21,2)`。
6. 3D 输出 `mano/episode/joints_3d.npy` 和 `mano_episode.json`，shape 为 `(N,2,21,3)`。
7. QC 输出 `qc/qc_report.json`，并在 result 中明确 `passed` 或 `qc_passed`。
8. QC 失败时 result 提供合法 frame segments。
9. segment 3D 输出 `mano/segments/<segment_id>`，complete artifact kind 为 `mano_segment_patch`。
10. Dashboard 中 `/workflow/stages/*` 能看到 queued/active/completed 状态正确推进。
