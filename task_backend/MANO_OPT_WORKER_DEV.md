# 3D/MANO Optimization Worker Developer Guide

本文档只覆盖 3D/MANO 优化 worker。后端 job 类型为 `mano_opt`。worker 负责读取 2D 输入，生成 episode 级 3D 结果或返修 segment 级 3D patch，并回填 artifact。

## 1. 基本约定

- Backend URL：`http://127.0.0.1:8765`，生产环境以部署配置为准。
- 协议：HTTP JSON。
- worker 只通过 API 通信，不直接改后端数据库。
- 优先使用 lease 响应顶层 `payload`。
- `mano_opt` 有两种 `scope`：`episode` 和 `segment`。

## 2. 前置操作

开启 3D 优化租约：

```bash
curl -s -X POST "$BASE/api/v1/workflow/stages/mano_opt/enable" \
  -H 'Content-Type: application/json' \
  -d '{"updated_by":"mano_opt_worker"}'
```

job 来源：

- episode 级 job：`auto_label` 成功后由后端自动创建。
- segment 级 job：QC 失败后，人工完成某个返修 segment 后由后端自动创建。

worker 不需要主动创建 `mano_opt` job。

## 3. 租取 Job

```http
POST /api/v1/jobs/lease
```

请求：

```json
{
  "type": "mano_opt",
  "worker_id": "mano_opt_worker_01",
  "lease_seconds": 300
}
```

可选过滤：

```json
{
  "task_name": "pick_object",
  "subject_id": "S001"
}
```

空队列返回 `404`。阶段未开启返回 `409`。

## 4. Lease 响应

响应结构：

```json
{
  "job": {},
  "episode": {},
  "artifacts": [],
  "payload": {}
}
```

通用关键字段：

| 字段 | 说明 |
| --- | --- |
| `job_id` | 后续 heartbeat/complete/fail 使用 |
| `episode_id` | episode ID |
| `data_uri` | episode 根 URI |
| `resolved_data_path` | 后端可解析时给出的本机路径 |
| `cameras` | 相机列表 |
| `frames` | 本 job 需要处理的帧 |
| `scope` / `mano_scope` | `episode` 或 `segment` |
| `rgb_path_template` | RGB 帧路径模板 |
| `prediction_dir` | 自动 2D 目录，默认 `pred_2d` |

episode 路径优先级：

1. 使用 `payload.resolved_data_path`。
2. worker 自行把 `payload.data_uri` 映射到本机挂载路径。

## 5. Episode 级优化

### 5.1 输入字段

`scope=episode` 时关键字段：

| 字段 | 说明 |
| --- | --- |
| `input_2d_uri` / `pred_uri` | 自动 2D 输入 URI |
| `pred_artifacts` | 已登记的 2D artifact 列表 |
| `output_uri` | 推荐输出 URI |
| `mano_output_dir` | 推荐输出目录，默认 `mano/episode` |

读取 2D 输入：

```text
<episode>/<prediction_dir>/<camera>/<frame:05d>.npy
```

每个 2D `.npy`：

- dtype：`float32`
- shape：`(2, 21, 2)`
- 不可见点：`[-1.0, -1.0]`

### 5.2 输出文件

推荐写入：

```text
<episode>/mano/episode/joints_3d.npy
<episode>/mano/episode/mano_episode.json
```

`joints_3d.npy`：

| 项 | 要求 |
| --- | --- |
| dtype | `float32` |
| shape | `(N, 2, 21, 3)` |
| N | `len(payload.frames)` |
| 帧顺序 | 与 `payload.frames` 一致 |

`mano_episode.json`：

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

### 5.3 完成请求

```http
POST /api/v1/jobs/{job_id}/complete
```

```json
{
  "result": {
    "ok": true,
    "scope": "episode",
    "worker_id": "mano_opt_worker_01",
    "model": "mano_v1",
    "frames_optimized": [0, 1, 2],
    "output_uri": "nas://.../episode_001/mano/episode"
  },
  "artifacts": [
    {
      "kind": "mano_episode",
      "uri": "nas://.../episode_001/mano/episode",
      "metadata": {
        "worker_id": "mano_opt_worker_01",
        "model": "mano_v1",
        "scope": "episode"
      }
    }
  ]
}
```

后端成功后会：

1. 登记 `mano_episode` artifact。
2. 将 episode 推进到 `mano_optimized`。
3. 写入或刷新 `workflow/final_3d_sources.json`。
4. 自动创建 `qc` job。

## 6. Segment 级优化

### 6.1 输入字段

`scope=segment` 时关键字段：

| 字段 | 说明 |
| --- | --- |
| `segment_id` | 返修分段 ID |
| `start_frame` / `end_frame` | 分段范围 |
| `frames` | 分段帧列表 |
| `input_2d_uri` / `manual_2d_uri` | 人工修正 2D 输入 |
| `manual_2d_dir` | 默认 `manual_2d/segments/<segment_id>` |
| `mano_episode_uri` | episode 级 3D 基线 |
| `mano_episode_dir` | 默认 `mano/episode` |
| `output_uri` | 推荐 patch 输出 URI |
| `mano_output_dir` | 默认 `mano/segments/<segment_id>` |

读取人工 2D：

```text
<episode>/manual_2d/segments/<segment_id>/<camera>/<frame:05d>.npy
```

### 6.2 输出文件

推荐写入：

```text
<episode>/mano/segments/<segment_id>/joints_3d.npy
<episode>/mano/segments/<segment_id>/mano_patch.json
```

`joints_3d.npy` 要求同 episode 级：`float32`，shape `(N, 2, 21, 3)`，帧顺序与 `payload.frames` 一致。

`mano_patch.json`：

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

### 6.3 完成请求

```json
{
  "result": {
    "ok": true,
    "scope": "segment",
    "segment_id": "<segment_id>",
    "worker_id": "mano_opt_worker_01",
    "model": "mano_v1",
    "frames_optimized": [10, 11, 12],
    "output_uri": "nas://.../episode_001/mano/segments/<segment_id>"
  },
  "artifacts": [
    {
      "kind": "mano_segment_patch",
      "uri": "nas://.../episode_001/mano/segments/<segment_id>",
      "metadata": {
        "worker_id": "mano_opt_worker_01",
        "model": "mano_v1",
        "scope": "segment",
        "segment_id": "<segment_id>"
      }
    }
  ]
}
```

后端成功后会：

1. 将 segment 标记为 `mano_succeeded`。
2. 刷新 `workflow/final_3d_sources.json`。
3. 如果该 episode 所有返修 segments 都成功，将 episode 推进到 `finalized`。

## 7. Heartbeat

```http
POST /api/v1/jobs/{job_id}/heartbeat
```

```json
{
  "worker_id": "mano_opt_worker_01",
  "lease_seconds": 300,
  "status": "running"
}
```

建议每 `lease_seconds / 3` 续约一次。

## 8. 失败处理

临时问题用 release：

```http
POST /api/v1/jobs/{job_id}/release
```

```json
{
  "reason": "worker_shutdown"
}
```

`release` 会把非终态 job 放回 `queued`。

不可恢复问题用 fail：

```http
POST /api/v1/jobs/{job_id}/fail
```

```json
{
  "error": "missing_2d_input",
  "result": {
    "worker_id": "mano_opt_worker_01"
  }
}
```

失败语义：

- episode 级 `mano_opt`：`failed` 是终态，不会自动重试。
- segment 级 `mano_opt`：后端会保留人工 2D，并自动创建新的 segment retry job。

## 9. Worker 主循环

```python
while True:
    leased = post("/api/v1/jobs/lease", {
        "type": "mano_opt",
        "worker_id": WORKER_ID,
        "lease_seconds": 300,
    })

    job_id = leased["job"]["job_id"]
    payload = leased["payload"]
    scope = payload.get("scope") or payload.get("mano_scope") or "episode"

    post(f"/api/v1/jobs/{job_id}/heartbeat", {
        "worker_id": WORKER_ID,
        "lease_seconds": 300,
        "status": "running",
    })

    episode_dir = payload.get("resolved_data_path") or resolve_uri(payload["data_uri"])

    if scope == "segment":
        uri = run_segment_mano_and_write_patch(episode_dir, payload)
        artifact_kind = "mano_segment_patch"
    else:
        uri = run_episode_mano_and_write_result(episode_dir, payload)
        artifact_kind = "mano_episode"

    post(f"/api/v1/jobs/{job_id}/complete", {
        "result": {
            "ok": True,
            "worker_id": WORKER_ID,
            "scope": scope,
            "segment_id": payload.get("segment_id", ""),
            "output_uri": uri,
        },
        "artifacts": [{
            "kind": artifact_kind,
            "uri": uri,
            "metadata": {
                "worker_id": WORKER_ID,
                "scope": scope,
                "segment_id": payload.get("segment_id", ""),
            },
        }],
    })
```

生产实现必须使用临时文件写入后原子 rename，避免下游读到半成品。

## 10. 验收清单

- `/workflow/stages/mano_opt` 能看到 queued/active/completed 状态推进。
- episode 级输出包含 `mano/episode/joints_3d.npy` 和 `mano_episode.json`。
- segment 级输出包含 `mano/segments/<segment_id>/joints_3d.npy` 和 `mano_patch.json`。
- `joints_3d.npy` 为 `float32`，shape 为 `(N,2,21,3)`。
- complete 后 artifact kind 正确：episode 用 `mano_episode`，segment 用 `mano_segment_patch`。
- episode 级 complete 后自动出现 `qc` job。
- segment 级全部 complete 后 episode 进入 `finalized`。
