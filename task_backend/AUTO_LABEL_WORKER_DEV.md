# Auto Label Worker Developer Guide

本文档只覆盖 `auto_label` worker。worker 的职责是租取后端 job，读取 episode 数据，产出 2D 手部关键点，回填产物。

## 1. 基本约定

- Backend URL：`http://127.0.0.1:8765`，生产环境以部署配置为准。
- 协议：HTTP JSON。
- worker 不修改后端数据库，只通过 API 通信。
- worker 应优先使用 lease 响应顶层 `payload`，不要只读 `job.payload`。
- 一个 job 只处理一个 episode，`scope=episode`。

## 2. 前置操作

开启自动标注租约：

```bash
curl -s -X POST "$BASE/api/v1/workflow/stages/auto_label/enable" \
  -H 'Content-Type: application/json' \
  -d '{"updated_by":"auto_label_worker"}'
```

把已上传 episode 推入自动标注队列：

```bash
curl -s -X POST "$BASE/api/v1/workflow/episodes/push-auto-label" \
  -H 'Content-Type: application/json' \
  -d '{"scope":"all","pushed_by":"auto_label_worker"}'
```

也可只推单个 episode：

```json
{
  "episode_id": "<episode_id>",
  "pushed_by": "auto_label_worker"
}
```

## 3. 租取 Job

```http
POST /api/v1/jobs/lease
```

请求：

```json
{
  "type": "auto_label",
  "worker_id": "auto_label_worker_01",
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

## 4. Lease 响应关键字段

响应结构：

```json
{
  "job": {},
  "episode": {},
  "artifacts": [],
  "payload": {}
}
```

`payload` 关键字段：

| 字段 | 说明 |
| --- | --- |
| `job_id` | 后续 heartbeat/complete/fail 使用 |
| `episode_id` | episode ID |
| `data_uri` | episode 根 URI，如 `nas://.../episode_001` |
| `resolved_data_path` | 后端可解析时给出的本机路径 |
| `cameras` | 需要处理的相机列表 |
| `frames` | 需要处理的帧号列表 |
| `rgb_path_template` | RGB 帧路径模板，默认 `{camera}/RGB/{frame:05d}.png` |
| `prediction_dir` | 输出目录名，默认 `pred_2d` |
| `episode_media` | 视频输入描述；常见 RGB 编码为 `h265` |

读取 episode 路径优先级：

1. 使用 `payload.resolved_data_path`。
2. worker 自行把 `payload.data_uri` 映射到本机挂载路径。

读取 RGB 优先级：

1. 如果 `episode_media.cameras[cam].rgb.path` 存在，读取该视频或帧源。
2. 否则按 `<episode>/<rgb_path_template>` 读取图片帧。

## 5. 输出格式

写入目录：

```text
<episode>/<prediction_dir>/<camera>/<frame:05d>.npy
```

默认：

```text
<episode>/pred_2d/00/00000.npy
<episode>/pred_2d/01/00000.npy
```

每个 `.npy` 必须满足：

| 项 | 要求 |
| --- | --- |
| dtype | `float32` |
| shape | `(2, 21, 2)` |
| 单位 | RGB 像素坐标 |
| 不可见点 | `[-1.0, -1.0]` |
| 手槽 | 第 0 维固定两个手槽 |

示例：

```python
import numpy as np

points = np.full((2, 21, 2), -1.0, dtype=np.float32)
points[0, 0] = [320.0, 240.0]
np.save("pred_2d/00/00000.npy", points)
```

必须覆盖 `payload.cameras` 和 `payload.frames` 中的全部组合。确实无手时也要写文件，内容全为 `-1`。

## 6. Heartbeat

长任务必须续租：

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

建议每 `lease_seconds / 3` 发送一次。

## 7. 完成 Job

```http
POST /api/v1/jobs/{job_id}/complete
```

请求：

```json
{
  "result": {
    "ok": true,
    "worker_id": "auto_label_worker_01",
    "model": "hand2d_v1",
    "frames_predicted": [0, 1, 2],
    "prediction_dir": "pred_2d"
  },
  "artifacts": [
    {
      "kind": "pred_2d",
      "uri": "nas://.../episode_001/pred_2d",
      "metadata": {
        "worker_id": "auto_label_worker_01",
        "model": "hand2d_v1"
      }
    }
  ]
}
```

`artifacts[0].kind` 必须是 `pred_2d` 或 `auto_2d`。推荐固定使用 `pred_2d`。

后端收到成功后会：

1. 登记 `pred_2d` artifact。
2. 将 episode 推进到 `auto_labeled`。
3. 自动创建下一阶段 `mano_opt` job。

## 8. 失败处理

临时问题用 release，不要 fail：

```http
POST /api/v1/jobs/{job_id}/release
```

```json
{
  "reason": "worker_shutdown"
}
```

`release` 会把非终态 job 放回 `queued`，可再次租取。

不可恢复问题才 fail：

```http
POST /api/v1/jobs/{job_id}/fail
```

```json
{
  "error": "missing_rgb_input",
  "result": {
    "worker_id": "auto_label_worker_01"
  }
}
```

`failed` 是终态。普通 `auto_label` job 失败后不会自动重试，也不会再被租取。需要人工排查或后端补 retry 机制。

## 9. Worker 主循环

```python
while True:
    leased = post("/api/v1/jobs/lease", {
        "type": "auto_label",
        "worker_id": WORKER_ID,
        "lease_seconds": 300,
    })

    job_id = leased["job"]["job_id"]
    payload = leased["payload"]

    post(f"/api/v1/jobs/{job_id}/heartbeat", {
        "worker_id": WORKER_ID,
        "lease_seconds": 300,
        "status": "running",
    })

    episode_dir = payload.get("resolved_data_path") or resolve_uri(payload["data_uri"])
    run_model_and_write_pred_2d(episode_dir, payload)

    pred_uri = payload["data_uri"].rstrip("/") + "/" + payload.get("prediction_dir", "pred_2d")
    post(f"/api/v1/jobs/{job_id}/complete", {
        "result": {
            "ok": True,
            "worker_id": WORKER_ID,
            "prediction_dir": payload.get("prediction_dir", "pred_2d"),
            "frames_predicted": payload.get("frames", []),
        },
        "artifacts": [{"kind": "pred_2d", "uri": pred_uri, "metadata": {"worker_id": WORKER_ID}}],
    })
```

生产实现必须捕获异常：临时异常调用 `release`，确定不可恢复再调用 `fail`。

## 10. 验收清单

- 能在阶段页面 `/workflow/stages/auto_label` 看到 worker 租取和完成记录。
- 每个 camera/frame 都生成 `.npy`。
- `.npy` shape 为 `(2,21,2)`，dtype 为 `float32`。
- 无手帧也有全 `-1` 输出。
- complete 后 episode 出现 `pred_2d` artifact。
- complete 后自动出现下一阶段 `mano_opt` job。
