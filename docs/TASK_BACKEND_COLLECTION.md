# Collection Task Backend

Collection now uses a backend-owned task and episode allocator. The frontend
still saves local data as:

```text
<saveRoot>/<subjectId>/<taskName>/episode_<N>
```

`N` is always the episode number returned by the backend reserve API.

## Start The Backend

From the repository root:

```bash
python3 scripts/task_backend_server.py \
  --save-root /path/to/saveRoot \
  --task-file tasks.json \
  --host 127.0.0.1 \
  --port 8765
```

If `--task-file` is omitted, the server searches:

1. `<saveRoot>/task.json`
2. `<saveRoot>/tasks.json`
3. `./task.json`
4. `./tasks.json`

Backend state is stored in `<saveRoot>/progress_state.json`. Updates are written
to a temporary file and atomically renamed. On Linux, a sidecar lock file is used
with `fcntl` to protect concurrent frontends.

## Task File Format

The existing object-style `tasks.json` is supported:

```json
{
  "hand_shape_calibration": {
    "repeat_times": 10,
    "task_description_cn": "任务描述",
    "task_description_en": "Task description"
  }
}
```

The backend maps:

- key or `task_name` to `task_name`
- `repeat_times` or `total` to required episode count
- `task_description_cn` or `description_cn` to Chinese description
- `task_description_en` or `description_en` to English description

Array form is also accepted:

```json
{
  "tasks": [
    {
      "task_name": "hand_shape_calibration",
      "total": 10,
      "description_cn": "任务描述",
      "description_en": "Task description"
    }
  ]
}
```

## Frontend Configuration

`src/sync/config.json` contains:

```json
"taskBackend": {
  "enabled": true,
  "baseUrl": "http://127.0.0.1:8765",
  "timeoutMs": 3000
}
```

Environment overrides:

```bash
export ORBBEC_TASK_BACKEND_URL=http://127.0.0.1:8765
export ORBBEC_TASK_BACKEND_TIMEOUT_MS=3000
export ORBBEC_TASK_BACKEND_ENABLED=1
```

When entering Collection, the frontend must reach the backend and fetch all
tasks. If the backend is unavailable, reserve fails, confirm fails, or release
fails, the UI shows an explicit error and does not advance completed progress
locally.

## Runtime Flow

1. The Config page `Load Tasks` action fetches all backend tasks and their
   `completed/total` progress.
2. The standalone Task Select page lets the operator choose exactly one
   unfinished task.
3. `Enter Capture` starts cameras and opens the capture page for only that
   selected task.
4. Start first reserves an episode from the backend.
5. Local capture writes to `saveRoot/subjectId/taskName/episode_N`.
6. Confirm finalizes local data, then confirms the reservation with an
   idempotency key.
7. If backend confirm fails, the UI enters `backend-sync-pending`; retry Confirm
   uses the same reservation and idempotency key.
8. Reset/Delete deletes local episode data and releases the reservation; it does
   not increase `completed`.
9. The capture page has a `Tasks` button for returning to the standalone task
   selection page; task selection is not mixed into the capture view.
10. ESC, Menu, Tasks, Config, and camera-error Exit paths show a confirmation
    dialog before stopping cameras or leaving collection when applicable.

The backend confirm endpoint is idempotent: repeating the same
`idempotency_key` for the same reservation returns current progress without
double-counting the episode.
