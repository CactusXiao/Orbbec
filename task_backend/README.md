# Collection Task Backend

Collection now uses a backend-owned task and episode allocator. The backend does
not need access to the capture machine's local files. The frontend still saves
local data on the capture machine as:

```text
<captureSaveRoot>/<subjectId>/<taskName>/episode_<N>
```

`N` is always the episode number returned by the backend reserve API.

## Start The Backend

From the repository root:

```bash
python3 task_backend/server.py \
  --data-root /var/lib/orbbec-task-backend \
  --task-file tasks.json \
  --host 0.0.0.0 \
  --port 8765
```

Use `--host 127.0.0.1` only when the frontend runs on the same machine. For a
separate backend machine, bind to an address reachable by the capture machine
such as `0.0.0.0`, and point the frontend `baseUrl` to the backend machine's IP.

If `--task-file` is omitted, the server searches:

1. `<dataRoot>/task.json`
2. `<dataRoot>/tasks.json`
3. `./task.json`
4. `./tasks.json`

Backend state is stored in `<dataRoot>/progress_state.json`. Updates are written
to a temporary file and atomically renamed. On Linux, a sidecar lock file is used
with `fcntl` to protect concurrent frontends.

`--save-root` is still accepted as a deprecated alias for `--data-root` so older
startup commands keep working, but new deployments should not use the capture
save directory as backend state storage.

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
  "baseUrl": "http://<backend-host>:8765",
  "timeoutMs": 3000
}
```

Environment overrides:

```bash
export ORBBEC_TASK_BACKEND_URL=http://<backend-host>:8765
export ORBBEC_TASK_BACKEND_TIMEOUT_MS=3000
export ORBBEC_TASK_BACKEND_ENABLED=1
```

When entering Collection, the frontend must reach the backend and fetch all
tasks. If the backend is unavailable, reserve fails, confirm fails, or release
fails, the UI shows an explicit error and does not advance completed progress
locally.

The Collection page's `save_path` remains a frontend-local setting. It does not
have to match any path on the backend machine.

## Runtime Flow

1. The Config page `Load Tasks` action fetches all backend tasks and their
   `completed/total` progress.
2. The standalone Task Select page lets the operator choose exactly one
   unfinished task.
3. `Enter Capture` starts cameras and opens the capture page for only that
   selected task.
4. Start first reserves an episode from the backend.
5. Local capture writes to `captureSaveRoot/subjectId/taskName/episode_N`.
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
