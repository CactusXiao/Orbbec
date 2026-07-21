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
python3 task_backend/server.py
```

By default this stores setup and instance state in `./task_backend_state`, binds
to `127.0.0.1`, and uses port `8765`.

If the capture frontend runs on another machine, bind to an address reachable by
the capture machine:

```bash
python3 task_backend/server.py --host 0.0.0.0
```

Then point the frontend `baseUrl` to the backend machine's IP.

After the process starts, open `http://<backend-host>:8765/` in a browser. The
server first shows a setup page. Select one registered `tasks.json` and one
instance, or create a new instance, then click start. The collection API is not
available until this setup step is completed.

If `--task-file ./tasks.json` is provided, that file is only used to seed the
setup page. It no longer bypasses setup or starts the collection API directly.
If `--task-file` is omitted, the server searches for an initial task file in:

1. `<dataRoot>/task.json`
2. `<dataRoot>/tasks.json`
3. `./task.json`
4. `./tasks.json`

The setup registry is stored in `<dataRoot>/task_backend_registry.json`.
Instance progress is stored separately, normally under:

```text
<dataRoot>/instances/<task_file_id>/<instance_id>/progress_state.json
```

Updates are written to a temporary file and atomically renamed. On Linux, a
sidecar lock file is used with `fcntl` to protect concurrent frontends.

`--save-root` is still accepted as a deprecated alias for `--data-root` so older
startup commands keep working, but new deployments should not use the capture
save directory as backend state storage.

## Task Files And Instances

The setup page supports multiple task files and multiple instances per task
file:

- A task file is a server-local path to a `tasks.json`.
- An instance is an isolated progress namespace for that task file.
- Each task file can have many instances, such as `default`, `debug-a`, or
  `formal-round-1`.
- Different instances do not share reservations, confirmed progress, released
  episodes, or idempotency keys.

Once an instance is started, the backend process is locked to that exact task
file and instance. The setup page will not allow switching or adding another
instance while the collection API is live. To change the task file or instance,
stop the backend process completely and start it again.

## Web Dashboard

The same service exposes a lightweight browser dashboard:

- Before setup is completed, `http://127.0.0.1:8765/` shows the setup page.
- After one instance is started, `http://127.0.0.1:8765/` shows the overall task
  and episode summary for the locked instance.
- `http://127.0.0.1:8765/tasks/<task_name>` shows one task and its episodes.
- `http://127.0.0.1:8765/episodes/<reservation_id>` shows one episode detail page.

The current detail pages use backend-owned metadata only: task definitions,
subject IDs, reservation IDs, episode numbers, status, timestamps, idempotency
keys, and the local capture path reported by the capture host. NAS-backed file
indexes and quality inspection results can be added to the episode detail model
later without changing the collection API.

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

1. Start `task_backend/server.py` and choose one task file instance on the setup
   page.
2. The Config page `Load Tasks` action fetches all backend tasks and their
   `completed/total` progress.
3. The standalone Task Select page lets the operator choose exactly one
   unfinished task.
4. `Enter Capture` starts cameras and opens the capture page for only that
   selected task.
5. Start first reserves an episode from the backend.
6. Local capture writes to `captureSaveRoot/subjectId/taskName/episode_N`.
7. Confirm finalizes local data, then confirms the reservation with an
   idempotency key.
8. If backend confirm fails, the UI enters `backend-sync-pending`; retry Confirm
   uses the same reservation and idempotency key.
9. Reset/Delete deletes local episode data and releases the reservation; it does
   not increase `completed`.
10. The capture page has a `Tasks` button for returning to the standalone task
   selection page; task selection is not mixed into the capture view.
11. ESC, Menu, Tasks, Config, and camera-error Exit paths show a confirmation
    dialog before stopping cameras or leaving collection when applicable.

The backend confirm endpoint is idempotent: repeating the same
`idempotency_key` for the same reservation returns current progress without
double-counting the episode.
