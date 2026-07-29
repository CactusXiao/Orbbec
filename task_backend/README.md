# Collection Task Backend

Collection now uses a backend-owned task and episode allocator. The backend does
not need access to the capture machine's local files. The frontend still saves
local data on the capture machine as:

```text
<captureSaveRoot>/<subjectId>/<taskName>/episode_<N>
```

`N` is always the episode number returned by the backend reserve API.

The same process also owns a lightweight workflow/job store for downstream
upload, auto-label, QC, review, and manual labeling work. This is an incremental
extension of `task_backend/server.py`: the original collection API and JSON
progress state remain compatible, while new workflow state is stored in
`<dataRoot>/workflow.sqlite3` with Python's standard `sqlite3` module.

NAS storage is represented by a built-in virtual NAS uploader for development
and system validation. Collection still saves the episode locally first. After
the frontend confirms the local save, the backend creates an `upload` job and a
background worker copies that local episode directory into a backend-local
virtual NAS directory. The server records upload progress, verifies copied file
and byte counts, then switches the episode `data_uri` from `local://...` to
`nas://...` only after the copy succeeds.

Automatic labeling and automatic QC remain decoupled worker stages. The server
stores abstract URIs such as `local:///data/episode_001` or
`nas://orbbec-virtual/S001/...`, job status, and artifact registrations, but it
does not import model code or run QC/model inference itself.

## Start The Backend

From the repository root:

```bash
python3 task_backend/server.py
```

Startup configuration is read from `.env` by default. If `.env` is absent, the
backend falls back to `./task_backend_state`, `127.0.0.1`, and port `8765`.

Supported `.env` keys:

```dotenv
ORBBEC_TASK_BACKEND_HOST=127.0.0.1
ORBBEC_TASK_BACKEND_PORT=8765
ORBBEC_TASK_BACKEND_DATA_ROOT=./task_backend_state
# Optional workflow SQLite override:
# ORBBEC_WORKFLOW_DB=./task_backend_state/workflow.sqlite3

# Optional virtual NAS override:
# ORBBEC_VIRTUAL_NAS_ENABLED=1
# ORBBEC_VIRTUAL_NAS_ROOT=./task_backend_state/virtual_nas
# ORBBEC_VIRTUAL_NAS_URI_PREFIX=nas://orbbec-virtual

# Upload success waits for an explicit push before queuing auto_label jobs.
# Set this to 1 only for legacy fully automatic local tests:
# ORBBEC_AUTO_LABEL_AFTER_UPLOAD=0
# ORBBEC_AUTO_LABEL_BATCH_SIZE=200

# Optional seed file for the setup page:
# ORBBEC_TASK_BACKEND_TASK_FILE=./tasks.json
# ORBBEC_TASK_BACKEND_STATE_FILE=./task_backend_state/progress_state.json
```

If the capture frontend runs on another machine, bind to an address reachable by
the capture machine in `.env`:

```dotenv
ORBBEC_TASK_BACKEND_HOST=0.0.0.0
```

Then point the frontend `baseUrl` to the backend machine's IP.

Command-line flags such as `--host`, `--port`, `--data-root`, `--task-file`,
and `--state-file` are still accepted for temporary overrides. Use `--env-file`
to load a different env file.

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

The current detail pages show backend-owned metadata plus workflow state:
task definitions, subject IDs, reservation IDs, episode numbers, status,
timestamps, idempotency keys, local capture paths, upload job status, upload
percentage, copied bytes/files, NAS URI, and upload errors.

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
8. Backend confirm creates an `upload` job. The virtual NAS uploader copies the
   already-saved local episode asynchronously and records progress in the
   workflow database.
9. Upload success marks the episode `uploaded`. It does not queue
   `auto_label` by default.
10. A dashboard, task-page, episode-page, or API push creates batched
    `auto_label` jobs according to `ORBBEC_AUTO_LABEL_BATCH_SIZE`. Repeating
    the push for the same episode is idempotent.
11. The collection UI returns to READY/IDLE after backend confirm succeeds. It
   keeps polling upload progress by reservation ID and displays the latest NAS
   status without blocking the next capture.
12. If backend confirm fails, the UI enters `backend-sync-pending`; retry Confirm
   uses the same reservation and idempotency key.
13. Reset/Delete deletes local episode data and releases the reservation; it does
   not increase `completed`.
14. The capture page has a `Tasks` button for returning to the standalone task
   selection page; task selection is not mixed into the capture view.
15. ESC, Menu, Tasks, Config, and camera-error Exit paths show a confirmation
    dialog before stopping cameras or leaving collection when applicable.

The backend confirm endpoint is idempotent: repeating the same
`idempotency_key` for the same reservation returns current progress without
double-counting the episode.

## Collection API Compatibility

Existing collection clients can keep using:

```text
GET  /api/v1/tasks
POST /api/v1/episodes/reserve
POST /api/v1/episodes/confirm
POST /api/v1/episodes/release
```

The server also exposes aliases for newer clients:

```text
GET  /api/v1/collection/tasks
POST /api/v1/collection/episodes/reserve
POST /api/v1/collection/episodes/confirm
POST /api/v1/collection/episodes/release
```

When collection confirms an episode, the workflow sidecar records an Episode
with status `captured` and queues an `upload` job. The built-in virtual NAS
uploader leases this job, copies local data to the virtual NAS, updates progress,
registers a `nas_episode` artifact, and marks the episode `uploaded` only after
the verified copy completes. `auto_label` jobs are created only after an
explicit push from the dashboard, task page, episode page, or workflow API.
Manual labeling completion stops at
`manual_labeled`; any later review or return flow is left to a future policy.

Upload status can be read by reservation/episode ID:

```text
GET /api/v1/episodes/<reservation_id>/upload
GET /api/v1/collection/episodes/<reservation_id>/upload
```

The response includes the upload job status, phase, percentage, byte and file
counts, local path, NAS URI, and error string.

## Workflow And Label APIs

Generic workers lease jobs by type:

```text
POST /api/v1/jobs/lease
POST /api/v1/jobs/{job_id}/heartbeat
POST /api/v1/jobs/{job_id}/complete
POST /api/v1/jobs/{job_id}/fail
POST /api/v1/jobs/{job_id}/release
```

`POST /api/v1/jobs/lease` accepts JSON like:

```json
{"type":"auto_label","worker_id":"auto_label_stub_01","lease_seconds":300}
```

Manual labeling uses a narrower API:

```text
POST /api/v1/label/jobs/lease
GET  /api/v1/label/jobs/{job_id}
POST /api/v1/label/jobs/{job_id}/heartbeat
POST /api/v1/label/jobs/{job_id}/complete
POST /api/v1/label/jobs/{job_id}/release
```

Lease semantics:

- only `queued` jobs, or jobs with an expired lease, can be leased;
- `auto_label`, `qc`, and `manual_label` leases are paused by default and return
  `HTTP 409` with `leasing disabled for job type: <type>` until enabled;
- lease writes `lease_owner`, `lease_until`, and `status=leased`;
- heartbeat extends the lease and may mark the job `running`;
- complete is idempotent;
- fail records an error and increments `attempt`;
- release clears the lease and returns unfinished work to `queued`.

Workflow stage controls and snapshots:

```text
GET  /api/v1/workflow/stages/<auto_label|qc|manual_label>
POST /api/v1/workflow/stages/<job_type>/enable
POST /api/v1/workflow/stages/<job_type>/disable
```

Push uploaded episodes into automatic labeling:

```text
POST /api/v1/workflow/episodes/push-auto-label
```

The push body may target one `episode_id`, one `task_name`, or all eligible
episodes with `{"scope":"all"}`. An episode is eligible when it is `uploaded`,
has a NAS/data URI, and has no existing `auto_label` job.

`manual_label` lease moves the Episode to `manual_labeling`; successful
completion moves it to `manual_labeled` and can register a `corrected_2d`
artifact.

## Development Manual Label Jobs

Until QC can create manual work automatically, use this temporary development
endpoint:

```text
POST /api/v1/dev/label/jobs
```

Example with local data:

```bash
curl -s http://127.0.0.1:8765/api/v1/dev/label/jobs \
  -H 'Content-Type: application/json' \
  -d '{
    "local_path": "/Users/cactusxiao/data/S001/pick_object/episode_000456",
    "subject_id": "S001",
    "task_name": "pick_object",
    "episode_id": "episode_000456",
    "cameras": ["camera_01", "camera_02"],
    "frames": [120, 121, 122, 123],
    "rgb_path_template": "{camera}/RGB/{frame:05d}.png",
    "prediction_dir": "pred_2d",
    "correction_dir": "corrected_2d"
  }'
```

The endpoint converts `local_path` to a `local://` data URI. It is for local
smoke tests and should be replaced by QC/worker-generated jobs later.

There is also a development-only generic helper for stubbing `upload`,
`auto_label`, `qc`, or `review` jobs:

```text
POST /api/v1/dev/jobs
```

Completing an `auto_label` job registers or accepts `pred_2d` and queues `qc`
after all auto-label batches for that episode succeed. Completing a `qc` job
registers or accepts `qc_report`; `{"result":{"passed":false}}` leaves the
episode `qc_failed` and queues one `manual_label` job.

## Label Frontend

The label GUI now defaults to server-driven work. On the home page enter:

```json
{
  "backend_url": "http://127.0.0.1:8765",
  "operator_id": "labeler_01",
  "mounts": {
    "nas://orbbec-dataset": "/Volumes/orbbec-dataset"
  }
}
```

Click `Get Next Task` to lease the next `manual_label` job from
`/api/v1/label/jobs/lease`. Legacy JSONL remains available for debugging, but it
is no longer the default task source.

The collection frontend also exposes a `Manual Label` button on the Collection
Config page and the Task Select page. It launches the same `python3 -m
label.main` GUI as a separate window and pre-fills the backend URL from the
collection config. Set `ORBBEC_LABEL_OPERATOR_ID` or use the current
`subject_id` as the operator hint; set `ORBBEC_LABEL_MOUNTS_JSON` when a
NAS-style URI needs a local mount mapping.

For local smoke tests, a payload with
`"data_uri":"local:///Users/cactusxiao/data/S001/pick_object/episode_000456"`
requires no mount mapping. A NAS-style URI such as
`nas://orbbec-dataset/S001/pick_object/episode_000456` resolves through the
configured mount prefix.

Label confirmation still writes corrected 2D arrays to:

```text
corrected_2d/<camera>/<frame:05d>.npy
```

After all frames in the leased job are complete, the GUI calls the backend
complete endpoint and registers a `corrected_2d` artifact. The local progress
CSV is retained only as a temporary cache/legacy aid.

## Artifact Kinds

Current artifact kind names:

- `pred_2d`: automatic or manual pre-label 2D output.
- `corrected_2d`: human-corrected 2D output.
- `kp3d`: 3D output.
- `qc_report`: quality-control report.

If an existing auto-label pipeline emits `kp2d`, adapt it at the artifact
boundary to `pred_2d`; do not couple model internals into the server.
