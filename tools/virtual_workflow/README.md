# Workflow Test Tools

This directory contains a standalone Python fixture for checking the current
task backend and frontends against the configured NAS mount and placeholder
auto-label/QC workers.

The tool talks to the backend only through HTTP APIs. It does not import or
modify the existing `task_backend` or `label` modules.

## What It Provides

- A NAS root, defaulting to `/mnt/nas`, exposed as `nas://ego/...`.
- Upload workers that copy or materialize a captured episode into the
  configured NAS root. Upload completion marks episodes `uploaded`; pushing
  auto-label is a separate dashboard/API action.
- Auto-label workers that write `(2,21,2)` prediction `.npy` files by reusing
  `src/sync/hand_joint_gt_worker.py`. The virtual worker requires the same
  MediaPipe/Python environment as interaction handGT; missing dependencies,
  incompatible MediaPipe APIs, missing RGB frames, or frames with no detected
  hands fail the job instead of writing synthetic predictions.
- MANO workers that write low-precision episode MANO outputs from `pred_2d` and
  segment patch outputs from real `manual_2d`.
- QC workers that pass or fail with `--qc-fail-rate`, default `0.5`. Failed QC
  writes `qc/qc_report.json` and returns random 2-3 failed frame ranges, each
  10-20 frames long when the episode is long enough.

Manual labeling is not simulated by this worker. Failed segments must be opened
from the real Label entry in the main menu, saved by the label frontend into
`manual_2d/segments/<segment_id>`, and then completed by that frontend.

## Start Backend

From the repository root:

```bash
python3 task_backend/server.py
```

The tool loads `./.env` by default before reading command defaults. Shell
environment variables override values from the file.

Use the same `.env` file for backend and worker configuration:

```dotenv
ORBBEC_TASK_BACKEND_HOST=0.0.0.0
ORBBEC_TASK_BACKEND_PORT=8765
ORBBEC_TASK_BACKEND_DATA_ROOT=./task_backend_state_fullflow
ORBBEC_NAS_ENABLED=1
ORBBEC_NAS_ROOT=/mnt/nas
ORBBEC_NAS_URI_PREFIX=nas://ego
ORBBEC_NAS_MOUNTS_JSON={"nas://ego":"/mnt/nas"}
ORBBEC_AUTO_LABEL_AFTER_UPLOAD=1

ORBBEC_VIRTUAL_WORKFLOW_WORKERS=all
ORBBEC_WORKFLOW_NAS_ROOT=/mnt/nas
ORBBEC_WORKFLOW_NAS_URI_PREFIX=nas://ego
ORBBEC_VIRTUAL_WORKFLOW_QC_FAIL_RATE=0.5
ORBBEC_VIRTUAL_WORKFLOW_MAX_ITERATIONS=0
ORBBEC_VIRTUAL_WORKFLOW_STOP_AFTER_IDLE_ROUNDS=0
ORBBEC_VIRTUAL_WORKFLOW_IDLE_SLEEP=1.0
ORBBEC_VIRTUAL_WORKFLOW_IDLE_LOG_INTERVAL=30
```

For deterministic manual-label flow validation, set
`ORBBEC_VIRTUAL_WORKFLOW_QC_FAIL_RATE=1.0`.

## Seed Existing Label Data Into Manual Label Queue

This reads `label/task.jsonl`, materializes NAS placeholder data, and
creates queued manual segments through `/api/v1/dev/label/jobs`.

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py seed-label \
  --jsonl label/task.jsonl \
  --frames-per-segment 32 \
  --max-jobs 3 \
  --limit 1
```

The label frontend does not need a mount mapping. It requests the next backend
task, and the backend returns a resolved local path for `nas://ego`.

## Real Collection And Label Flow

Start the backend, then open the existing main-menu `Collection` page/app and
capture an episode normally. Collection confirm creates an `upload` job.

Start all virtual workers once. They read backend URL, NAS settings, QC rate,
loop limits, and worker selection from `.env`:

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py run-workers
```

Enable the controlled stages and push uploaded episodes into auto-label:

```bash
curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/stages/auto_label/enable \
  -H 'Content-Type: application/json' -d '{"updated_by":"virtual_workflow"}'
curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/stages/mano_opt/enable \
  -H 'Content-Type: application/json' -d '{"updated_by":"virtual_workflow"}'
curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/stages/qc/enable \
  -H 'Content-Type: application/json' -d '{"updated_by":"virtual_workflow"}'
curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/stages/manual_segment/enable \
  -H 'Content-Type: application/json' -d '{"updated_by":"virtual_workflow"}'

curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/episodes/push-auto-label \
  -H 'Content-Type: application/json' -d '{"scope":"all","pushed_by":"virtual_workflow"}'
```

If QC fails, open the existing main-menu `Label` page/app. The label frontend
leases the failed segments from the backend, opens the backend-resolved NAS
episode path, and writes:

```text
manual_2d/segments/<segment_id>/<camera>/<frame:05d>.npy
```

After the Label frontend completes a segment, the already-running virtual
workers consume the new segment-level `mano_opt` job and write the segment
patch.

Repeat Label completion and `mano-opt` until the episode finalizes. The backend
keeps the final NAS source map at:

```text
workflow/final_3d_sources.json
```

`default` and `all` workers both include only `upload`, `auto-label`,
`mano-opt`, and `qc`; manual labeling stays in the real Label frontend.

## Simulate Captured Upload Jobs

For smoke tests without opening Collection, seed upload jobs from a label JSONL:

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py seed-captured \
  --jsonl label/task.jsonl \
  --limit 1
```

Then run the same workflow worker commands above.

## Useful Options

- `ORBBEC_WORKFLOW_NAS_ROOT`: local directory for the NAS mount. It must match
  the backend mount for `ORBBEC_WORKFLOW_NAS_URI_PREFIX`.
- `ORBBEC_WORKFLOW_NAS_URI_PREFIX`: NAS URI prefix, default `nas://ego`.
- `ORBBEC_VIRTUAL_WORKFLOW_WORKERS`: `default`, `all`, or comma list:
  `upload,auto-label,mano-opt,qc`.
- `ORBBEC_VIRTUAL_WORKFLOW_QC_FAIL_RATE`: QC failure probability.
- `ORBBEC_VIRTUAL_WORKFLOW_STOP_AFTER_IDLE_ROUNDS`: use `0` for a long-running
  worker process.
- `ORBBEC_VIRTUAL_WORKFLOW_IDLE_LOG_INTERVAL`: print one idle log every N idle
  rounds; use `0` to silence idle logs.
- `seed-label --frames-per-segment N`: split one JSONL task into manual-label segments.
- `seed-label --max-jobs N`: seed only the first N manual-label segments.
- `--copy-source`: copy real episode folders if they exist.
- `--max-materialized-frames N`: limit generated placeholder frames for faster
  tests. `0` means all frames listed in the input payload.
- `--once`: process one available job and exit.
