# Virtual Workflow Test Tools

This directory contains a standalone Python fixture for checking the current
task backend and frontends without depending on real NAS, auto-label, or QC
systems.

The tool talks to the backend only through HTTP APIs. It does not import or
modify the existing `task_backend` or `label` modules.

## What It Simulates

- A local NAS, stored under `task_backend_state/virtual_nas` by default,
  exposed as `nas://orbbec-virtual/...`. This matches the backend's default
  virtual NAS mount.
- Upload workers that copy or materialize a captured episode into the virtual
  NAS. Upload completion marks episodes `uploaded`; pushing auto-label is a
  separate dashboard/API action.
- Auto-label workers that write low-precision `(2,21,2)` prediction `.npy`
  files. They use the interaction handGT-style lightweight MediaPipe path when
  those optional libraries are installed, and fall back to a deterministic
  visible hand skeleton otherwise.
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

The default backend URL used by the tool is `http://127.0.0.1:8765`. Override it
with `--backend-url` or `ORBBEC_TASK_BACKEND_URL`.

## Seed Existing Label Data Into Manual Label Queue

This reads `label/task.jsonl`, materializes local NAS placeholder data, and
creates queued manual segments through `/api/v1/dev/label/jobs`.

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py seed-label \
  --jsonl label/task.jsonl \
  --use-nas \
  --frames-per-job 32 \
  --max-jobs 3 \
  --limit 1
```

The label frontend does not need a mount mapping. It requests the next backend
task, and the backend returns a resolved local path for `nas://orbbec-virtual`.

## Real Collection And Label Flow

Start the backend, then open the existing main-menu `Collection` page/app and
capture an episode normally. Collection confirm creates an `upload` job.

Run the virtual upload worker against the same NAS root and URI prefix as the
backend:

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py run-workers \
  --workers upload \
  --max-iterations 20 \
  --stop-after-idle-rounds 3
```

Enable the controlled stages, push uploaded episodes into auto-label, and run
the virtual auto-label, MANO, and QC workers:

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

python3 tools/virtual_workflow/orbbec_virtual_workflow.py run-workers \
  --workers auto-label,mano-opt,qc \
  --qc-fail-rate 0.5 \
  --max-iterations 20 \
  --stop-after-idle-rounds 3
```

If QC fails, open the existing main-menu `Label` page/app. The label frontend
leases the failed segments from the backend, opens the backend-resolved NAS
episode path, and writes:

```text
manual_2d/segments/<segment_id>/<camera>/<frame:05d>.npy
```

After the Label frontend completes a segment, run the virtual MANO worker again
to consume the real `manual_2d` files and write the segment patch:

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py run-workers \
  --workers mano-opt \
  --max-iterations 20 \
  --stop-after-idle-rounds 3
```

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

Then run the same virtual worker commands above.

## Useful Options

- `--nas-root PATH`: local directory for the virtual NAS. It must match the
  backend mount for `--nas-uri-prefix`.
- `--nas-uri-prefix URI`: NAS URI prefix, default `nas://orbbec-virtual`.
- `seed-label --frames-per-job N`: split one JSONL task into frame batches.
- `seed-label --max-jobs N`: seed only the first N manual-label batches.
- `--copy-source`: copy real episode folders if they exist.
- `--max-materialized-frames N`: limit generated placeholder frames for faster
  tests. `0` means all frames listed in the input payload.
- `--once`: process one available job and exit.
