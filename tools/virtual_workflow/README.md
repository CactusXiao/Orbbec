# Virtual Workflow Test Tools

This directory contains a standalone Python fixture for checking the current
task backend and frontends without depending on real NAS, auto-label, or QC
systems.

The tool talks to the backend only through HTTP APIs. It does not import or
modify the existing `task_backend` or `label` modules.

## What It Simulates

- A local NAS, stored under `.virtual_nas` by default, exposed as
  `nas://orbbec-virtual/...`.
- Upload workers that copy or materialize a captured episode into the virtual
  NAS. The backend queues batched `auto_label` jobs when upload completes.
- Auto-label workers that write placeholder 2D prediction `.npy` files, then
  enqueue a `qc` job.
- QC workers that randomly pass or fail. Failed QC creates a queued
  `manual_label` job so the label frontend can lease it.
- Optional manual label workers that complete queued manual jobs with
  placeholder corrected `.npy` artifacts.

## Start Backend

From the repository root:

```bash
python3 task_backend/server.py
```

The default backend URL used by the tool is `http://127.0.0.1:8765`. Override it
with `--backend-url` or `ORBBEC_TASK_BACKEND_URL`.

## Seed Existing Label Data Into Manual Label Queue

This reads `label/task.jsonl`, materializes local NAS placeholder data, and
creates queued `manual_label` jobs through `/api/v1/dev/label/jobs`.

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py seed-label \
  --jsonl label/task.jsonl \
  --use-nas \
  --frames-per-job 32 \
  --max-jobs 3 \
  --limit 1
```

If you want the label frontend to resolve `nas://orbbec-virtual/...`, set its
mount mapping JSON to:

```json
{
  "nas://orbbec-virtual": "/Users/cactusxiao/Desktop/demo/Orbbec/.virtual_nas"
}
```

Then open the label frontend and request the next backend task.

## Simulate Capture To QC

Seed upload jobs from the same label JSONL:

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py seed-captured \
  --jsonl label/task.jsonl \
  --limit 1
```

Run the virtual upload, auto-label, and QC workers:

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py run-workers \
  --workers default \
  --qc-fail-rate 0.8 \
  --max-iterations 20 \
  --stop-after-idle-rounds 3
```

`default` workers include `upload`, `auto-label`, and `qc`. They intentionally
leave failed QC episodes in the manual label queue for the label frontend to
pick up.

## Complete Manual Labels Virtually

To drain queued manual label jobs without opening the real label UI:

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py run-workers \
  --workers manual-label \
  --max-iterations 20 \
  --stop-after-idle-rounds 3
```

Use `--workers all` if you want one process to run upload, auto-label, QC, and
manual label simulation together.

## Useful Options

- `--nas-root PATH`: local directory for the virtual NAS.
- `--nas-uri-prefix URI`: NAS URI prefix, default `nas://orbbec-virtual`.
- `seed-label --frames-per-job N`: split one JSONL task into frame batches.
- `seed-label --max-jobs N`: seed only the first N manual-label batches.
- `--copy-source`: copy real episode folders if they exist.
- `--max-materialized-frames N`: limit generated placeholder frames for faster
  tests. `0` means all frames listed in the input payload.
- `--once`: process one available job and exit.
