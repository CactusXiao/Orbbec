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
  NAS. Upload completion marks episodes `uploaded`; pushing auto-label is a
  separate dashboard/API action.
- Auto-label workers that write placeholder 2D prediction `.npy` files, then
  let the backend enqueue an episode-level `mano_opt` job.
- MANO workers that write placeholder episode MANO outputs and segment patch
  outputs.
- QC workers that randomly pass or fail. Failed QC lets the backend create a
  set of `pending_manual` segments so the label frontend can lease them.
- Optional manual workers that complete queued segments with placeholder
  `manual_2d` artifacts.

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

## Simulate Capture To QC

Seed upload jobs from the same label JSONL:

```bash
python3 tools/virtual_workflow/orbbec_virtual_workflow.py seed-captured \
  --jsonl label/task.jsonl \
  --limit 1
```

Run the virtual upload, auto-label, and QC workers:

```bash
curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/stages/auto_label/enable \
  -H 'Content-Type: application/json' -d '{"updated_by":"virtual_workflow"}'
curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/stages/mano_opt/enable \
  -H 'Content-Type: application/json' -d '{"updated_by":"virtual_workflow"}'
curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/stages/qc/enable \
  -H 'Content-Type: application/json' -d '{"updated_by":"virtual_workflow"}'
curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/stages/manual_segment/enable \
  -H 'Content-Type: application/json' -d '{"updated_by":"virtual_workflow"}'

python3 tools/virtual_workflow/orbbec_virtual_workflow.py run-workers \
  --workers upload \
  --max-iterations 20 \
  --stop-after-idle-rounds 3

curl -s -X POST http://127.0.0.1:8765/api/v1/workflow/episodes/push-auto-label \
  -H 'Content-Type: application/json' -d '{"scope":"all","pushed_by":"virtual_workflow"}'

python3 tools/virtual_workflow/orbbec_virtual_workflow.py run-workers \
  --workers auto-label,mano-opt,qc \
  --qc-fail-rate 0.8 \
  --max-iterations 20 \
  --stop-after-idle-rounds 3
```

`default` workers include `upload`, `auto-label`, and `qc`; use it only after
the uploaded episodes have already been pushed into `auto_label`. Failed QC
episodes are left in the manual label queue for the label frontend to pick up.

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
