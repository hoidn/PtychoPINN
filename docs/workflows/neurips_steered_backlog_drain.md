# NeurIPS Steered Backlog Drain Runbook

This runbook covers the steering-aware NeurIPS backlog drain workflow:

- workflow: `workflows/examples/neurips_steered_backlog_drain.yaml`
- queue roots:
  - `docs/backlog/active/`
  - `docs/backlog/in_progress/`
  - `docs/backlog/done/`
  - `docs/backlog/paused/`

Read `docs/workflows/orchestration_start_here.md` first for workflow/step/prompt
ownership boundaries.

## What The Workflow Consumes

- `docs/steering.md`
- the approved NeurIPS design and roadmap
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- active backlog items under `docs/backlog/active/*.md`
- workflow-managed run state under `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/`

## Queue Lifecycle

1. Build a deterministic manifest from `docs/backlog/active/*.md`
2. Select one next item with provider judgment
3. Move the item `active -> in_progress`
4. Run narrow roadmap sync
5. Draft and review a fresh plan
6. Rewrite the item's `plan_path` to that fresh approved plan
7. Run implementation, deterministic targeted checks, and implementation review in one local phase
8. Move the item to `done` on success, or leave it in `in_progress` with blocker
   state on failure

The workflow keeps `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/current_roadmap_path.txt`
as the authoritative roadmap pointer after each roadmap-sync decision.

## Launch

Run from `/home/ollie/Documents/PtychoPINN` in the `ptycho311` environment.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
export PYTHONPATH=/home/ollie/Documents/agent-orchestration
python -m orchestrator run \
  workflows/examples/neurips_steered_backlog_drain.yaml \
  --debug \
  --stream-output
```

## Resume

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
export PYTHONPATH=/home/ollie/Documents/agent-orchestration
python -m orchestrator resume <run_id> --debug --stream-output
```

Prefer `resume` when a run has already passed earlier approval gates and later
fails downstream.
