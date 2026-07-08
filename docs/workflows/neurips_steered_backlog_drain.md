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
- `docs/backlog/roadmap_gate.json`
- `state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json`
- active backlog items under `docs/backlog/active/*.md`
- workflow-managed run state under `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/`

## Queue Lifecycle

1. Build a deterministic raw manifest from `docs/backlog/active/*.md`
2. Reconcile the raw manifest against `docs/backlog/roadmap_gate.json` and emit a derived eligible manifest
3. If eligible items exist, select one next item with provider judgment from the eligible manifest only
4. If no eligible item exists because authorized roadmap work is missing from the queue, draft one controlled backlog-gap item and loop
5. Run narrow roadmap sync for the selected item
6. Move the item `active -> in_progress` only after roadmap sync accepts it
7. Draft and review a fresh plan
8. Rewrite the item's `plan_path` to that fresh approved plan
9. Run implementation, deterministic targeted checks, and implementation review in one local phase
10. Move the item to `done` on success, or leave it in `in_progress` with blocker
   state on failure

The workflow keeps `state/NEURIPS-HYBRID-RESNET-2026/backlog_drain/current_roadmap_path.txt`
as the authoritative roadmap pointer after each roadmap-sync decision.

`docs/backlog/roadmap_gate.json` is the deterministic phase gate. The selector
does not decide phase legality; it ranks only already-eligible items. The gap
drafter may create a missing active backlog item for already-authorized roadmap
work, but it must not edit the roadmap or advance to later phases.

The raw manifest and eligible manifest are intentionally not equal. The raw
manifest is gate input and provenance; the eligible manifest is the downstream
authority for both selection and active selected-item execution. The eligible
manifest records `source_manifest_path`, and active selected-item materialization
rejects manifests that do not carry gated-manifest lineage or do not contain the
selected item.

## Launch

Run from `/home/ollie/Documents/PtychoPINN` in the `ptycho311` environment.

```bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
export PYTHONPATH=/home/ollie/Documents/agent-orchestration
python -m orchestrator run \
  workflows/examples/neurips_steered_backlog_drain.yaml \
  --input steering_path=docs/steering.md \
  --input design_path=docs/plans/2026-04-20-neurips-hybrid-resnet-submission-design.md \
  --input roadmap_path=docs/plans/2026-04-20-neurips-hybrid-resnet-submission-roadmap.md \
  --input roadmap_gate_path=docs/backlog/roadmap_gate.json \
  --input progress_ledger_path=state/NEURIPS-HYBRID-RESNET-2026/progress_ledger.json \
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
