# FNO Hyperparam Study — Per-Run Visuals (Design)

**Date:** 2026-01-29  
**Status:** Draft (approved by user)  
**Owner:** Codex (with user approval)

## Goal
Ensure each hyperparameter sweep run produces a reconstruction PNG in its own run directory, inline with the sweep (no post-processing pass required).

## Scope
- **In scope:** `scripts/studies/fno_hyperparam_study.py` inline visual generation per run.
- **Out of scope:** New rendering scripts, changes to model/training behavior, changes to global compare visuals.

## Current Behavior
- Each run saves `recons/<label>/recon.npz` (via `save_recon_artifact` in the torch runner).
- The sweep only produces a global `visuals/compare_amp_phase.png` at the study root (if invoked).
- Individual run directories do **not** receive per-run PNGs.

## Proposed Change
After each successful run in `run_sweep()`, call:
```
render_grid_lines_visuals(run_root, order=(f"pinn_{arch}",))
```
This uses the existing renderer in `ptycho/workflows/grid_lines_workflow.py` to emit:
```
run_root/visuals/amp_phase_pinn_fno.png
# or
run_root/visuals/amp_phase_pinn_hybrid.png
```

## Error Handling
- If `recon.npz` is missing or lacks `amp/phase`, the renderer already skips that label.
- The sweep should log a warning and continue, so long runs are not blocked by a single bad artifact.

## Testing Plan (TDD)
- Add a focused unit test in `tests/test_fno_hyperparam_study.py`:
  - Create a temp run dir with a minimal `recons/<label>/recon.npz` containing `amp` and `phase`.
  - Invoke the sweep’s inline visual step (or a small helper in the study file).
  - Assert `run_dir/visuals/amp_phase_<label>.png` exists.
- Verify RED before implementation, then GREEN after wiring the call.

## Notes
- No changes to training or inference config paths.
- No change to comparison composite logic (requires `gt`, which per-run directories do not include).
