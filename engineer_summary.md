# Engineer Summary — Phase 9 Crash Hunt (Depth Sweep)

**Focus:** FNO-STABILITY-OVERHAUL-001, Phase 9 Task 1: Crash Hunt depth sweep
**Branch:** fno2-phase8-optimizers
**Date:** 2026-01-29

## What was done

1. **Disk remediation:** Deleted `outputs/fno_hyperparam_study_e10` (18 GB), `fno_hyperparam_study` and rerun variants (~2 GB each), and accumulated `training_outputs/checkpoints/` (32 GB) to free ~50 GB total. Disk went from 100% (866 MB free) to 92% (38 GB free).

2. **Warmup tests:** Ran `test_wrapper_handles_stable_hybrid` and `test_wrapper_passes_max_hidden_channels` — both passed. Log archived as `pytest_warmup.log`.

3. **Dataset verification:** Confirmed all 9 depth/seed directories have valid symlinks to `outputs/grid_lines_stage_a/arm_control/datasets`. No metadata.json found but `--set-phi` is CLI-driven.

4. **Crash Hunt execution:** Ran 3 depths × 3 seeds = 9 runs of the control hybrid (norm clip 1.0, max_hidden_channels=512, 30 epochs, MAE loss):
   - **Depth 4:** 3/3 completed. Seed 20260129 CRASHED (amp_ssim=0.277, constant amplitude). Seeds 20260128/30 survived (amp_ssim 0.898/0.768).
   - **Depth 6:** 3/3 completed and stable. amp_ssim 0.78–0.80 across all seeds.
   - **Depth 8:** 3/3 OOM (18 GiB CUDA allocation on 24 GB GPU).

5. **Stats capture:** Ran `stage_a_dump_stats.py` for all 6 successful runs; wrote OOM stub JSONs for depth 8.

6. **Crash summary:** Generated `crash_hunt_summary.json` and `crash_hunt_summary.md` with P_crash statistics.

7. **Doc updates:**
   - `implementation.md` Phase 9 status updated with Crash Hunt findings
   - `docs/findings.md` — added STABLE-CRASH-DEPTH-001
   - `docs/strategy/mainstrategy.md` §3 — replaced prep status with complete results
   - `docs/fix_plan.md` — added Crash Hunt execution entry + FSM update

8. **Shootout staging:** Created `outputs/grid_lines_shootout/{control,stable_layerscale,optimizer}/seed{A,B,C}` with Stage A dataset symlinks.

## Key Finding

**DEPTH_CRASH = 4** (not 6 as hypothesized). The crash is stochastic and seed-dependent at depth 4, while depth 6 (with channel cap) is paradoxically more stable. This challenges the assumption that deeper models are less stable — the channel cap may provide implicit regularization.

## Files changed
- `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/` (artifacts hub: README, logs, stats, summaries)
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Phase 9 status)
- `docs/findings.md` (STABLE-CRASH-DEPTH-001)
- `docs/strategy/mainstrategy.md` (§3 Crash Hunt results)
- `docs/fix_plan.md` (attempts history + FSM)
- `engineer_summary.md` (this file)

## Tests
- Warmup selectors: 2/2 passed (`pytest_warmup.log`)
- No code changes requiring additional test runs

## Open questions / next steps
- Shootout at depth 4 with more seeds (≥5) to get tighter P_crash estimates
- Consider dual depth 4+6 Shootout to stress-test at scale
- stable_hybrid (LayerScale) collapses at all depths (STABLE-LS-001), so Shootout compares control hybrid variants only
