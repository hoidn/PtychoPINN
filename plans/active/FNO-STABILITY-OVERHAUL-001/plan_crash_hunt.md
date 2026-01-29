# Stable Hybrid Crash Hunt & Shootout Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Measure the stochastic failure rate (P_crash) of deep hybrid/LayerScale variants by executing the Crash Hunt (depth sweep) and three-arm Shootout described in `docs/strategy/mainstrategy.md §3`, enforcing the new memory caps and `--set-phi` dataset requirement.

**Architecture:** Reuse the grid_lines Stage A harness with deterministic dataset caches, keep `--set-phi` enabled for every run so phase metrics hold meaning, and cap Hybrid/StableHybrid models with `--torch-max-hidden-channels 512` to stay within 24 GB GPUs. Crash Hunt sweeps depths [4, 6, 8] for the control arm to locate the lowest depth with ≥1 failure out of 3 seeds. The Shootout then runs three arms (Control Hybrid, LayerScale Hybrid, Optimizer/AGC candidate) at that crash depth with 3 seeds each, logging seed-level stability outcomes.

**Tech Stack:** Python 3.11, PyTorch Lightning, rsync, pytest, Stage A CLI harness, jq/python for metrics aggregation.

---

### Task 1: Crash Hunt Depth Sweep (Control Hybrid)

**Files / Paths:**
- Outputs: `outputs/grid_lines_crash_hunt/depth{4,6,8}_seed{A,B,C}`
- Artifacts hub: `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/`

**Step 1:** Prep datasets per depth.
```bash
for depth in 4 6 8; do
  for split in seedA seedB seedC; do
    mkdir -p outputs/grid_lines_crash_hunt/depth${depth}_${split}
    rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ \
      outputs/grid_lines_crash_hunt/depth${depth}_${split}/datasets/
  done
done
```

**Step 2:** Run control arm (hybrid, norm clip 1.0) for each depth × seed triple. Example command (seed placeholder expands to 20260128/29/30):
```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --set-phi \
  --output-dir outputs/grid_lines_crash_hunt/depth${DEPTH}_seed${SEED} \
  --architectures hybrid \
  --seed ${SEED} \
  --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
  --nepochs 30 --torch-epochs 30 \
  --fno-blocks ${DEPTH} \
  --torch-max-hidden-channels 512 \
  --torch-grad-clip 1.0 --torch-grad-clip-algorithm norm \
  --torch-loss-mode mae --torch-infer-batch-size 8 \
  --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/depth${DEPTH}_seed${SEED}.log
```
- Depth 8 runs may still OOM; note blockers in README if so.

**Step 3:** Capture stats per run via `scripts/internal/stage_a_dump_stats.py --run-dir ... --out-json .../depth${DEPTH}_seed${SEED}_stats.json` and tabulate P_crash (count collapsed runs / total). Save summary to `crash_hunt_summary.json`.

### Task 2: Three-Arm Shootout at Crash Depth

**Prereq:** From Task 1, identify the smallest depth with ≥1 crash. Call it `DEPTH_CRASH` (likely 6).

**Arms:**
1. Control Hybrid (norm clip 1.0)
2. LayerScale Hybrid (`stable_hybrid`, no clip)
3. Optimizer candidate (e.g., Hybrid + AdamW or AGC) — reference latest Phase 8 winner

**Step 1:** For each arm, create directories `outputs/grid_lines_shootout/<arm>/seed{A,B,C}` and rsync Stage A datasets per seed. Ensure `--torch-max-hidden-channels 512` if `DEPTH_CRASH ≥ 6`.

**Step 2:** Example LayerScale command:
```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 64 --gridsize 1 \
  --set-phi \
  --output-dir outputs/grid_lines_shootout/stable_layerscale/seed${SEED} \
  --architectures stable_hybrid \
  --seed ${SEED} \
  --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
  --nepochs 30 --torch-epochs 30 \
  --fno-blocks ${DEPTH_CRASH} \
  --torch-max-hidden-channels 512 \
  --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
  --torch-scheduler WarmupCosine --torch-lr-warmup-epochs 5 --torch-lr-min-ratio 0.05 \
  --torch-learning-rate 5e-4 \
  --torch-infer-batch-size 8 \
  2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-02-01T000000Z/shootout_stable_seed${SEED}.log
```
- Control and optimizer arms follow the same template with their respective clipping/optimizer flags.

**Step 3:** After each seed finishes, capture stats JSON + `model.pt` path. Compute P_crash per arm (number of NaNs / constant-amp collapses out of 3). Store aggregated table in `shootout_results.json` with columns: arm, seed, best_val_loss, amp_ssim, crashed (bool).

### Task 3: Aggregation, Findings, and Plan Sync

1. Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` Phase 9 status with Crash Hunt + Shootout outcomes, referencing this plan.
2. Extend `docs/strategy/mainstrategy.md §3` with actual P_crash numbers and next hypotheses.
3. Add/refresh findings (e.g., new `STABLE-CRASH-DEPTH-001` capturing the crash depth evidence).
4. Update `docs/fix_plan.md` attempts history + FSM entry with the new artifacts hub path.
5. Run regression selectors after code changes (if any) and archive logs with the crash hunt evidence.

