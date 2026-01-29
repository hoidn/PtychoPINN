# FNO-STABILITY-OVERHAUL-001 — Phase 6 Task 6.3 (WarmupCosine Stage A rerun)

**Summary:** Execute the stable_hybrid Stage A arm with the new Warmup+Cosine scheduler and propagate the results through metrics, docs, and findings.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 6 (Training dynamics / WarmupCosine scheduler)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock::test_layerscale_grad_flow -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T235959Z/`

**Next Up (optional):** If WarmupCosine succeeds quickly, start outlining gradient-checkpointing requirements for `fno_blocks>6` depth tests.

---

## Do Now — Phase 6 plan_training_dynamics.md Task 6.3

1. **Re-read Phase 6 instructions:** `plans/active/FNO-STABILITY-OVERHAUL-001/plan_training_dynamics.md` (§Task 6.3) for the exact CLI, artifact expectations, and doc sync checklist. Keep STABLE-LS-001 (docs/findings.md) in mind when evaluating outcomes.
2. **Prep datasets + run WarmupCosine arm:**
   - Mirror the Stage A control datasets into `outputs/grid_lines_stage_a/arm_stable_warmup` (rsync command is spelled out in the plan) and clear any stale `runs/` directories.
   - Run the stable_hybrid arm with the WarmupCosine knobs via `python scripts/studies/grid_lines_compare_wrapper.py ...`: `--output-dir outputs/grid_lines_stage_a/arm_stable_warmup --architectures stable_hybrid --torch-scheduler WarmupCosine --torch-learning-rate 5e-4 --torch-lr-warmup-epochs 5 --torch-lr-min-ratio 0.05 --torch-grad-clip 0.0 --fno-blocks 4 --torch-epochs 20 --torch-loss-mode mae --torch-infer-batch-size 8 --seed 20260128`, reusing the cached dataset and `--nimgs-train 1 --nimgs-test 1 --nphotons 1e9 --gridsize 1 --N 64 --set-phi`. Tee the CLI output to `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T235959Z/stage_a_arm_stable_warmup.log`.
   - Optional comparison (time permitting): rerun with `--torch-scheduler Default --torch-learning-rate 2.5e-4` into `arm_stable_lowlr` to isolate whether improvements come from the schedule vs. LR magnitude.
3. **Archive metrics + stats:**
   - Copy each run’s `history.json`, `metrics.json`, checkpoints, and grad norm traces into the artifacts directory.
   - Run `python scripts/internal/stage_a_dump_stats.py --run-dir .../arm_stable_warmup/runs/pinn_stable_hybrid --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T235959Z/stage_a_arm_stable_warmup_stats.json` (repeat for the optional low-LR run if executed).
   - Append the new rows to `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T010000Z/stage_a_metrics.json` (or create a Phase 6 metrics file if structure requires) and summarize findings in `stage_a_summary.md`, highlighting whether WarmupCosine prevents the collapse noted in STABLE-LS-001.
4. **Docs + findings sync:**
   - Update `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Phase 6 status), `plans/active/FNO-STABILITY-OVERHAUL-001/summary.md`, `docs/strategy/mainstrategy.md` (Stage A table), `docs/fix_plan.md`, and `docs/findings.md` with the new evidence. Close or update STABLE-LS-001; add any new findings if behavior changes.
   - If WarmupCosine fails to stabilize, document the observed failure curve (epoch/time of collapse) and hypothesis for next levers.
5. **Regression tests + hygiene:**
   - Run the mapped selectors listed above plus any additional quick checks you need for touched modules, archiving `pytest` logs under the artifacts directory.
   - Keep large artifacts out of git; ensure all logs/metrics from this task live under `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T235959Z/`. EOF
