# FNO-STABILITY-OVERHAUL-001 — Phase 6: Training Dynamics Plan

**Summary:** Implement the Warmup+Cosine scheduler plumbing (Task 6.1), add the scheduler helper (Task 6.2), then rerun the Stage A stable arm per the new plan to test whether the collapse is resolved.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 6 (Training dynamics / warmup-cosine scheduler)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_training_config_lr_scheduler_roundtrip -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestTorchRunnerConfig::test_setup_configs_threads_scheduler_fields -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_scheduler_knobs -v`
- `pytest tests/torch/test_lr_scheduler.py::test_warmup_cosine_scheduler_progression -v`
- `pytest tests/torch/test_model_training.py::test_configure_optimizers_selects_warmup_scheduler -v`
- `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_handles_stable_hybrid -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_stable_hybrid -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T235500Z/` (plan_training_dynamics.md, README)

**Next Up (optional):** If scheduler run succeeds quickly, start evaluating gradient-checkpointing requirements for fno_blocks>6.

---

## Do Now — Execute plan_training_dynamics.md Tasks

1. **Read Phase 6 plan:** `plans/active/FNO-STABILITY-OVERHAUL-001/plan_training_dynamics.md` (same content as `docs/plans/2026-01-29-stable-hybrid-training-dynamics.md`). Follow Tasks 6.1–6.3 in order; do not skip tests.
2. **Task 6.1 (scheduler plumbing):**
   - Extend TF/Torch `TrainingConfig` dataclasses + config bridge with `scheduler`, `lr_warmup_epochs`, and `lr_min_ratio`.
   - Update `TorchRunnerConfig`, `setup_torch_configs()`, and the torch runner CLI to accept `--scheduler`, `--lr-warmup-epochs`, `--lr-min-ratio`, and ensure `training_config.learning_rate` honors `--learning-rate`.
   - Propagate the new flags through `grid_lines_compare_wrapper.py`.
   - Add/adjust the three targeted regression tests listed above plus rerun the existing grid-lines CLI tests the plan calls out. Archive all pytest logs in the artifacts directory.
3. **Task 6.2 (WarmupCosine helper):**
   - Add `ptycho_torch/schedulers.py` with `build_warmup_cosine_scheduler()` (Linear warmup → Cosine anneal) and teach `PtychoPINN_Lightning.configure_optimizers()` to dispatch it when `scheduler == 'WarmupCosine'`.
   - Create the new scheduler unit test + Lightning smoke test selectors. Ensure they pass along with `tests/torch/test_fno_generators.py::TestStablePtychoBlock -v`.
4. **Task 6.3 (Stage A rerun + docs):**
   - Reuse the Stage A control dataset and run the stable arm with `--torch-scheduler WarmupCosine --torch-learning-rate 5e-4 --torch-lr-warmup-epochs 5 --torch-lr-min-ratio 0.05`. Tee logs to the artifacts path and copy `history.json`, `metrics.json`, `model.pt`, and stats JSONs there.
   - (Optional but helpful) Run a low constant LR baseline for comparison if time permits.
   - Update `stage_a_metrics.json`, `stage_a_summary.md`, `docs/strategy/mainstrategy.md`, `docs/fix_plan.md`, `docs/findings.md`, and `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` with the new results. Keep STABLE-LS-001 status accurate.
5. **Archiving & hygiene:** Stick to CLAUDE.md rules—no datasets committed, run all mapped selectors, and drop every new log/README/test output under `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T235500Z/`. EOF
