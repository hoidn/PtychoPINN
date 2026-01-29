# FNO-STABILITY-OVERHAUL-001 — Phase 8 Optimizer Sweep + Activation Diagnostics

Plan: `plans/active/FNO-STABILITY-OVERHAUL-001/plan_optimizer_diagnostics.md`

References:
- `docs/strategy/mainstrategy.md`
- `docs/plans/2026-01-30-stable-hybrid-optimizer-diagnostics.md`
- `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md`
- `docs/findings.md`
- `docs/TESTING_GUIDE.md`

**Summary:** Implement Phase 8 Tasks 1–3 from `plan_optimizer_diagnostics.md`: add optimizer selection plumbing (Adam/AdamW/SGD) with tests across config_bridge, runner, and Lightning; enhance `scripts/debug_fno_activations.py` so it loads `stable_hybrid` checkpoints; then run the two Stage A experiments (SGD + AdamW) with activation captures and archive everything under the new reports hub.
**Summary (goal):** Ship optimizer plumbing + activation tooling, then determine whether SGD or AdamW prevents the STABLE-LS-001 collapse. All Stage A invocations must include `--set-phi` per docs/strategy/mainstrategy.md.

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 8 (optimizer + activation diagnostics)

**Branch:** fno2-phase8-optimizers

**Mapped tests:**
- `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_training_config_optimizer_roundtrip -vv`
- `pytest tests/torch/test_grid_lines_torch_runner.py::TestChannelGridsizeAlignment::test_runner_accepts_optimizer -vv`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_optimizer -vv`
- `pytest tests/torch/test_model_training.py::TestOptimizerSelection::test_configures_sgd -vv`
- `pytest tests/torch/test_debug_fno_activations.py::test_debug_fno_activations_emits_report -vv`
- `pytest tests/torch/test_fno_generators.py::TestStablePtychoBlock::test_layerscale_grad_flow -vv`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z/`

**Next Up (optional):** If both optimizers still collapse, pull gradient histograms at the collapse epoch (`scripts/internal/stage_a_dump_stats.py --plot-grad-hist`) to prep for Phase 9 architecture pivots.

---

## Do Now — plan_optimizer_diagnostics.md Tasks 1–3

1. **Re-read references:** `docs/strategy/mainstrategy.md` (Phase 7 outcome + Phase 8 intent), `docs/findings.md` (STABLE-LS-001), `docs/plans/2026-01-30-stable-hybrid-optimizer-diagnostics.md` + `plans/active/FNO-STABILITY-OVERHAUL-001/plan_optimizer_diagnostics.md` (Task details), and `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Phase 8 section). Keep `docs/TESTING_GUIDE.md` handy for artifacting rules.
2. **Task 1 — Optimizer plumbing & tests:**
   - Extend TF + Torch `TrainingConfig` dataclasses with `optimizer` literal + `momentum`, `weight_decay`, `adam_beta1`, `adam_beta2`; mirror fields through `config_bridge`, `config_factory`, and `TorchRunnerConfig`.
   - Add CLI flags to `scripts/studies/grid_lines_torch_runner.py` (`--torch-optimizer`, `--torch-weight-decay`, `--torch-momentum`, `--torch-beta1`, `--torch-beta2`) and forward them via `grid_lines_compare_wrapper.py`.
   - Update `PtychoPINN_Lightning.configure_optimizers()` to branch on the new field (Adam/AdamW/SGD) and keep scheduler wiring intact.
   - Tests: update/add the selectors listed above (config bridge roundtrip, runner CLI, wrapper forwarding, Lightning optimizer smoke test). Archive each pytest log under the artifacts directory.
3. **Task 2 — Activation debug upgrades:**
   - Enhance `scripts/debug_fno_activations.py` to accept `--architecture stable_hybrid`, `--checkpoint <model.pt>`, and optional `--output-json-name`; load checkpoints by stripping the `model.` prefix before calling `load_state_dict`.
   - Expand `tests/torch/test_debug_fno_activations.py` so it saves a temporary `StableHybridUNOGenerator` state_dict, runs the script with the new flags, and asserts the custom JSON filename exists.
   - Run `pytest tests/torch/test_debug_fno_activations.py -vv` and archive the log.
4. **Task 3 prep — workspace + datasets:** from `fno2-phase8-optimizers`, create Stage A arm directories:
   ```bash
   mkdir -p outputs/grid_lines_stage_a/{arm_stable_sgd,arm_stable_adamw}
   rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_stable_sgd/datasets/
   rsync -a outputs/grid_lines_stage_a/arm_control/datasets/ outputs/grid_lines_stage_a/arm_stable_adamw/datasets/
   rm -rf outputs/grid_lines_stage_a/arm_stable_sgd/runs outputs/grid_lines_stage_a/arm_stable_adamw/runs
   echo "Stage A shared params: seed=20260128, N=64, gridsize=1, nimgs=1, nphotons=1e9, epochs=20, ALWAYS pass --set-phi" > plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z/README.md
   ```
5. **Task 3 — SGD arm (WarmupCosine, LR 3e-4, no clip):**
   ```bash
   AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md \
   python scripts/studies/grid_lines_compare_wrapper.py \
     --N 64 --gridsize 1 \
     --set-phi \
     --output-dir outputs/grid_lines_stage_a/arm_stable_sgd \
     --architectures stable_hybrid \
     --seed 20260128 \
     --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 \
     --nepochs 20 --torch-epochs 20 \
     --torch-learning-rate 3e-4 \
     --torch-optimizer sgd --torch-momentum 0.9 --torch-weight-decay 0.0 \
     --torch-scheduler WarmupCosine --torch-lr-warmup-epochs 5 --torch-lr-min-ratio 0.05 \
     --torch-grad-clip 0.0 --torch-grad-clip-algorithm norm \
     --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8 \
     --torch-log-grad-norm --torch-grad-norm-log-freq 1 \
     2>&1 | tee plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z/stage_a_arm_stable_sgd.log
   ```
   - Copy `history.json`, `metrics.json`, `model.pt`, grad-norm traces into the hub, run `scripts/internal/stage_a_dump_stats.py --run-dir outputs/grid_lines_stage_a/arm_stable_sgd/runs/pinn_stable_hybrid --out-json plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z/stage_a_arm_stable_sgd_stats.json`.
   - Immediately capture activations:
   ```bash
   python scripts/debug_fno_activations.py \
     --input outputs/grid_lines_stage_a/arm_control/datasets/N64/gs1/train/train.npz \
     --output plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-30T050000Z \
     --architecture stable_hybrid \
     --checkpoint outputs/grid_lines_stage_a/arm_stable_sgd/runs/pinn_stable_hybrid/model.pt \
     --batch-size 1 --max-samples 1 --device cpu \
     --output-json-name activation_report_sgd.json
   ```
6. **Task 3 — AdamW arm (weight decay 0.01):** repeat Step 5 with `--output-dir .../arm_stable_adamw`, `--torch-optimizer adamw --torch-weight-decay 0.01 --torch-beta1 0.9 --torch-beta2 0.999`, and log to `stage_a_arm_stable_adamw.log`. Keep `--set-phi` in place to preserve phase metrics. Archive artifacts + stats JSON and run the activation capture with `activation_report_adamw.json`.
7. **Aggregate + docs:** append both runs to `stage_a_metrics_phase8.json` (new file under the hub) and write `stage_a_optimizer_summary.md` comparing SGD/AdamW vs Phase 7 metrics. Update `plans/active/.../implementation.md`, `plans/active/.../summary.md`, `docs/strategy/mainstrategy.md`, `docs/findings.md` (update STABLE-LS-001 or add STABLE-OPT-001), and `docs/fix_plan.md` with the outcomes.
8. **Regression selectors:** after code + runs are done, re-run the mapped regression selectors listed above (plus the three existing stable_hybrid selectors if time allows) and place the pytest logs inside the artifacts directory.
9. **Git hygiene:** keep large logs/models out of git (only references in hub). Stage only source + doc changes, leave `outputs/` untracked.
