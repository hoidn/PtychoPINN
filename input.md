# FNO-STABILITY-OVERHAUL-001 — Phase 1: Finish config + CLI plumbing

**Summary:** Close the remaining gaps in Phase 1 so every runner can actually request AGC (`docs/strategy/mainstrategy.md` §2B).

**Focus:** FNO-STABILITY-OVERHAUL-001 — Phase 1 (Foundation)

**Branch:** fno2

**Mapped tests:**
- `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_training_config_gradient_clip_algorithm_roundtrip -v`
- `pytest tests/torch/test_grid_lines_torch_runner.py -k gradient_clip_algorithm -v`
- `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_grad_clip_algorithm -v`

**Artifacts:** `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-28T020500Z/`

---

## Do Now

Implement the remaining Phase 1 work from `plans/active/FNO-STABILITY-OVERHAUL-001/implementation.md` (Tasks 1.1 & 1.4). Keep Task IDs in commit messages.

### Task 1 — Finish `gradient_clip_algorithm` config parity
1. **TF dataclass:** In `ptycho/config/config.py:232` add `gradient_clip_algorithm: Literal['norm','value','agc'] = 'norm'` immediately after `gradient_clip_val` inside `PyTorchExecutionConfig`. (Torch `TrainingConfig` already has the field; we need parity on the canonical config used by the runner bridge per `docs/workflows/pytorch.md`.)
2. **Bridge plumbing:** Update `ptycho_torch/config_bridge.py::to_training_config` so it copies `training.gradient_clip_algorithm` into the TensorFlow `TrainingConfig`. Add a regression test in `tests/torch/test_config_bridge.py::TestConfigBridgeParity` that constructs a PyTorch TrainingConfig with algorithm `'agc'`, runs through the bridge, calls `update_legacy_dict`, and asserts both the TF dataclass and `params.cfg['gradient_clip_algorithm']` match.
3. **Docs/tests:** If the new test changes collected selectors, update `docs/development/TEST_SUITE_INDEX.md` entry for `tests/torch/test_config_bridge.py` per `docs/TESTING_GUIDE.md` §2 only if names change (not required if just adding a case).

### Task 2 — Expose AGC selection end-to-end on the compare harness
1. **CLI flag:** In `scripts/studies/grid_lines_compare_wrapper.py`, add `--torch-grad-clip-algorithm` (choices `['norm','value','agc']`, default `'norm'`) to `parse_args`. Thread this argument through `run_grid_lines_compare()` to the `TorchRunnerConfig` constructor (`gradient_clip_algorithm=...`). Follow the precedent from `--torch-grad-clip` introduced in `docs/plans/2026-01-27-wire-torch-grad-clip-val.md`.
2. **Wrapper tests:** Extend `tests/test_grid_lines_compare_wrapper.py` with `test_wrapper_passes_grad_clip_algorithm` that monkeypatches the torch runner, runs the wrapper with `--torch-grad-clip-algorithm agc`, and asserts the captured config field equals `'agc'`. While there, add a parse_args assertion for the default.
3. **Runner regression:** `tests/torch/test_grid_lines_torch_runner.py` already builds configs via `setup_torch_configs`. Add a focused test (e.g., `test_gradient_clip_algorithm_forwarded`) ensuring the dataclass inherits the CLI value and that `TrainingConfig.gradient_clip_algorithm` feeds through to `PyTorchExecutionConfig` if needed. Keep coverage under the existing class to avoid new fixtures.

### Task 3 — Verification
Run the mapped selectors from above (capture logs under the artifacts path):
1. `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_training_config_gradient_clip_algorithm_roundtrip -v`
2. `pytest tests/torch/test_grid_lines_torch_runner.py -k gradient_clip_algorithm -v`
3. `pytest tests/test_grid_lines_compare_wrapper.py::test_wrapper_passes_grad_clip_algorithm -v`

Archive each command’s stdout/stderr in `plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-28T020500Z/` following `docs/TESTING_GUIDE.md` log naming conventions.

---

## Next Up

Once config + CLI plumbing are green, proceed to Phase 2 (`StablePtychoBlock`, `StableHybridGenerator`, registry + tests) before launching the shootout in Phase 3.
