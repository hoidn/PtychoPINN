# Torch Runner Inference Fixes (Incremental Plan 3)

**Goal:** Close the remaining Torch runner gaps flagged in the updated plans: inference signature alignment, inference batching to avoid OOM, and config‑driven FNO hyperparameters.

**Source Plans (updated):**
- `docs/plans/2026-01-27-modular-generator-implementation.md` — Task 6 (runner contract now requires correct inference signature + batching + FNO hyperparams).
- `docs/plans/2026-01-27-fno-hybrid-correctness-fixes.md` — Task 4b (inference signature + batching) and Task 4c (hyperparameter exposure).

---

## Task A: Align inference signature (Lightning forward_predict)

**Why:** Current runner calls `model(X)` for FNO/Hybrid; Lightning expects `forward_predict(x, positions, probe, input_scale_factor)` and will crash.

**Implementation target:** `scripts/studies/grid_lines_torch_runner.py`

**Steps:**
1. Add `infer_batch_size` to `TorchRunnerConfig`.
2. Build `positions`, `probe`, and `input_scale_factor` tensors from test data (or safe defaults).
3. Call `model.forward_predict(...)` in a batched loop (see Task B).
4. Update tests in `tests/torch/test_grid_lines_torch_runner.py` to assert the signature is used.

**Exit criteria:** A unit test proves `forward_predict` is invoked with all required args.

---

## Task B: Add inference batching (OOM guard)

**Why:** Dense runs (~5k images) will OOM if the full tensor is pushed to GPU in one call.

**Implementation target:** `scripts/studies/grid_lines_torch_runner.py`

**Steps:**
1. Iterate inference in batches (slice X + coords), respecting `infer_batch_size`.
2. Aggregate predictions on CPU to avoid holding full GPU outputs.
3. Reuse the same batching for both raw patch outputs and any stitched path (if used).

**Exit criteria:** Inference path never loads the full test tensor onto GPU at once.

---

## Task C: Expose FNO hyperparameters via config

**Why:** Hardcoded defaults make the architecture untunable and can cause OOM or under‑capacity.

**Implementation targets:**
- `ptycho/config/config.py` (TF ModelConfig)
- `ptycho_torch/config_params.py` (PT ModelConfig)
- `ptycho_torch/config_factory.py` (override plumbing)
- `ptycho_torch/model.py` (generator instantiation)
- `scripts/studies/grid_lines_torch_runner.py` (CLI overrides)

**Steps:**
1. Add `fno_modes`, `fno_width`, `fno_blocks`, `fno_cnn_blocks` to configs with safe defaults.
2. Thread overrides through config factory and runner CLI.
3. Use config values when instantiating FNO/Hybrid generators.
4. Add a unit test asserting non‑default hyperparams propagate into generator construction.

**Exit criteria:** Hyperparameters are adjustable without code edits and reflected in generator instantiation.

---

## Verification

Run targeted tests:
```bash
pytest tests/torch/test_grid_lines_torch_runner.py -v
pytest tests/torch/test_fno_integration.py -v
```

Capture logs under `.artifacts/torch_runner_inference_fixes/` and link them in this plan.

**Evidence (2026-01-27):**
- `.artifacts/torch_runner_inference_fixes/pytest_grid_lines_torch_runner.log`
- `.artifacts/torch_runner_inference_fixes/pytest_fno_integration.log`
