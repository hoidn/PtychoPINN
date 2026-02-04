# Grid Lines Torch Sim Backend Fixes Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Fix the Torch grid-lines sim backend so it validates CLI inputs, passes probe configuration into simulation, and supplies stitching metadata.

**Architecture:** Extend the Torch runner to conditionally validate CLI args and construct a `GridLinesConfig` with probe scaling/mask parameters when `--sim-backend torch` is used. Create metadata using `MetadataManager` so `_stitch_for_metrics` has `nimgs_test`/`outer_offset_test` in `additional_parameters`, matching the grid-lines NPZ contract.

**Tech Stack:** Python, pytest, PyTorch (`ptycho_torch`), TensorFlow-backed grid-lines simulator (`ptycho/workflows/grid_lines_workflow.py`).

### Task 1: CLI Validation for Sim Backend

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Run failing test**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::TestSimBackend::test_validate_cli_args_requires_npz_without_sim_backend -v`  
Expected: FAIL with `AttributeError` (`_validate_cli_args` missing).

**Step 2: Write minimal implementation**

Add `_validate_cli_args(args)` to enforce:
- If `sim_backend` is None: `train_npz` and `test_npz` required.
- If `sim_backend` is set: `probe_npz` required, `train_npz`/`test_npz` optional.

Adjust argparse so `--train-npz`/`--test-npz` are not required and call `_validate_cli_args(args)` in `main`.

**Step 3: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::TestSimBackend::test_validate_cli_args_requires_npz_without_sim_backend -v`  
Expected: PASS.

### Task 2: Sim Backend Metadata + Probe Config Plumbing

**Files:**
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Run failing test**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::TestSimBackend::test_sim_backend_provides_metadata_for_stitching -v`  
Expected: FAIL with `metadata is None`.

**Step 2: Write minimal implementation**

- Extend `TorchRunnerConfig` to include probe config overrides:
  - `probe_smoothing_sigma`, `probe_mask_diameter`, `probe_scale_mode`
- Add CLI flags for these (defaults match `GridLinesConfig`).
- In sim-backend path, build `GridLinesConfig` with probe parameters from `TorchRunnerConfig`.
- Create metadata via `MetadataManager.create_metadata(...)` using `setup_torch_configs(cfg)` and `GridLinesConfig` values, then pass to `_stitch_for_metrics`.

**Step 3: Run test to verify it passes**

Run: `pytest tests/torch/test_grid_lines_torch_runner.py::TestSimBackend::test_sim_backend_provides_metadata_for_stitching -v`  
Expected: PASS.

### Task 3: Targeted Regression Checks

**Files:**
- Test: `tests/torch/test_grid_lines_sim.py`
- Test: `tests/torch/test_grid_lines_torch_runner.py`

**Step 1: Run key selectors**

Run:
```
pytest tests/torch/test_grid_lines_sim.py::TestTorchGridLinesSim::test_torch_sim_includes_probe_guess -v
pytest tests/torch/test_grid_lines_sim.py::TestTorchGridLinesSim::test_torch_sim_applies_probe_mask -v
pytest tests/torch/test_grid_lines_torch_runner.py::TestSimBackend::test_validate_cli_args_requires_probe_npz_with_sim_backend -v
pytest tests/torch/test_grid_lines_torch_runner.py::TestSimBackend::test_validate_cli_args_allows_missing_npz_with_sim_backend -v
```
Expected: PASS.

### Task 4: Repo Testing Requirements

**Files:**
- Modify: `docs/TESTING_GUIDE.md` (only if selectors/coverage updated)

**Step 1: Run required integration gate**

Run: `pytest -v -m integration`  
Expected: PASS. Save log under the active plan hub per `docs/TESTING_GUIDE.md`.

**Step 2: Commit**

```
git add scripts/studies/grid_lines_torch_runner.py tests/torch/test_grid_lines_torch_runner.py docs/plans/2026-02-01-grid-lines-torch-sim-backend-fixes.md
git commit -m "fix grid-lines: harden sim backend CLI+metadata (tests: sim backend selectors)"
```
