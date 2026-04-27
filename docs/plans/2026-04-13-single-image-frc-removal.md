# Single-Image FRC Removal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [x]`) syntax for tracking. Project override: do not create a git worktree.

**Goal:** Remove the no-reference single-image FRC feature and its import/runtime dependency while preserving ground-truth FRC metrics.

**Architecture:** Treat no-reference single-image FRC as removed, not optional. `ptycho.evaluation` must import and evaluate without `ptycho.single_image_frc`, and it should no longer expose `single_image_frc_metrics` or accept `single_image_frc*` keyword arguments. Grid-lines workflows, wrappers, tests, metrics tables, and docs should stop requesting or advertising `single_frc50` / `single_frc1over7`.

**Tech Stack:** Python, pytest, argparse, NumPy/TensorFlow metric tests, existing `ptycho.evaluation`, grid-lines TensorFlow workflow, grid-lines Torch runner, grid-lines comparison wrapper, metrics table helpers.

---

## Initiative

- ID: `single-image-frc-removal`
- Title: Remove no-reference single-image FRC
- Status: implemented
- Owner: optional
- Spec/Source: user request on 2026-04-13; current import failure around `ptycho.single_image_frc`

## Compliance Matrix

- [x] **Project instruction:** Read `docs/index.md` and `docs/findings.md` before implementation; preserve project docs/spec authority order.
- [x] **Finding/Policy ID:** `ANTIPATTERN-001` - avoid import-time side effects and hard import dependencies.
- [x] **Finding/Policy ID:** `CONFIG-001` - do not alter legacy `params.cfg` initialization paths while changing evaluation signatures.
- [x] **Finding/Policy ID:** `DATA-001` - metric output shape/key changes must be deliberate and documented for downstream table consumers.
- [x] **Paper-revision process:** If this blocks or unblocks revision-study work, update `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`.

## Spec Alignment

- **Normative Spec:** `specs/data_contracts.md` for data-shape preservation; `docs/DEVELOPER_GUIDE.md` for import-side-effect policy.
- **Key Clauses:** Keep ground-truth metrics (`mae`, `mse`, `psnr`, `ssim`, `ms_ssim`, `frc50`, `frc1over7`, `frc`) unchanged. Remove no-reference `single_frc50` and `single_frc1over7` outputs from normal evaluation, metrics tables, and reviewer-study workflows.

## Context Priming

Read before editing:

- `docs/index.md`
- `docs/findings.md`
- `docs/DEVELOPER_GUIDE.md`
- `ptycho/evaluation.py`
- `frc/__init__.py`
- `frc/single_image_frc.py`
- `tests/test_evaluation_single_image_frc.py`
- `ptycho/workflows/grid_lines_workflow.py`
- `scripts/studies/grid_lines_torch_runner.py`
- `scripts/studies/grid_lines_compare_wrapper.py`
- `scripts/studies/metrics_tables.py`
- `scripts/studies/analyze_single_image_frc_alignment.py`
- `tests/test_grid_lines_workflow.py`
- `tests/torch/test_grid_lines_torch_runner.py`
- `tests/test_grid_lines_compare_wrapper.py`
- `tests/studies/test_metrics_tables.py`
- If relevant to revision-study status: `/home/ollie/Documents/ptychopinnpaper2/reviewer_revision_checklist.md`

Current state to account for:

- `ptycho/evaluation.py` imports `ptycho.single_image_frc` at module import time.
- `ptycho/single_image_frc.py` exists locally but is untracked in the current checkout.
- `frc/single_image_frc.py` is tracked, but currently only shims to `ptycho.single_image_frc`.
- `frc/__init__.py` exports single-image FRC helper names from that shim.
- Several grid-lines paths default `single_image_frc` to enabled.
- `tests/test_evaluation_single_image_frc.py` mostly tests behavior that will be removed.
- `tests/test_evaluation_single_image_frc.py::test_evaluation_import_does_not_require_top_level_frc_package` currently fails for a subprocess harness reason: the temp script path, not the repo root, becomes `sys.path[0]`.

## Architecture / Interfaces

- `ptycho.evaluation` remains the public metric API for reference-based evaluation.
- Ground-truth FRC (`frc50`, `frc1over7`, `frc`) remains available and unaffected.
- No-reference single-image FRC (`single_frc50`, `single_frc1over7`) is removed from tracked APIs.
- `eval_reconstruction(...)` should no longer accept `single_image_frc`, `single_image_frc_split_mode`, `single_image_frc_rng_seed`, or related calibration/binomial kwargs.
- `ptycho.evaluation.single_image_frc_metrics` should be removed.
- `frc.single_image_frc` and the exports in `frc.__init__` should be removed or neutralized so `import frc` does not depend on an absent module.
- Grid-lines TF/Torch workflow code should stop passing single-image FRC kwargs.
- CLI flags such as `--single-image-frc`, `--no-single-image-frc`, `--torch-single-image-frc`, and `--torch-no-single-image-frc` should be removed rather than retained as no-ops.
- Metrics table helpers should not display or format `single_frc50` / `single_frc1over7` columns by default.

## Files and Responsibilities

- `ptycho/evaluation.py`: remove the hard import, wrapper, removed kwargs, and no-reference FRC branch; preserve reference-based metrics.
- `frc/__init__.py`: stop exporting single-image FRC helper names.
- `frc/single_image_frc.py`: delete tracked compatibility shim.
- `ptycho/single_image_frc.py`: leave untracked file uncommitted; delete only after confirming the user wants local cleanup.
- `ptycho/workflows/grid_lines_workflow.py`: remove hard-coded `single_image_frc=True` calls.
- `scripts/studies/grid_lines_torch_runner.py`: remove config fields, `compute_metrics` kwargs, argparse flags, and pass-through wiring.
- `scripts/studies/grid_lines_compare_wrapper.py`: remove single-image FRC params, imports, binomial post-processing, and argparse flags.
- `scripts/studies/metrics_tables.py`: remove default single-image FRC metric columns and formatting special cases.
- `scripts/studies/analyze_single_image_frc_alignment.py`: remove the obsolete analysis script or quarantine it under an explicitly ignored artifact path; do not leave it as an advertised runnable study script.
- `tests/test_evaluation_single_image_frc.py`: replace numeric feature tests with removal/import-regression tests, or delete the test module if equivalent coverage is moved elsewhere.
- Grid-lines and metrics-table test files: update expectations to the removed API.
- Docs under `docs/` and `scripts/`: remove or rewrite mentions that single-image FRC is available for normal workflows.

## Tasks

### Task 1: Lock the Removal Contract With Failing Tests

**Files:**
- Modify: `tests/test_evaluation_single_image_frc.py`
- Modify: `tests/test_grid_lines_workflow.py`
- Modify: `tests/torch/test_grid_lines_torch_runner.py`
- Modify: `tests/test_grid_lines_compare_wrapper.py`
- Modify: `tests/studies/test_metrics_tables.py`

- [x] **Step 1: Add an import-boundary regression test for `ptycho.evaluation`.**

  Add a subprocess test that blocks imports of `ptycho.single_image_frc` and then imports `ptycho.evaluation`. Use `python -c` from `cwd=repo_root` or set `PYTHONPATH=repo_root` explicitly so the test checks the removed dependency rather than temp-script package discovery.

  ```python
  def test_evaluation_import_does_not_require_single_image_frc_module():
      repo_root = Path(__file__).resolve().parents[1]
      code = r'''
  import builtins
  real_import = builtins.__import__
  def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
      if name == "ptycho.single_image_frc":
          raise ModuleNotFoundError(name)
      return real_import(name, globals, locals, fromlist, level)
  builtins.__import__ = guarded_import
  import ptycho.evaluation
  print("evaluation import ok")
  '''
      proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root, text=True, capture_output=True)
      assert proc.returncode == 0, proc.stderr
      assert "evaluation import ok" in proc.stdout
  ```

  Run:

  ```bash
  pytest tests/test_evaluation_single_image_frc.py::test_evaluation_import_does_not_require_single_image_frc_module -v
  ```

  Expected before implementation: FAIL because `ptycho.evaluation` imports `ptycho.single_image_frc` at module import time.

- [x] **Step 2: Add a reference-metrics preservation test.**

  Verify `eval_reconstruction(pred, gt, label="pinn")` returns legacy metric keys without `single_frc50` / `single_frc1over7`.

  Required assertions:

  ```python
  assert {"mae", "mse", "psnr", "ssim", "ms_ssim", "frc50", "frc1over7", "frc"}.issubset(out)
  assert "single_frc50" not in out
  assert "single_frc1over7" not in out
  ```

  Expected before implementation: may PASS for omitted kwargs, but should be kept as a regression test.

- [x] **Step 3: Add removed-API tests.**

  Assert the removed feature surface is gone:

  ```python
  import ptycho.evaluation as evaluation

  assert not hasattr(evaluation, "single_image_frc_metrics")
  with pytest.raises(TypeError, match="single_image_frc"):
      evaluation.eval_reconstruction(pred, gt, label="pinn", single_image_frc=True)
  ```

  Expected before implementation: FAIL because the wrapper and kwargs still exist.

- [x] **Step 4: Update grid-lines tests to expect no single-image FRC pass-through.**

  Update or add tests showing:

  - `ptycho.workflows.grid_lines_workflow` calls `eval_reconstruction(...)` without a `single_image_frc` kwarg.
  - `scripts/studies/grid_lines_torch_runner.TorchRunnerConfig()` has no `single_image_frc` attribute.
  - `scripts/studies/grid_lines_torch_runner.parse_args([...])` has no `single_image_frc` attribute.
  - `scripts/studies/grid_lines_compare_wrapper.parse_args([...])` has no `torch_single_image_frc` attribute.

  Run targeted tests:

  ```bash
  pytest tests/test_grid_lines_workflow.py -k single_image_frc -v
  pytest tests/torch/test_grid_lines_torch_runner.py -k single_image_frc -v
  pytest tests/test_grid_lines_compare_wrapper.py -k single_image_frc -v
  ```

  Expected before implementation: FAIL where single-image FRC fields and flags still exist.

- [x] **Step 5: Add metrics table tests for removed columns.**

  Verify the default table metric list excludes:

  - `single_frc50_binomial`
  - `single_frc1over7_binomial`

  Run:

  ```bash
  pytest tests/studies/test_metrics_tables.py -v
  ```

  Expected before implementation: FAIL if tests assert the old default headers.

### Task 2: Remove Single-Image FRC From `ptycho.evaluation`

**Files:**
- Modify: `ptycho/evaluation.py`

- [x] **Step 1: Remove the top-level import.**

  Delete:

  ```python
  from ptycho.single_image_frc import single_image_frc_metrics as _single_image_frc_metrics_impl
  ```

- [x] **Step 2: Remove the compatibility wrapper.**

  Delete the `single_image_frc_metrics(...)` function in `ptycho/evaluation.py`.

- [x] **Step 3: Remove kwargs from `eval_reconstruction(...)`.**

  Remove these parameters from the function signature:

  - `single_image_frc`
  - `single_image_frc_split_mode`
  - `single_image_frc_rng_seed`
  - `single_image_frc_spatial_antialias_sigma`
  - `single_image_frc_spatial_calibration_json`
  - `single_image_frc_spatial_calibration_profile`
  - `single_image_frc_binomial_mean_lambda`
  - `single_image_frc_binomial_normalize_intensity`
  - `single_image_frc_binomial_count_scale`

- [x] **Step 4: Remove the no-reference FRC branch.**

  Delete the `if single_image_frc:` block that calls `single_image_frc_metrics(...)` and merges `single_frc50` / `single_frc1over7` into the output.

- [x] **Step 5: Run evaluation tests.**

  ```bash
  pytest tests/test_evaluation_single_image_frc.py -v
  ```

  Expected: import/removal/reference-metrics tests pass without requiring `ptycho/single_image_frc.py`.

### Task 3: Remove Tracked `frc` Shim Exports

**Files:**
- Modify: `frc/__init__.py`
- Delete: `frc/single_image_frc.py`
- Do not add: `ptycho/single_image_frc.py`

- [x] **Step 1: Remove single-image FRC exports from `frc/__init__.py`.**

  Delete imports and `__all__` entries for:

  - `center_crop_even_square`
  - `fit_and_remove_plane`
  - `first_below_threshold`
  - `single_image_frc_curve`
  - `single_image_frc_metrics`
  - `split_binomial_thinned`
  - `split_diagonal_interleaved`
  - `split_diagonal_strided_anti`
  - `split_diagonal_strided_main`
  - `trim_image`

  If `frc` has no remaining public exports, leave a minimal docstring and empty `__all__ = []`.

- [x] **Step 2: Delete the tracked shim.**

  Remove `frc/single_image_frc.py` so tracked code no longer imports the untracked canonical file.

- [x] **Step 3: Confirm the local canonical implementation remains untracked.**

  ```bash
  git status --short --untracked-files=all ptycho/single_image_frc.py
  git ls-files --error-unmatch ptycho/single_image_frc.py
  ```

  Expected: `ptycho/single_image_frc.py` is untracked and `git ls-files` reports it is not known to git.

- [x] **Step 4: Decide local cleanup.**

  Because `ptycho/single_image_frc.py` is untracked, do not delete it during implementation unless this hard-removal approval is confirmed to include local untracked-file cleanup. In either case, do not commit it.

### Task 4: Remove Grid-Lines Single-Image FRC Controls

**Files:**
- Modify: `ptycho/workflows/grid_lines_workflow.py`
- Modify: `scripts/studies/grid_lines_torch_runner.py`
- Modify: `scripts/studies/grid_lines_compare_wrapper.py`
- Modify: related tests listed in Task 1

- [x] **Step 1: Remove TensorFlow grid-lines kwargs.**

  Change calls in `ptycho/workflows/grid_lines_workflow.py` from:

  ```python
  eval_reconstruction(..., single_image_frc=True, ...)
  ```

  to calls without any `single_image_frc` kwarg.

- [x] **Step 2: Remove Torch runner config and CLI controls.**

  In `scripts/studies/grid_lines_torch_runner.py`:

  - Delete `TorchRunnerConfig.single_image_frc`.
  - Delete `TorchRunnerConfig.single_image_frc_split_mode`.
  - Delete `TorchRunnerConfig.single_image_frc_rng_seed`.
  - Delete `compute_metrics(..., single_image_frc=..., single_image_frc_split_mode=..., single_image_frc_rng_seed=...)` parameters.
  - Delete argparse flags for `--single-image-frc`, `--no-single-image-frc`, `--single-image-frc-split-mode`, and `--single-image-frc-rng-seed`.
  - Remove pass-through wiring into `eval_reconstruction(...)`.

- [x] **Step 3: Remove comparison wrapper controls.**

  In `scripts/studies/grid_lines_compare_wrapper.py`:

  - Delete `evaluate_selected_models(..., single_image_frc=..., single_image_frc_split_mode=..., single_image_frc_rng_seed=...)` parameters.
  - Stop importing `single_image_frc_metrics`.
  - Delete the binomial single-image FRC post-processing block.
  - Delete `torch_single_image_frc`, `torch_single_image_frc_split_mode`, and `torch_single_image_frc_rng_seed` wrapper parameters.
  - Delete argparse flags for `--torch-single-image-frc`, `--torch-no-single-image-frc`, `--torch-single-image-frc-split-mode`, and `--torch-single-image-frc-rng-seed`.
  - Remove pass-through wiring into `run_torch_comparison(...)` and related config construction.

- [x] **Step 4: Run workflow/wrapper tests.**

  ```bash
  pytest tests/test_grid_lines_workflow.py -k single_image_frc -v
  pytest tests/torch/test_grid_lines_torch_runner.py -k single_image_frc -v
  pytest tests/test_grid_lines_compare_wrapper.py -k single_image_frc -v
  ```

  Expected: tests either pass with updated removal assertions or no longer match stale `single_image_frc` selectors after the old tests are renamed/removed.

### Task 5: Remove Metrics-Table Columns and Obsolete Analysis Script

**Files:**
- Modify: `scripts/studies/metrics_tables.py`
- Modify: `tests/studies/test_metrics_tables.py`
- Delete or quarantine: `scripts/studies/analyze_single_image_frc_alignment.py`
- Modify docs that mention single-image FRC defaults or outputs

- [x] **Step 1: Remove single-image FRC columns from table defaults.**

  Delete these default metrics from `scripts/studies/metrics_tables.py`:

  - `single_frc50_binomial`
  - `single_frc1over7_binomial`

  Also delete formatting special cases that exist only for those keys.

- [x] **Step 2: Update metrics-table tests.**

  Replace tests that assert default headers contain binomial single-image FRC columns with tests asserting the default table omits them.

- [x] **Step 3: Remove or quarantine obsolete single-image FRC analysis script.**

  Preferred path: delete `scripts/studies/analyze_single_image_frc_alignment.py` if no active plan depends on it.

  If historical reproducibility requires keeping it, move the content to a non-runnable historical note outside normal script discovery and update docs to say the removed metric is no longer part of supported workflows. Do not leave a normal runnable script that imports removed APIs.

- [x] **Step 4: Update documentation.**

  Search:

  ```bash
  rg -n "single_image_frc|single_frc50|single_frc1over7|single-image FRC|1FRC" docs scripts ptycho tests
  ```

  Update docs to say the no-reference single-image FRC feature was removed, not disabled. Preserve references in this plan and in changelog-style historical notes only if they clearly say the feature is removed.

### Task 6: Final Verification

**Files:**
- Read: `git diff --stat`
- Read: all modified code and test files

- [x] **Step 1: Run targeted evaluation tests.**

  ```bash
  pytest tests/test_evaluation_single_image_frc.py -v
  ```

  Expected: tests pass without requiring `ptycho/single_image_frc.py`.

- [x] **Step 2: Run grid-lines and table tests.**

  ```bash
  pytest tests/test_grid_lines_workflow.py -k single_image_frc -v
  pytest tests/torch/test_grid_lines_torch_runner.py -k single_image_frc -v
  pytest tests/test_grid_lines_compare_wrapper.py -k single_image_frc -v
  pytest tests/studies/test_metrics_tables.py -v
  ```

  Expected: all updated tests pass. If `-k single_image_frc` matches zero tests because the old surface is fully removed and renamed, document that and run the relevant containing test files.

- [x] **Step 3: Run import preflight without optional module.**

  ```bash
  python - <<'PY'
  import builtins
  real_import = builtins.__import__
  def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
      if name == "ptycho.single_image_frc":
          raise ModuleNotFoundError(name)
      return real_import(name, globals, locals, fromlist, level)
  builtins.__import__ = guarded_import
  import ptycho.evaluation
  print("evaluation import ok without single-image FRC")
  PY
  ```

  Expected: prints `evaluation import ok without single-image FRC`.

- [x] **Step 4: Run reference scan.**

  ```bash
  rg -n "single_image_frc|single_frc50|single_frc1over7|single-image FRC|1FRC" ptycho frc scripts tests docs
  ```

  Expected: no production/test references remain except this plan, changelog-style historical notes, or explicit documentation that the feature was removed.

- [x] **Step 5: Run whitespace and status checks.**

  ```bash
  git diff --check
  git status --short --untracked-files=all ptycho/single_image_frc.py frc/single_image_frc.py frc/__init__.py ptycho/evaluation.py
  ```

  Expected: no whitespace errors; tracked code no longer requires `ptycho/single_image_frc.py`; the untracked local file is either still untracked and reported, or removed after explicit cleanup approval.

## Dependency Analysis

- This removes a metric payload that some study scripts assumed existed. The main risk is stale downstream references to `single_frc50` / `single_frc1over7`.
- The plan intentionally keeps ground-truth FRC metrics untouched.
- The plan intentionally avoids committing the current untracked `ptycho/single_image_frc.py`.
- The plan should not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py`.

## Execution Notes

- 2026-04-13: Implemented the hard-removal path. `ptycho/single_image_frc.py` remains an untracked local file and was not deleted or committed.
- 2026-04-13: Targeted verification passed: single-image FRC removal tests, grid-lines selector tests, wrapper selector tests, metrics-table tests, HIO benchmark tests, import guard, production reference scan, and `git diff --check`.

## Workflow Compatibility Contract

When executed by a backlog workflow:

- Backlog item must point `plan_path` to this file.
- `check_commands` should include the Task 6 targeted pytest selectors, import preflight, reference scan, and `git diff --check`.
- Completion requires:
  - no default import/runtime dependency on `ptycho.single_image_frc`
  - no tracked public API for `single_image_frc_metrics`
  - no grid-lines flags or kwargs for single-image FRC
  - target tests pass

## Verification Commands

```bash
pytest tests/test_evaluation_single_image_frc.py -v
pytest tests/test_grid_lines_workflow.py -k single_image_frc -v
pytest tests/torch/test_grid_lines_torch_runner.py -k single_image_frc -v
pytest tests/test_grid_lines_compare_wrapper.py -k single_image_frc -v
pytest tests/studies/test_metrics_tables.py -v
python - <<'PY'
import builtins
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "ptycho.single_image_frc":
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
import ptycho.evaluation
print("evaluation import ok without single-image FRC")
PY
rg -n "single_image_frc|single_frc50|single_frc1over7|single-image FRC|1FRC" ptycho frc scripts tests docs
git diff --check
```

## Completion Criteria

- [x] `import ptycho.evaluation` succeeds when `ptycho.single_image_frc` import is blocked.
- [x] `eval_reconstruction(...)` no longer accepts `single_image_frc*` kwargs.
- [x] `eval_reconstruction(...)` returns legacy ground-truth metric keys and no `single_frc50` / `single_frc1over7` keys.
- [x] `ptycho.evaluation.single_image_frc_metrics` is absent.
- [x] `frc.single_image_frc` tracked shim is deleted and `frc.__init__` does not export removed helpers.
- [x] Grid-lines TF/Torch/wrapper code no longer exposes or passes single-image FRC controls.
- [x] Default metrics tables omit single-image FRC columns.
- [x] No tracked production code imports an absent `ptycho.single_image_frc` or `frc.single_image_frc`.
- [x] Revision checklist is updated if this changes paper-revision study readiness.

## Artifacts Index

- Plan: `docs/plans/2026-04-13-single-image-frc-removal.md`
- Reports root if executed: `docs/plans/2026-04-13-single-image-frc-removal/reports/`
