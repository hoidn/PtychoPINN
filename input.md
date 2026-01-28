# GRID-LINES-WORKFLOW-001 — Fix 2 Failing Torch Runner Tests

**Summary:** Fix 2 test failures in `tests/torch/test_grid_lines_torch_runner.py` caused by missing NPZ metadata in the `synthetic_npz` fixture.

**Focus:** GRID-LINES-WORKFLOW-001 — Grid-based lines simulation + training workflow

**Branch:** fno2

**Mapped tests:**
- `tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_creates_run_directory_structure` — currently FAILING
- `tests/torch/test_grid_lines_torch_runner.py::TestOutputContractConversion::test_runner_returns_predictions_complex` — currently FAILING
- `tests/torch/test_grid_lines_torch_runner.py` — full suite regression (21 pass, 2 fail)

**Artifacts:** `plans/active/GRID-LINES-WORKFLOW-001/reports/2026-01-28T000000Z/`

---

## Do Now

### Fix: `tests/torch/test_grid_lines_torch_runner.py::synthetic_npz` fixture

**Implement: `tests/torch/test_grid_lines_torch_runner.py::synthetic_npz`**

The `synthetic_npz` fixture at line 22 saves NPZ files using plain `np.savez()`, but the runner's `_stitch_for_metrics()` → `_configure_stitching_params()` at `scripts/studies/grid_lines_torch_runner.py:140-155` loads NPZ via `MetadataManager.load_with_metadata()` and requires metadata with `additional_parameters.nimgs_test` and `additional_parameters.outer_offset_test`.

**Root cause:** The fixture doesn't embed metadata. `MetadataManager.load_with_metadata()` returns `None` for metadata, then `_configure_stitching_params` raises `ValueError("Missing metadata; cannot stitch predictions for metrics.")`.

**Fix:** Update the `synthetic_npz` fixture to save using `MetadataManager.save_with_metadata()` instead of `np.savez()`. This is the same pattern used in the passing test `test_metrics_stitch_predictions_to_ground_truth` (line 296-316).

```python
# In synthetic_npz fixture, replace:
#   np.savez(train_path, **data)
#   np.savez(test_path, **data)
# With:
from ptycho.metadata import MetadataManager
from ptycho.config.config import TrainingConfig, ModelConfig

cfg_for_meta = TrainingConfig(model=ModelConfig(N=64, gridsize=1))
metadata = MetadataManager.create_metadata(
    cfg_for_meta,
    script_name="test_fixture",
    nimgs_test=4,  # matches n_samples
    outer_offset_test=20,
)
MetadataManager.save_with_metadata(str(train_path), data, metadata)
MetadataManager.save_with_metadata(str(test_path), data, metadata)
```

Also update the test data's `YY_ground_truth` shape to match what stitching produces for the given params. Add `norm_Y_I` to the data dict if not present. Verify by running the test after the fix.

### Verification

```bash
# Run the 2 failing tests
pytest tests/torch/test_grid_lines_torch_runner.py::TestRunGridLinesTorchScaffold::test_runner_creates_run_directory_structure tests/torch/test_grid_lines_torch_runner.py::TestOutputContractConversion::test_runner_returns_predictions_complex -v 2>&1 | tee plans/active/GRID-LINES-WORKFLOW-001/reports/2026-01-28T000000Z/pytest_fix.log

# Full torch runner test suite regression
pytest tests/torch/test_grid_lines_torch_runner.py -v 2>&1 | tee plans/active/GRID-LINES-WORKFLOW-001/reports/2026-01-28T000000Z/pytest_torch_runner_full.log

# TF workflow tests regression
pytest tests/test_grid_lines_workflow.py -v 2>&1 | tee plans/active/GRID-LINES-WORKFLOW-001/reports/2026-01-28T000000Z/pytest_tf_workflow.log
```

---

## How-To Map

| Step | Command | Artifact |
|------|---------|----------|
| Fix fixture | Edit `tests/torch/test_grid_lines_torch_runner.py::synthetic_npz` | N/A |
| Run failing tests | `pytest tests/torch/test_grid_lines_torch_runner.py -k "test_runner_creates_run or test_runner_returns"` | `pytest_fix.log` |
| Full regression | `pytest tests/torch/test_grid_lines_torch_runner.py -v` | `pytest_torch_runner_full.log` |
| TF regression | `pytest tests/test_grid_lines_workflow.py -v` | `pytest_tf_workflow.log` |

---

## Pitfalls To Avoid

1. **DO NOT** change the runner code (`grid_lines_torch_runner.py`) — this is a test fixture bug, not a code bug
2. **DO** keep the fixture's data shapes consistent (N=64, gridsize=1, n_samples=4)
3. **DO** verify that `norm_Y_I` is available in the loaded test data (add to fixture if missing)
4. **DO** ensure `YY_ground_truth` shape matches what stitching produces for the given params
5. **DO NOT** add mocks for `_stitch_for_metrics` — the fixture should provide real metadata so the integration path is tested
6. **DO** check the passing test `test_metrics_stitch_predictions_to_ground_truth` (line 296) as a reference for correct metadata setup

---

## If Blocked

If `MetadataManager.create_metadata` or `save_with_metadata` fail:
1. Check import: `from ptycho.metadata import MetadataManager`
2. Check `MetadataManager.create_metadata` signature matches current API
3. Log exact error to `plans/active/GRID-LINES-WORKFLOW-001/reports/2026-01-28T000000Z/blocker.md`

---

## Findings Applied

| Finding ID | Adherence |
|------------|-----------|
| CONFIG-001 | Fixture sets N/gridsize via TrainingConfig for metadata |
| STITCH-GRIDSIZE-001 | Using workflow's stitch_predictions which bypasses guard |

---

## Pointers

- Fixture: `tests/torch/test_grid_lines_torch_runner.py:22-45`
- Runner stitch path: `scripts/studies/grid_lines_torch_runner.py:140-167`
- Reference passing test: `tests/torch/test_grid_lines_torch_runner.py:296-316`
- MetadataManager: `ptycho/metadata.py`
- Fix plan: `docs/fix_plan.md` § GRID-LINES-WORKFLOW-001
