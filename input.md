# PARALLEL-API-INFERENCE — Task 2: Update Demo Script to Use New TF Helper

**Summary:** Update `scripts/pytorch_api_demo.py` to use `_run_tf_inference_and_reconstruct` and add smoke test.

**Focus:** PARALLEL-API-INFERENCE — Programmatic TF/PyTorch API parity (Task 2-3)

**Branch:** feature/torchapi-newprompt-2

**Mapped tests:**
- `tests/scripts/test_api_demo.py::TestPyTorchApiDemo` — new smoke test (to be created)
- `tests/scripts/test_tf_inference_helper.py::TestTFInferenceHelper` — regression (existing, 7 tests)

**Artifacts:** `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T030000Z/`

---

## Do Now

### Task 2: Update demo script to use new TF helper

**Implement: `scripts/pytorch_api_demo.py::run_backend`**

The demo script currently uses the old TF inference path (`tf_components.perform_inference`). Update it to use the new `_run_tf_inference_and_reconstruct` helper for API parity.

1. **Add import for new helper** (replace `tf_components.perform_inference`):

   ```python
   from scripts.inference.inference import (
       _run_tf_inference_and_reconstruct as tf_infer,
       extract_ground_truth,
   )
   ```

2. **Update TF inference call** in `run_backend()` (lines 79-82):

   Replace:
   ```python
   else:
       container = tf_components.create_ptycho_data_container(test_data, infer_cfg.model)
       amp, phase = tf_components.perform_inference(model, container, params_dict, infer_cfg, quiet=True)
       tf_components.save_outputs(amp, phase, {}, infer_cfg.output_dir)
   ```

   With:
   ```python
   else:
       # Use new helper for API parity with PyTorch path
       config = {'N': infer_cfg.model.N, 'gridsize': infer_cfg.model.gridsize}
       amp, phase = tf_infer(
           model=model,
           raw_data=RawData.from_file(str(DATA)),
           config=config,
           K=7,  # neighbor_count
           nsamples=infer_cfg.n_groups,
           quiet=True,
       )
       # Save outputs
       import numpy as np
       out_path = infer_cfg.output_dir
       out_path.mkdir(parents=True, exist_ok=True)
       np.savez(out_path / "reconstruction.npz", amplitude=amp, phase=phase)
   ```

3. **Remove unused import** `tf_components` if no longer needed.

### Task 3: Add smoke test

**Implement: `tests/scripts/test_api_demo.py`**

Create minimal smoke test that verifies the demo script's `run_backend` function works for both backends.

```python
"""Smoke tests for scripts/pytorch_api_demo.py (PARALLEL-API-INFERENCE)."""
import pytest
from pathlib import Path
import shutil


class TestPyTorchApiDemo:
    """Smoke tests for the unified API demo script."""

    @pytest.fixture
    def work_dir(self, tmp_path):
        """Create temporary work directory."""
        return tmp_path / "api_demo_test"

    def test_demo_module_importable(self):
        """Demo script can be imported without side effects."""
        from scripts import pytorch_api_demo
        assert hasattr(pytorch_api_demo, 'run_backend')
        assert hasattr(pytorch_api_demo, 'main')

    def test_run_backend_function_signature(self):
        """run_backend function has expected signature."""
        import inspect
        from scripts.pytorch_api_demo import run_backend
        sig = inspect.signature(run_backend)
        params = list(sig.parameters.keys())
        assert 'backend' in params
        assert 'out_dir' in params

    @pytest.mark.slow
    def test_tensorflow_backend_runs(self, work_dir):
        """TensorFlow backend executes without error (slow)."""
        from scripts.pytorch_api_demo import run_backend

        out_dir = work_dir / "tf"
        run_backend("tensorflow", out_dir)

        # Verify outputs exist
        assert (out_dir / "train_outputs").exists()
        assert (out_dir / "inference_outputs").exists()
        assert (out_dir / "inference_outputs" / "reconstruction.npz").exists()

    @pytest.mark.slow
    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="PyTorch not available"),
        reason="PyTorch required"
    )
    def test_pytorch_backend_runs(self, work_dir):
        """PyTorch backend executes without error (slow)."""
        from scripts.pytorch_api_demo import run_backend

        out_dir = work_dir / "pytorch"
        run_backend("pytorch", out_dir)

        # Verify outputs exist
        assert (out_dir / "train_outputs").exists()
        assert (out_dir / "inference_outputs").exists()
```

### Verification

```bash
# Run new smoke tests (imports only, fast)
pytest tests/scripts/test_api_demo.py -v -k "not slow" 2>&1 | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T030000Z/pytest_api_demo.log

# Run TF helper regression
pytest tests/scripts/test_tf_inference_helper.py -v 2>&1 | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T030000Z/pytest_tf_helper_regression.log

# Collect test count
pytest tests/scripts/ --collect-only 2>&1 | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T030000Z/pytest_collect.log
```

---

## How-To Map

| Step | Command | Artifact |
|------|---------|----------|
| Run smoke tests (fast) | `pytest tests/scripts/test_api_demo.py -v -k "not slow"` | `pytest_api_demo.log` |
| Run TF helper regression | `pytest tests/scripts/test_tf_inference_helper.py -v` | `pytest_tf_helper_regression.log` |
| Collect test count | `pytest tests/scripts/ --collect-only` | `pytest_collect.log` |

---

## Pitfalls To Avoid

1. **DO NOT** break the existing PyTorch path — only update TF path
2. **DO NOT** change the `DATA` path or fixture requirements
3. **DO** ensure params.cfg is populated before TF inference (CONFIG-001)
4. **DO** use `from __future__ import annotations` for forward references
5. **DO** preserve the training step — only update inference
6. **DO** add `# noqa: F401` if imports appear unused (they're for CLI/main)
7. **DO** mark slow tests with `@pytest.mark.slow` for CI gating

---

## If Blocked

If imports fail or the demo script errors:
1. Check `params.cfg` initialization via `update_legacy_dict`
2. Verify the minimal fixture exists at `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`
3. Log exact error to `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T030000Z/blocker.md`

---

## Findings Applied

| Finding ID | Adherence |
|------------|-----------|
| CONFIG-001 | Ensure params.cfg populated before TF inference |
| POLICY-001 | PyTorch path unchanged, torch available |
| ANTIPATTERN-001 | Lazy imports inside function; no module-level side effects |

---

## Pointers

- New TF helper: `scripts/inference/inference.py:353-457`
- PyTorch helper: `ptycho_torch/inference.py:426-632`
- Demo script: `scripts/pytorch_api_demo.py:33-84`
- Fix plan item: `docs/fix_plan.md` § PARALLEL-API-INFERENCE
- Testing guide: `docs/TESTING_GUIDE.md` § Integration Tests

---

## Next Up (if finished early)

- Task 4: Document in `docs/workflows/pytorch.md` (add "Programmatic usage" section)
