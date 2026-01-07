# PARALLEL-API-INFERENCE — Task 1: Extract TF Inference Helper

**Summary:** Extract reusable TF inference helper from `scripts/inference/inference.py` following Phase A extraction design.

**Focus:** PARALLEL-API-INFERENCE — Programmatic TF/PyTorch API parity (Task 1)

**Branch:** feature/torchapi-newprompt-2

**Mapped tests:**
- `tests/test_integration_workflow.py` — regression for TF inference (existing)
- `tests/scripts/test_tf_inference_helper.py::TestTFInferenceHelper` — new signature tests (to be created)

**Artifacts:** `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/`

---

## Do Now

### Task 1: Extract `_run_tf_inference_and_reconstruct()` helper

**Implement: `scripts/inference/inference.py::_run_tf_inference_and_reconstruct`**

1. **Create new helper function** at `scripts/inference/inference.py` (after `perform_inference`):

   ```python
   def _run_tf_inference_and_reconstruct(
       model: tf.keras.Model,
       raw_data: "RawData",
       config: dict,
       K: int = 4,
       nsamples: Optional[int] = None,
       quiet: bool = False,
       debug_dump_dir: Optional[Path] = None,
       debug_patch_limit: int = 16,
       seed: int = 45,
   ) -> Tuple[np.ndarray, np.ndarray]:
       """
       Core TensorFlow inference helper for programmatic use.

       Mirrors PyTorch `_run_inference_and_reconstruct()` signature for API parity.

       Args:
           model: Loaded TensorFlow model (from load_inference_bundle_with_backend)
           raw_data: RawData instance with test data
           config: Dict with 'N', 'gridsize' keys (from model bundle)
           K: Number of nearest neighbors (default: 4)
           nsamples: Number of samples; if None, uses all available
           quiet: Suppress progress output
           debug_dump_dir: Optional directory for debug artifacts
           debug_patch_limit: Patches to visualize in debug mode
           seed: Random seed for reproducibility (default: 45)

       Returns:
           Tuple of (amplitude, phase) as numpy arrays

       Notes:
           - Expects params.cfg to be populated via CONFIG-001 before call
           - Ground truth not returned (use extract_ground_truth separately)
       """
   ```

2. **Move core logic from `perform_inference` body** (lines 341-424) into new helper:
   - Lines 348-349: Set random seeds
   - Lines 353-354: Generate grouped data via `raw_data.generate_grouped_data()`
   - Lines 362-364: Create `PtychoDataContainer` via `loader.load()`
   - Lines 370-372: Call `reconstruct_image()` and capture offsets
   - Lines 376-381: Reassemble and extract amplitude/phase
   - Lines 415-422: Debug artifact dump (conditional)
   - Return only `(amplitude, phase)` — not ground truth

3. **Create `extract_ground_truth()` utility** in same file:

   ```python
   def extract_ground_truth(raw_data: "RawData") -> Optional[Tuple[np.ndarray, np.ndarray]]:
       """
       Extract ground truth amplitude/phase from RawData if available and valid.

       Returns:
           (amplitude, phase) tuple if valid ground truth exists, else None
       """
       from ptycho.nbutils import crop_to_non_uniform_region_with_buffer

       if not hasattr(raw_data, 'objectGuess') or raw_data.objectGuess is None:
           return None
       if np.allclose(raw_data.objectGuess, 0, atol=1e-10):
           return None
       obj_complex = raw_data.objectGuess
       if (np.allclose(obj_complex.real, obj_complex.real.flat[0], atol=1e-10) and
           np.allclose(obj_complex.imag, obj_complex.imag.flat[0], atol=1e-10)):
           return None

       epie_phase = crop_to_non_uniform_region_with_buffer(np.angle(obj_complex), buffer=-20)
       epie_amplitude = crop_to_non_uniform_region_with_buffer(np.abs(obj_complex), buffer=-20)
       return (epie_amplitude, epie_phase)
   ```

4. **Refactor `perform_inference` as deprecated wrapper**:

   ```python
   def perform_inference(model, test_data, config, K, nsamples,
                         debug_dump_dir=None, debug_patch_limit=16):
       """
       .. deprecated::
           Use `_run_tf_inference_and_reconstruct()` for new code.
       """
       import warnings
       warnings.warn(
           "perform_inference is deprecated; use _run_tf_inference_and_reconstruct",
           DeprecationWarning, stacklevel=2
       )
       amp, phase = _run_tf_inference_and_reconstruct(
           model, test_data, config, K=K, nsamples=nsamples,
           quiet=False, debug_dump_dir=debug_dump_dir,
           debug_patch_limit=debug_patch_limit, seed=45,
       )
       gt = extract_ground_truth(test_data)
       if gt:
           return amp, phase, gt[0], gt[1]
       return amp, phase, None, None
   ```

5. **Create minimal test** `tests/scripts/test_tf_inference_helper.py`:

   ```python
   """Tests for TensorFlow inference helper extraction (PARALLEL-API-INFERENCE)."""
   import pytest
   import inspect


   class TestTFInferenceHelper:
       """Validate _run_tf_inference_and_reconstruct signature and availability."""

       def test_helper_is_importable(self):
           """Helper function can be imported from scripts.inference.inference."""
           from scripts.inference.inference import _run_tf_inference_and_reconstruct
           assert callable(_run_tf_inference_and_reconstruct)

       def test_helper_signature_matches_spec(self):
           """Helper signature has required parameters per extraction design."""
           from scripts.inference.inference import _run_tf_inference_and_reconstruct
           sig = inspect.signature(_run_tf_inference_and_reconstruct)
           params = list(sig.parameters.keys())

           # Required positional params
           assert 'model' in params
           assert 'raw_data' in params
           assert 'config' in params

           # Optional params with defaults
           assert 'K' in params
           assert 'nsamples' in params
           assert 'quiet' in params
           assert 'seed' in params

       def test_extract_ground_truth_is_importable(self):
           """extract_ground_truth utility can be imported."""
           from scripts.inference.inference import extract_ground_truth
           assert callable(extract_ground_truth)

       def test_deprecated_wrapper_still_works(self):
           """perform_inference wrapper emits deprecation warning."""
           import warnings
           from scripts.inference.inference import perform_inference
           # Just check it's callable — full test is in integration suite
           assert callable(perform_inference)
   ```

### Verification

```bash
# Create test directory
mkdir -p tests/scripts

# Run new tests
pytest tests/scripts/test_tf_inference_helper.py -v 2>&1 | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/pytest_tf_helper.log

# Run integration regression
pytest tests/test_integration_workflow.py -v 2>&1 | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/pytest_integration.log

# Collect test count
pytest tests/scripts/ --collect-only 2>&1 | tee plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/pytest_collect.log
```

---

## How-To Map

| Step | Command | Artifact |
|------|---------|----------|
| Run new helper tests | `pytest tests/scripts/test_tf_inference_helper.py -v` | `pytest_tf_helper.log` |
| Run integration regression | `pytest tests/test_integration_workflow.py -v` | `pytest_integration.log` |
| Collect test count | `pytest tests/scripts/ --collect-only` | `pytest_collect.log` |

---

## Pitfalls To Avoid

1. **DO NOT** change the return type of existing `perform_inference` — it must remain backward-compatible (4-tuple)
2. **DO NOT** remove ground truth handling from `perform_inference` — move to separate utility, call from wrapper
3. **DO NOT** modify `params.cfg` handling — assume CONFIG-001 compliance as precondition
4. **DO** use `from __future__ import annotations` for forward references in type hints
5. **DO** keep the `quiet` parameter for progress suppression (matches PyTorch helper)
6. **DO** preserve existing DEBUG print statements in helper (move them, don't delete)
7. **DO** preserve `_dump_tf_inference_debug_artifacts` call for debug mode

---

## If Blocked

If imports fail or tests don't collect:
1. Check Python path includes repo root (`export PYTHONPATH=.`)
2. Verify `tests/scripts/__init__.py` exists
3. Log exact error to `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/blocker.md`

---

## Findings Applied

| Finding ID | Adherence |
|------------|-----------|
| CONFIG-001 | Document precondition in docstring; assume params.cfg populated before call |
| POLICY-001 | N/A (TensorFlow-only extraction) |
| ANTIPATTERN-001 | Avoid import-time side effects; lazy imports inside function body |

---

## Pointers

- Extraction design: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T010000Z/extraction_design.md`
- TF inference source: `scripts/inference/inference.py:321-428`
- PyTorch reference: `ptycho_torch/inference.py:426-632`
- Fix plan item: `docs/fix_plan.md` § PARALLEL-API-INFERENCE
- Testing guide: `docs/TESTING_GUIDE.md` § Integration Tests

---

## Next Up (if finished early)

- Task 2: Create `scripts/pytorch_api_demo.py` with both backends
