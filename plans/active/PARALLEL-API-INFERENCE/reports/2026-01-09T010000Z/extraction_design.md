# TensorFlow Inference Helper Extraction Design

**Date:** 2026-01-09
**Initiative:** PARALLEL-API-INFERENCE Phase A
**Mode:** Planning (exploration only)

---

## 1. Current State Analysis

### TensorFlow Inference (`scripts/inference/inference.py`)

The TF inference script (737 lines) is CLI-centric with interleaved orchestration and core logic:

| Function | Lines | Purpose | Reusable |
|----------|-------|---------|----------|
| `perform_inference()` | 321-428 | Core inference logic | Partially |
| `save_comparison_plot()` | 430-489 | 2x2 comparison figure | Yes |
| `save_reconstruction_images()` | 490-531 | Separate amp/phase PNGs | Yes |
| `_dump_tf_inference_debug_artifacts()` | 221-318 | Debug artifact generation | Yes |
| `setup_inference_configuration()` | 178-219 | CLI config construction | No |
| `interpret_sampling_parameters()` | 124-177 | Sampling interpretation | Yes |
| `main()` | 561-737 | CLI orchestration | No |

### PyTorch Inference (`ptycho_torch/inference.py`)

Already well-abstracted with `_run_inference_and_reconstruct()` (lines 426-632):
- Clean signature: `(model, raw_data, config, execution_config, device, ...)`
- Returns: `(amplitude, phase)` as numpy arrays
- No CLI state dependency in core helper

---

## 2. Dependencies Analysis for `perform_inference()`

### Direct Imports (within function body)
```python
from ptycho.nbutils import reconstruct_image, crop_to_non_uniform_region_with_buffer
from ptycho import loader
from ptycho.tf_helper import reassemble_position
```

### Implicit Dependencies
| Dependency | Source | Propagation |
|------------|--------|-------------|
| `params.cfg['N']` | via `config` dict | Passed as parameter |
| `params.cfg['gridsize']` | via `config` dict | Passed as parameter |
| `RawData.generate_grouped_data()` | Uses `params.cfg` internally | OK if CONFIG-001 compliant |
| `tf.random.set_seed()` | Called inside | OK for isolation |
| `loader.load()` | Expects grouped data dict | OK |
| `reconstruct_image()` | Expects `PtychoDataContainer` | OK |

### Return Values
```python
return reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase
```
- `epie_amplitude`/`epie_phase` are ground truth if available, else `None`

---

## 3. Proposed Extraction

### New Function: `_run_tf_inference_and_reconstruct()`

**Location:** `scripts/inference/inference.py` (or new module `ptycho/inference_helper.py`)

**Signature (mirroring PyTorch pattern):**
```python
def _run_tf_inference_and_reconstruct(
    model: tf.keras.Model,
    raw_data: RawData,
    config: InferenceConfig,
    K: int = 4,
    nsamples: Optional[int] = None,
    quiet: bool = False,
    debug_dump_dir: Optional[Path] = None,
    debug_patch_limit: int = 16,
    seed: int = 45,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Core TensorFlow inference helper for programmatic use.

    Args:
        model: Loaded TensorFlow model (from load_inference_bundle_with_backend)
        raw_data: RawData instance with test data (from load_data or RawData.from_file)
        config: InferenceConfig with model.gridsize, model.N, etc.
        K: Number of nearest neighbors for grouped data generation (default: 4)
        nsamples: Number of samples; if None, uses all available
        quiet: Suppress progress output
        debug_dump_dir: Optional directory for debug artifacts
        debug_patch_limit: Number of patches to visualize in debug mode
        seed: Random seed for reproducibility

    Returns:
        Tuple of (amplitude, phase) numpy arrays

    Notes:
        - Expects params.cfg to be populated via CONFIG-001 before call
        - Ground truth comparison not returned (use save_comparison_plot separately)
    """
```

**Key Changes from `perform_inference()`:**
1. Accept `InferenceConfig` instead of raw dict (type safety)
2. Return only `(amplitude, phase)` — matches PyTorch helper
3. Move ground truth extraction to separate optional function
4. Make seed explicit parameter instead of hardcoded
5. Remove CLI-specific logging redirection

### Extraction Steps

1. **Move `perform_inference()` body** into `_run_tf_inference_and_reconstruct()`
2. **Extract ground truth handling** into `extract_ground_truth(raw_data) -> Optional[Tuple[np.ndarray, np.ndarray]]`
3. **Update `main()`** to call new helper
4. **Add docstrings** with CONFIG-001 prerequisites

---

## 4. CLI Wrapper Changes

After extraction, `main()` becomes a thin wrapper:

```python
def main():
    args = parse_arguments()
    config = setup_inference_configuration(args, args.config)

    # CONFIG-001: update_legacy_dict called inside load_inference_bundle_with_backend
    model, _ = load_inference_bundle_with_backend(config.model_path, config)

    test_data = load_data(args.test_data, n_images=..., n_subsample=..., ...)

    # NEW: Call extracted helper
    amplitude, phase = _run_tf_inference_and_reconstruct(
        model=model,
        raw_data=test_data,
        config=config,
        K=4,
        nsamples=nsamples,
        quiet=False,
        debug_dump_dir=debug_dump_dir,
    )

    # Save outputs
    save_reconstruction_images(amplitude, phase, config.output_dir)

    # Optional comparison
    if args.comparison_plot:
        gt = extract_ground_truth(test_data)
        if gt:
            save_comparison_plot(amplitude, phase, gt[0], gt[1], config.output_dir)
```

---

## 5. Integration Points with Backend Selector

**Current `ptycho/workflows/backend_selector.py`:**
- `load_inference_bundle_with_backend()` — Returns `(model, config_dict)`
- Delegates to TF or PyTorch loaders based on `config.backend`

**Unified Demo Script Pattern:**
```python
from ptycho.workflows.backend_selector import load_inference_bundle_with_backend
from scripts.inference.inference import _run_tf_inference_and_reconstruct
from ptycho_torch.inference import _run_inference_and_reconstruct as torch_run_inference

def run_inference(backend: str, model_path: Path, test_data: RawData, config: InferenceConfig):
    model, _ = load_inference_bundle_with_backend(model_path, config)

    if backend == 'tensorflow':
        return _run_tf_inference_and_reconstruct(model, test_data, config)
    else:
        # PyTorch path
        return torch_run_inference(model, test_data, config, execution_config, device)
```

---

## 6. Return Value Specification

### Current TF `perform_inference()`:
```python
return (reconstructed_amplitude, reconstructed_phase, epie_amplitude, epie_phase)
```
- 4-tuple with optional ground truth

### Proposed `_run_tf_inference_and_reconstruct()`:
```python
return (amplitude, phase)
```
- 2-tuple matching PyTorch helper
- Ground truth handled separately via `extract_ground_truth()`

### Ground Truth Helper (new):
```python
def extract_ground_truth(raw_data: RawData) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract ground truth amplitude/phase from RawData if available and valid.

    Returns:
        (epie_amplitude, epie_phase) tuple if valid ground truth exists, else None
    """
```

---

## 7. Implementation Risk Assessment

| Risk | Mitigation |
|------|------------|
| Breaking existing CLI behavior | Keep `perform_inference()` as deprecated wrapper calling new helper |
| Implicit `params.cfg` dependencies | Document CONFIG-001 precondition; fail-fast if params not populated |
| Test coverage gaps | Add `tests/test_tf_inference_helper.py` with minimal fixture |
| PyTorch parity drift | Unified test comparing both outputs on same synthetic data |

---

## 8. Summary

**Extraction yields a new helper:**
- `_run_tf_inference_and_reconstruct(model, raw_data, config, ...) -> (amp, phase)`
- Mirrors PyTorch `_run_inference_and_reconstruct()` signature
- Enables programmatic use without CLI dependencies
- Ground truth handling moved to separate utility

**No code changes in this planning phase — design document only.**

---

## References

- TF inference: `scripts/inference/inference.py:321-428`
- PyTorch inference: `ptycho_torch/inference.py:426-632`
- Backend selector: `ptycho/workflows/backend_selector.py`
- Data contracts: `specs/data_contracts.md`
- Ptychodus API spec: `specs/ptychodus_api_spec.md`
