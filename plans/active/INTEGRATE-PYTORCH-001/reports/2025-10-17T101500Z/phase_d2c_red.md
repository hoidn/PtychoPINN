# Phase D2.C Red Phase — Inference/Stitching Path Design

**Initiative:** INTEGRATE-PYTORCH-001
**Phase:** D2.C — Inference + stitching workflow
**Mode:** TDD Red Phase
**Timestamp:** 2025-10-17T101500Z

---

## Summary

Authored red-phase parity test `test_run_cdi_example_invokes_training` in `tests/torch/test_workflows_components.py` to document the expected behavior of `run_cdi_example_torch` when implementing full orchestration per TF baseline `ptycho/workflows/components.py:676-723`.

## Test Contract

### Entry Signature
```python
def run_cdi_example_torch(
    train_data: Union[RawData, RawDataTorch, PtychoDataContainerTorch],
    test_data: Optional[Union[RawData, RawDataTorch, PtychoDataContainerTorch]],
    config: TrainingConfig,
    flip_x: bool = False,
    flip_y: bool = False,
    transpose: bool = False,
    M: int = 20,
    do_stitching: bool = False
) -> Tuple[Optional[Any], Optional[Any], Dict[str, Any]]
```

### Required Behavior

1. **CONFIG-001 Compliance**: Already implemented — calls `update_legacy_dict(params.cfg, config)` before delegating
2. **Training Delegation**: MUST invoke `train_cdi_model_torch(train_data, test_data, config)` first
3. **No-Stitching Path** (`do_stitching=False`):
   - Return `(None, None, results_dict)` where results_dict contains training outputs
4. **Stitching Path** (`do_stitching=True` + `test_data is not None`):
   - After training completes, invoke reassembly helper
   - Return `(recon_amp, recon_phase, results_dict)` with merged training + reconstruction outputs
5. **TensorFlow Parity**: Mirror behavior of `ptycho/workflows/components.py:676-723`

## Test Design

### Monkeypatch Strategy
- Spy on `train_cdi_model_torch` to validate:
  - Function was called with correct `(train_data, test_data, config)` arguments
  - Results dict from training is propagated to final return value
- For D2.C red phase, only test `do_stitching=False` path (simplest case)
- Stitching path testing deferred to separate test case (can be added incrementally)

### Red Phase Results

**Pytest Command:**
```bash
pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsRun::test_run_cdi_example_invokes_training -vv
```

**Status:** ❌ FAILED (expected)
**Error:** `NotImplementedError: PyTorch training path not yet implemented`

This is the correct red-phase outcome — test documents expected API, implementation does not yet exist.

## Implementation Roadmap (Green Phase)

### Step 1: Replace `NotImplementedError` with orchestration logic

Current stub in `ptycho_torch/workflows/components.py:154-170`:
```python
raise NotImplementedError(
    "PyTorch training path not yet implemented. "
    "Phase D2.B will implement Lightning trainer orchestration. "
    "See plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md for roadmap."
)
```

**Replace with:**
```python
# Step 1: Train the model
train_results = train_cdi_model_torch(train_data, test_data, config)

# Step 2: Initialize return values
recon_amp, recon_phase = None, None

# Step 3: Optional stitching path
if do_stitching and test_data is not None:
    logger.info("Performing image stitching...")
    recon_amp, recon_phase, reassemble_results = _reassemble_cdi_image_torch(
        test_data, config, flip_x, flip_y, transpose, M
    )
    # Merge reassembly outputs into training results
    train_results.update(reassemble_results)
else:
    logger.info("Skipping image stitching (disabled or no test data available)")

# Step 4: Return tuple per TF baseline signature
return recon_amp, recon_phase, train_results
```

### Step 2: Implement `_reassemble_cdi_image_torch` helper (stub for Phase D2.C)

**Signature:**
```python
def _reassemble_cdi_image_torch(
    test_data: Union[RawData, RawDataTorch, PtychoDataContainerTorch],
    config: TrainingConfig,
    flip_x: bool,
    flip_y: bool,
    transpose: bool,
    M: int
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]
```

**Stub Implementation (Phase D2.C initial):**
```python
def _reassemble_cdi_image_torch(
    test_data, config, flip_x, flip_y, transpose, M
):
    """
    Reassemble CDI image using trained PyTorch model (stub for Phase D2.C).

    Full implementation will:
    1. Normalize test_data → PtychoDataContainerTorch via _ensure_container
    2. Run model inference to get reconstructed patches
    3. Apply coordinate transformations (flip_x, flip_y, transpose)
    4. Reassemble patches using reassemble_position equivalent
    5. Extract amplitude + phase from complex reconstruction
    6. Return (recon_amp, recon_phase, results_dict)
    """
    raise NotImplementedError(
        "PyTorch inference/stitching path not yet implemented. "
        "Phase D2.C will implement model inference + reassembly logic. "
        "See plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md D2.C for details."
    )
```

**Note:** For Phase D2.C TDD cycle, we'll keep this as a stub that raises `NotImplementedError`. The test currently only validates the `do_stitching=False` path, so this stub won't be invoked yet.

### Step 3: Update Test Expectations

Once green-phase implementation lands, update test to remove `pytest.raises(NotImplementedError)` wrapper and add assertions validating the orchestration flow succeeded.

## Design Parameters

### TensorFlow Baseline Reference

From `ptycho/workflows/components.py:676-723`:
```python
def run_cdi_example(train_data, test_data, config, flip_x, flip_y, transpose, M, do_stitching):
    # 1. Update params.cfg (CONFIG-001)
    update_legacy_dict(params.cfg, config)

    # 2. Train model
    train_results = train_cdi_model(train_data, test_data, config)

    recon_amp, recon_phase = None, None

    # 3. Optional stitching
    if do_stitching and test_data is not None and 'reconstructed_obj' in train_results:
        logger.info("Performing image stitching...")
        recon_amp, recon_phase, reassemble_results = reassemble_cdi_image(
            test_data, config, flip_x, flip_y, transpose, M=M
        )
        train_results.update(reassemble_results)
    else:
        logger.info("Skipping image stitching (disabled or no test data available)")

    return recon_amp, recon_phase, train_results
```

### PyTorch Parity Notes

1. **CONFIG-001 Guard**: Already implemented at `ptycho_torch/workflows/components.py:155-157` ✅
2. **Training Delegation**: Will call `train_cdi_model_torch` (already implemented in Phase D2.B) ✅
3. **Stitching Guard**: Condition `do_stitching and test_data is not None` mirrors TF exactly
4. **Result Merging**: Use `.update()` pattern identical to TF baseline
5. **Logging**: Mirror TF log messages for consistency

### Torch-Optional Considerations

- No new torch imports required (reuses existing `_ensure_container`, `train_cdi_model_torch`)
- `_reassemble_cdi_image_torch` stub won't break torch-optional imports (returns NotImplementedError when called)
- Full stitching implementation (Phase D3+) will need torch-optional guards for model inference

## Exit Criteria

Red phase complete when:
- ✅ Test authored documenting expected API
- ✅ Test runs and fails with expected `NotImplementedError`
- ✅ Red-phase pytest log captured at `pytest_red.log`

Green phase complete when:
- [ ] `run_cdi_example_torch` orchestration logic implemented
- [ ] Test passes without `NotImplementedError` (monkeypatch validates orchestration flow)
- [ ] Green-phase pytest log captured
- [ ] Regression test confirms no new failures in full test suite

---

**Next Step:** Implement green-phase orchestration logic per roadmap above, then rerun targeted test to validate.
