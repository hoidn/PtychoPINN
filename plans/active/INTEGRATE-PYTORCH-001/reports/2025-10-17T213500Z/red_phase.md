# Phase E2.B2 — Red Test Evidence & Failure Analysis

**Initiative:** INTEGRATE-PYTORCH-001
**Task ID:** E2.B2
**Created:** 2025-10-17T213500Z
**Purpose:** Document PyTorch integration test failures (RED phase) before Phase E2.C implementation.

---

## 1. Test Execution Summary

**Test Selector:**
```bash
pytest tests/torch/test_integration_workflow_torch.py -vv
```

**Result:**
- **1 FAILED** — `test_pytorch_train_save_load_infer_cycle`
- **1 SKIPPED** — `test_pytorch_tf_output_parity` (deferred to Phase E2.D)
- **Runtime:** 5.35s

**Artifacts:**
- Test log: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/phase_e_red_integration.log`
- Test source: `tests/torch/test_integration_workflow_torch.py`

---

## 2. Failure Analysis

### 2.1. Primary Failure: ModuleNotFoundError (lightning)

**Error Message:**
```
ModuleNotFoundError: No module named 'lightning'
```

**Source:**
```python
File "/home/ollie/Documents/PtychoPINN2/ptycho_torch/train.py", line 23, in <module>
  import lightning as L
```

**Root Cause:**
PyTorch Lightning (`lightning`) is not installed in the current environment. The `ptycho_torch/train.py` module requires Lightning for training orchestration.

**Expected Behavior (Phase E2.C Target):**
- Lightning should be listed as a PyTorch backend dependency
- Installation via `pip install -e .[torch]` or explicit `pip install lightning`
- Alternatively, TEST-PYTORCH-001 may propose standalone training script without Lightning dependency

**Remediation:**
- **Option A:** Add `lightning` to `setup.py` optional dependencies under `[torch]` extra
- **Option B:** Document Lightning as a PyTorch runtime prerequisite in `docs/workflows/pytorch.md`
- **Option C:** Refactor `ptycho_torch/train.py` to avoid Lightning dependency (TEST-PYTORCH-001 decision)

---

### 2.2. Secondary Failure: CLI Interface Mismatch

**Expected CLI Flags (per test fixture_sync.md §2):**
```bash
python -m ptycho_torch.train \
  --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output_dir <temp_dir>/training_outputs \
  --nepochs 2 \
  --n_images 64 \
  --gridsize 1 \
  --batch_size 4 \
  --device cpu \
  --disable_mlflow
```

**Current `ptycho_torch/train.py` Interface (per docs/workflows/pytorch.md):**
```python
train.main(ptycho_dir, probe_dir)  # Function-based, not CLI
```

**Gap:**
`ptycho_torch/train.py` does not expose a CLI entrypoint compatible with `python -m ptycho_torch.train`. The current interface expects:
- Function calls with directory arguments (`ptycho_dir`, `probe_dir`)
- No `--train_data_file`, `--output_dir`, or `--device` flags
- No `--disable_mlflow` flag (required for CI per TEST-PYTORCH-001 §Prerequisites)

**Expected Behavior (Phase E2.C Target):**
- Add `argparse` CLI wrapper to `ptycho_torch/train.py`
- Support flags mirroring TensorFlow training script (`scripts/training/train.py`)
- Enable `python -m ptycho_torch.train <args>` invocation

**Remediation (Phase E2.C):**
1. Add CLI argument parser to `ptycho_torch/train.py`
2. Map CLI flags to existing function interface
3. Add `--disable_mlflow` flag to suppress MLflow autologging
4. Validate against test fixture parameters (§fixture_sync.md §2)

---

### 2.3. Inference Module Missing

**Expected Command (per test):**
```bash
python -m ptycho_torch.inference \
  --model_path <training_output_dir> \
  --test_data <data_file> \
  --output_dir <inference_output_dir> \
  --n_images 32 \
  --device cpu
```

**Current Status:**
`ptycho_torch/inference.py` does not exist yet (per TEST-PYTORCH-001 §Open Questions #2).

**Expected Behavior (Phase E2.C Target):**
- Author `ptycho_torch/inference.py` module
- Load Lightning checkpoint via `LightningModule.load_from_checkpoint()`
- Execute `trainer.predict()` or manual forward pass
- Generate reconstruction outputs (amplitude/phase images)
- Align output naming with TensorFlow inference script (`reconstructed_amplitude.png`, `reconstructed_phase.png`)

**Remediation (Phase E2.C or TEST-PYTORCH-001):**
1. Author `ptycho_torch/inference.py` with CLI entrypoint
2. Implement checkpoint loading and batch prediction
3. Add reconstruction visualization helpers
4. Document in TEST-PYTORCH-001 §Next Steps #3

---

## 3. Test Coverage Validation

### 3.1. Expected Test Outcomes (Phase E2.B Red Phase)

| Test Case | Expected Result | Actual Result | Analysis |
| :--- | :--- | :--- | :--- |
| `test_pytorch_train_save_load_infer_cycle` | **FAIL** (Lightning import error or CLI mismatch) | **FAIL** (ModuleNotFoundError: lightning) | ✅ Expected failure — confirms test is correctly detecting missing implementation |
| `test_pytorch_tf_output_parity` | **SKIP** (deferred to Phase E2.D) | **SKIP** (Phase E2.D parity verification) | ✅ Expected skip — parity comparison requires both backends functional |

### 3.2. Failure Mode Validation

**Checklist:**
- [x] Test fails due to missing PyTorch backend implementation (not pytest/test infrastructure issues)
- [x] Error messages are actionable (ModuleNotFoundError, traceback to `ptycho_torch/train.py:23`)
- [x] Test assumptions documented in fixture_sync.md align with failure evidence
- [x] Failure captures subprocess stderr/stdout for debugging
- [x] Test structure mirrors TensorFlow integration test (`tests/test_integration_workflow.py`)

---

## 4. Phase E2.C Implementation Guidance

### 4.1. Required Changes (derived from red test evidence)

1. **Dependency Management:**
   - Add `lightning` to `setup.py` optional dependencies: `extras_require={'torch': ['torch>=2.2', 'lightning', 'mlflow', 'tensordict']}`
   - Update `docs/workflows/pytorch.md` §2 Prerequisites to list Lightning requirement

2. **CLI Interface (`ptycho_torch/train.py`):**
   - Add `argparse` CLI parser mirroring `scripts/training/train.py` interface
   - Support flags: `--train_data_file`, `--test_data_file`, `--output_dir`, `--nepochs`/`--max_epochs`, `--n_images`, `--gridsize`, `--batch_size`, `--device`, `--disable_mlflow`
   - Ensure `python -m ptycho_torch.train` entrypoint works
   - Validate params against fixture_sync.md §2 reproduction commands

3. **Inference Module (`ptycho_torch/inference.py`):**
   - Author new module with CLI entrypoint
   - Load Lightning checkpoint from `--model_path`
   - Execute prediction on `--test_data` NPZ
   - Generate reconstruction images in `--output_dir`
   - Naming convention: `reconstructed_amplitude.png`, `reconstructed_phase.png` (match TensorFlow outputs)

4. **CONFIG-001 Compliance:**
   - Ensure `update_legacy_dict(ptycho.params.cfg, config)` called before data loading
   - Add assertion in test to verify `params.cfg['N']` and `params.cfg['gridsize']` populated

5. **Artifact Persistence:**
   - Define checkpoint naming convention (e.g., `<output_dir>/checkpoints/last.ckpt` or `<output_dir>/wts.pt`)
   - Document in TEST-PYTORCH-001 and update test assertions

### 4.2. Acceptance Criteria (Phase E2.C Green Phase)

- [ ] `pytest tests/torch/test_integration_workflow_torch.py::test_pytorch_train_save_load_infer_cycle -vv` **PASSES**
- [ ] Training subprocess completes with returncode 0
- [ ] Checkpoint artifact created in expected location
- [ ] Inference subprocess loads checkpoint and generates reconstructions
- [ ] Output images pass file size validation (>1000 bytes)
- [ ] No Lightning import errors
- [ ] CLI flags match fixture_sync.md reproduction commands

---

## 5. Risk Log

| Risk | Severity | Mitigation |
| :--- | :--- | :--- |
| Lightning dependency adds ~500MB install footprint | Medium | Document optional dependency in setup.py; TEST-PYTORCH-001 may propose lighter alternative |
| CLI interface drift from TensorFlow baseline | High | Enforce parameter naming parity in E2.C implementation; cross-check with `scripts/training/train.py` |
| Checkpoint format incompatibility with Ptychodus | High | Coordinate with Phase E1 backend dispatcher design; ensure persistence contract honored |
| MLflow autologging pollutes temp dirs in CI | Medium | Implement `--disable_mlflow` flag (TEST-PYTORCH-001 §Prerequisites) |
| Diffraction shape transpose (H, W, n) vs (n, H, W) | Medium | Document in fixture_sync.md §5; validate with `tests/torch/test_data_pipeline.py` |

---

## 6. Next Actions

1. **Phase E2.C (Implementation):**
   - Install Lightning: `pip install lightning`
   - Add CLI wrapper to `ptycho_torch/train.py` (§4.1 item 2)
   - Author `ptycho_torch/inference.py` (§4.1 item 3)
   - Re-run `pytest tests/torch/test_integration_workflow_torch.py -vv` → expect GREEN

2. **Documentation Updates:**
   - Update `docs/workflows/pytorch.md` §2 Prerequisites with Lightning requirement
   - Add CLI usage examples to pytorch workflow guide
   - Document checkpoint format in TEST-PYTORCH-001

3. **Handoff to TEST-PYTORCH-001:**
   - Share red phase evidence (this document + phase_e_red_integration.log)
   - Request decision on Lightning dependency vs standalone training script
   - Propose canonical checkpoint naming convention

---

## 7. References

- **Test Source:** `tests/torch/test_integration_workflow_torch.py`
- **Test Log:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/phase_e_red_integration.log`
- **Fixture Inventory:** `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/phase_e_fixture_sync.md`
- **Phase E Plan:** `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` §E2.B
- **TEST-PYTORCH-001 Plan:** `plans/pytorch_integration_test_plan.md`
- **TensorFlow Baseline Test:** `tests/test_integration_workflow.py`

---

**Status:** RED phase evidence captured successfully. Test fails as expected with actionable error messages. Ready for Phase E2.C implementation guidance handoff.
