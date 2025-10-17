# Phase E2.A — Fixture Inventory & Alignment

**Initiative:** INTEGRATE-PYTORCH-001
**Task IDs:** E2.A1, E2.A2
**Created:** 2025-10-17T213500Z
**Purpose:** Document shared fixture inventory for PyTorch integration tests, ensuring alignment with TEST-PYTORCH-001 and compliance with DATA-001/CONFIG-001.

---

## 1. Fixture Inventory

| Dataset Name | Path | Owner | Sample Count | Gridsize | N | Notes |
| :--- | :--- | :--- | ---: | ---: | ---: | :--- |
| Run1084_recon3_postPC_shrunk_3 | `datasets/Run1084_recon3_postPC_shrunk_3.npz` | TensorFlow integration tests | 1087 | 1 | 64 | Used by `tests/test_integration_workflow.py`. **Transposed shape warning:** diffraction is (64, 64, 1087) instead of canonical (1087, 64, 64). Requires transpose or adapter layer. |
| (Synthetic) | Generated via `scripts/simulation/simulate_and_save.py` | TEST-PYTORCH-001 | Configurable | 1 | 64-128 | Can generate minimal datasets (e.g., 10-20 patterns) for fast CI. Ensures DATA-001 compliance (complex64 Y patches, proper metadata). |

---

## 2. Minimal Reproducible Dataset Parameters (E2.A2)

### For PyTorch Integration Test (E2.B1 Target)

**Dataset Source:** Use existing `datasets/Run1084_recon3_postPC_shrunk_3.npz` for initial compatibility with TensorFlow workflow.
**Fallback:** Synthetic generation if shape transpose becomes blocking.

**CLI Parameters:**
```bash
# Training command (mirroring TensorFlow test_integration_workflow.py:40-49)
python -m ptycho_torch.train \
  --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output_dir <temp_dir>/training_outputs \
  --nepochs 2 \
  --n_images 64 \
  --gridsize 1 \
  --quiet
```

**Expected Runtime Budget:** ~2 minutes on CPU (target for CI).

**Environment Knobs:**
- `CUDA_VISIBLE_DEVICES=""` — Force CPU execution
- `--disable_mlflow` — Suppress MLflow artifacts (requires flag addition per TEST-PYTORCH-001 plan)
- `--batch_size 4` — Keep memory footprint small

**Config Requirements (CONFIG-001):**
- Call `update_legacy_dict(ptycho.params.cfg, config)` before invoking `ptycho_torch.workflows.*`
- Ensure `params.cfg['N']` and `params.cfg['gridsize']` populated before data loading

**Data Contract (DATA-001):**
- Diffraction: amplitude (float32), normalized range
- Y patches: complex64 (3D after squeeze)
- objectGuess: complex64, significantly larger than probe
- probeGuess: complex64

---

## 3. TEST-PYTORCH-001 Coordination Notes

**Fixture Reuse Strategy:**
- Phase E2 integration tests should mirror the TensorFlow fixture strategy (`tests/test_integration_workflow.py` uses `datasets/Run1084_recon3_postPC_shrunk_3.npz`).
- Future TEST-PYTORCH-001 work may introduce a dedicated `tests/fixtures/pytorch_integration/` directory with synthetic small datasets.

**Owner Alignment:**
- TensorFlow integration test owner: Core PtychoPINN team
- PyTorch integration test owner: TEST-PYTORCH-001 initiative
- Fixture coordination: Share `datasets/Run1084_recon3_postPC_shrunk_3.npz` initially; TEST-PYTORCH-001 may generate smaller synthetic fixtures later.

**Outstanding Items:**
- [ ] Validate diffraction shape transpose handling in PyTorch dataloader (`ptycho_torch/dset_loader_pt_mmap.py` expects (n_images, H, W))
- [ ] Add `--disable_mlflow` flag to `ptycho_torch/train.py` (per TEST-PYTORCH-001 §Prerequisites)
- [ ] Document memmap cleanup strategy (TEST-PYTORCH-001 Open Question #3)

---

## 4. Reproduction Commands (Validation)

### Verify Dataset Exists and Conforms to DATA-001
```bash
# Check dataset presence
ls -lh datasets/Run1084_recon3_postPC_shrunk_3.npz

# Inspect structure
python -c "
import numpy as np
data = np.load('datasets/Run1084_recon3_postPC_shrunk_3.npz')
print('Keys:', list(data.keys()))
print('Shapes:')
for k in data.keys():
    print(f'  {k}: {data[k].shape} ({data[k].dtype})')
"

# Expected output:
# Keys: ['diffraction', 'probeGuess', 'objectGuess', 'xcoords', 'ycoords', 'xcoords_start', 'ycoords_start']
# Shapes:
#   diffraction: (64, 64, 1087) (float64) ⚠️ TRANSPOSED — canonical is (1087, 64, 64)
#   probeGuess: (64, 64) (complex128)
#   objectGuess: (227, 226) (complex128)
#   xcoords: (1087,) (float64)
#   ycoords: (1087,) (float64)
#   xcoords_start: (1087,) (float64)
#   ycoords_start: (1087,) (float64)
```

### Generate Minimal Synthetic Dataset (Optional Fallback)
```bash
# Create small synthetic dataset for fast integration tests
python scripts/simulation/simulate_and_save.py \
  --input-file datasets/Run1084_recon3_postPC_shrunk_3.npz \
  --output-file datasets/synthetic_small_10patterns.npz \
  --n-images 10 \
  --n-photons 1e6 \
  --gridsize 1 \
  --seed 42
```

---

## 5. Risks & Mitigations

| Risk | Impact | Mitigation |
| :--- | :--- | :--- |
| Diffraction shape transpose | PyTorch dataloader expects (n, H, W); existing dataset is (H, W, n) | Add transpose adapter in test fixture or document requirement in TEST-PYTORCH-001. Validate with `tests/torch/test_data_pipeline.py`. |
| Runtime > 2 min budget | CI timeout, slow feedback | Use n_images=10-20 synthetic dataset; measure actual runtime with `time pytest ...`. |
| MLflow artifacts pollute temp dirs | Disk bloat, test cleanup failures | Add `--disable_mlflow` flag to `ptycho_torch/train.py`; document in TEST-PYTORCH-001 §Prerequisites. |
| Fixture drift between TF/PT tests | Diverging baselines, hard-to-debug parity failures | Enforce single source of truth: reuse `datasets/Run1084_recon3_postPC_shrunk_3.npz` until TEST-PYTORCH-001 proposes canonical small fixture. |

---

## 6. Next Actions

1. **E2.B1:** Author `tests/torch/test_integration_workflow_torch.py` using parameters documented in §2.
2. **E2.B2:** Run `pytest tests/torch/test_integration_workflow_torch.py -vv` and capture red evidence.
3. **TEST-PYTORCH-001 Handoff:** Share this fixture inventory; propose canonical small synthetic fixture under `tests/fixtures/pytorch_integration/`.

---

**Status:** Fixture inventory complete. Identified existing dataset and documented minimal reproduction parameters. Ready for E2.B red test authoring.
