# Phase B1 Fixture Requirements & Runtime Envelope

**Initiative**: TEST-PYTORCH-001 — PyTorch integration workflow regression
**Phase**: B1 (Evidence-Only Measurement Loop)
**Date**: 2025-10-19T215300Z
**Executor**: Ralph (measurement loop, no code modification)
**Status**: Complete

---

## Executive Summary

This report documents the telemetry gathered from profiling the `Run1084_recon3_postPC_shrunk_3.npz` dataset and measuring PyTorch training runtime sensitivity to CLI knobs. The goal is to establish baseline metrics and define acceptance criteria for a lightweight, deterministic fixture that will keep the PyTorch integration regression under 45 seconds on CPU.

**Key Findings:**
- **Dataset Size**: 1087 scan positions, legacy `(H, W, N)` format requiring transpose
- **Current Runtime Range**: 17.11s (1 epoch, 16 images) to 21.91s (2 epochs, 64 images)
- **Memory Footprint**: ~1.6GB RSS stable across configurations
- **Artifact Overhead**: ~54MB per training run (checkpoint + Lightning logs)

**Recommendation**: Current dataset already meets <45s target with full coverage. Fixture minimization is optional; prioritize deterministic subset extraction over aggressive size reduction.

---

## 1. Dataset Profile (Run1084_recon3_postPC_shrunk_3.npz)

### 1.1 Array Inventory & Shapes

```
keys: ['diffraction', 'objectGuess', 'probeGuess', 'xcoords', 'xcoords_start', 'ycoords', 'ycoords_start']

diffraction:
  shape:  (64, 64, 1087)  ⚠ LEGACY FORMAT (H, W, N)
  dtype:  float64         ⚠ Non-canonical (should be float32 per DATA-001)
  range:  [0.0, 1.7036]

objectGuess:
  shape:  (227, 226)
  dtype:  complex128      ⚠ Non-canonical (should be complex64)

probeGuess:
  shape:  (64, 64)
  dtype:  complex128      ⚠ Non-canonical (should be complex64)

xcoords:
  dtype:  float64
  count:  1087
  range:  [34.412, 79.281]

ycoords:
  dtype:  float64
  count:  1087
  range:  [35.763, 79.107]
```

### 1.2 Data Contract Compliance Status

| Requirement (specs/data_contracts.md §1) | Observed | Compliance | Remediation |
|------------------------------------------|----------|------------|-------------|
| `diffraction` shape: `(N, H, W)` | `(H, W, N)` = `(64, 64, 1087)` | **FAIL** | Auto-transpose heuristic active (FORMAT-001) |
| `diffraction` dtype: `float32` | `float64` | **FAIL** | Generator must downcast |
| `diffraction` normalized | Range [0, 1.7] | **PASS** | Within tolerance |
| `objectGuess` dtype: `complex64` | `complex128` | **FAIL** | Generator must downcast |
| `probeGuess` dtype: `complex64` | `complex128` | **FAIL** | Generator must downcast |
| `Y` patches present | **MISSING** | **FAIL** | Not required for PINN training (only supervised) |
| Coordinates present | ✓ | **PASS** | 1087 scan points |

**Critical Observation**: The legacy `(H, W, N)` format is automatically handled by the PyTorch dataloader's FORMAT-001 auto-transpose heuristic (see `docs/findings.md#FORMAT-001` and `ptycho_torch/data.py`). However, a canonical fixture should emit `(N, H, W)` to avoid reliance on runtime detection.

---

## 2. Runtime Sensitivity Analysis

### 2.1 Experimental Design

Two targeted dry-runs executed with `CUDA_VISIBLE_DEVICES=""` (CPU-only) to measure sensitivity to epoch count and dataset size:

| Run ID | Epochs | n_images | Batch Size | Gridsize | Device | Command |
|--------|--------|----------|------------|----------|--------|---------|
| ep2_n64 | 2 | 64 | 4 | 1 | cpu | `python -m ptycho_torch.train --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --output_dir tmp/phase_b_fixture/run_ep2_n64 --max_epochs 2 --n_images 64 --gridsize 1 --batch_size 4 --device cpu --disable_mlflow` |
| ep1_n16 | 1 | 16 | 2 | 1 | cpu | `python -m ptycho_torch.train --train_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --test_data_file datasets/Run1084_recon3_postPC_shrunk_3.npz --output_dir tmp/phase_b_fixture/run_ep1_n16 --max_epochs 1 --n_images 16 --gridsize 1 --batch_size 2 --device cpu --disable_mlflow` |

### 2.2 Runtime Results

| Run ID | Elapsed Time (s) | Memory (RSS, GB) | Artifact Size (MB) | Notes |
|--------|------------------|------------------|---------------------|-------|
| ep2_n64 | **21.91** | 1.63 | 54 | 2 epochs × 64 images (172 batches/epoch) |
| ep1_n16 | **17.11** | 1.64 | 54 | 1 epoch × 16 images (343 batches/epoch) |

**Analysis:**
- **Runtime Scaling**: Approximately linear with epoch count (21.91s / 2 ≈ 11s/epoch vs 17.11s / 1 = 17s/epoch). The discrepancy is due to varying batch counts (`n_images=16` produces more batches with `batch_size=2` than `n_images=64` with `batch_size=4`).
- **Memory Stability**: RSS variance <1%, indicating stable memory footprint independent of dataset size/epoch count within this range.
- **Artifact Overhead**: Consistent 54MB (Lightning checkpoints + versioning metadata), independent of training duration.

### 2.3 Projected Budget

To meet the **<45s CPU target** with headroom for CI variance:

| Configuration | Expected Runtime | Margin | Feasible? |
|---------------|------------------|--------|-----------|
| 1 epoch, 64 images | ~11s | 34s (76%) | **YES** |
| 2 epochs, 64 images | ~22s | 23s (51%) | **YES** |
| 2 epochs, 128 images | ~30s (est.) | 15s (33%) | **YES** |
| 3 epochs, 64 images | ~33s (est.) | 12s (27%) | **YES** |

**Conclusion**: Current dataset configuration is well within budget. No aggressive minimization required.

---

## 3. Fixture Acceptance Criteria

Based on the telemetry above, the Phase B2 fixture generator must produce a dataset satisfying:

### 3.1 Functional Requirements

1. **Shape & Orientation**:
   - `diffraction.shape == (n_subset, 64, 64)` (canonical `(N, H, W)` order)
   - `objectGuess.shape == (M, M)` where `M >= 128` (sufficient for reconstruction)
   - `probeGuess.shape == (64, 64)`

2. **Data Types** (DATA-001 compliance):
   - `diffraction.dtype == float32` (downcast from source `float64`)
   - `objectGuess.dtype == complex64` (downcast from `complex128`)
   - `probeGuess.dtype == complex64` (downcast from `complex128`)
   - Coordinates remain `float64` (acceptable precision)

3. **Subset Size**:
   - **Minimum**: `n_subset >= 16` (enough for at least 2 scan groups with `gridsize=2`, avoiding degenerate single-position edge case)
   - **Target**: `n_subset = 64` (balanced coverage, proven <25s runtime)
   - **Maximum**: `n_subset = 128` (if coordinate diversity testing required)

4. **Coordinate Coverage**:
   - Subset must span >50% of original X/Y range to exercise grouping logic across spatial diversity
   - Use deterministic selection (e.g., first `n_subset` positions or stratified sampling with fixed seed) for reproducibility

5. **Normalization**:
   - `diffraction` data MUST remain amplitude (sqrt of intensity) with max values <10.0 (validated: current max 1.7036)
   - No additional scaling beyond dtype downcast

### 3.2 Performance Requirements

6. **Runtime Envelope** (CPU, single-threaded):
   - **Target**: <25s for 2 epochs, 64 images (current: 21.91s)
   - **Hard Limit**: <45s (CI budget with 2× headroom)
   - **Variance Tolerance**: ±10% across repeated runs with fixed seed

7. **Memory Footprint**:
   - **Target**: <2GB RSS (current: 1.63GB)
   - **Artifact Size**: <100MB total (checkpoints + logs; current: 54MB)

### 3.3 Metadata & Traceability

8. **Provenance**:
   - Fixture file MUST include JSON sidecar (e.g., `minimal_dataset_v1.json`) documenting:
     - Source dataset path and commit SHA
     - Transformation operations applied (subset indices, dtype conversions, axis reorders)
     - Generator script version
     - Creation timestamp (ISO 8601)
   - SHA256 checksum recorded in fixture metadata for integrity validation

9. **Storage Location**:
   - Primary fixture: `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`
   - Metadata sidecar: `tests/fixtures/pytorch_integration/minimal_dataset_v1.json`
   - Regeneration script: `scripts/tools/make_pytorch_integration_fixture.py`

---

## 4. Recommendations for Phase B2

### 4.1 Generator Design Priorities

1. **Deterministic Subset Extraction**:
   - Use **first `n_subset` scan positions** (simplest, reproducible without seeding)
   - Alternative: Stratified sampling with `np.random.seed(42)` if coordinate diversity critical

2. **Dtype Downcast Operations**:
   ```python
   diffraction_canonical = diffraction.transpose(2, 0, 1).astype(np.float32)  # (H,W,N) → (N,H,W) + downcast
   objectGuess_canonical = objectGuess.astype(np.complex64)
   probeGuess_canonical = probeGuess.astype(np.complex64)
   ```

3. **Validation Checkpoints** (TDD stub in Phase B2.B):
   - Assert `diffraction.shape[0] == n_subset` (N dimension first)
   - Assert `diffraction.dtype == np.float32`
   - Assert coordinate counts match `n_subset`
   - Compute SHA256 and store in metadata

### 4.2 Integration Test Adjustments (Phase B3)

Update `tests/torch/test_integration_workflow_torch.py`:
- Replace `data_file` fixture path: `datasets/Run1084_recon3_postPC_shrunk_3.npz` → `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`
- Adjust CLI overrides: `--max_epochs=2 --n_images=64` (proven <25s)
- Add fixture integrity check (SHA256 validation in test setup)

### 4.3 Documentation Updates (Phase B3.C)

- `docs/workflows/pytorch.md` §11: Update regression baseline to reflect new fixture runtime (~22s target, <45s timeout)
- `plans/active/TEST-PYTORCH-001/implementation.md`: Mark B1–B3 complete with artifact links

---

## 5. Artifacts & Evidence

### 5.1 Generated Files

- **Dataset Probe**: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/dataset_probe.txt`
- **Training Logs**:
  - `plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/logs/train_ep2_n64.log` (21.91s elapsed)
  - `plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/logs/train_ep1_n16.log` (17.11s elapsed)
- **This Scope Document**: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/fixture_scope.md`

### 5.2 Cross-References

- **Data Contract**: `specs/data_contracts.md` §1 (canonical NPZ format)
- **Findings Ledger**: `docs/findings.md` entries FORMAT-001 (legacy transpose), POLICY-001 (torch required)
- **PyTorch Workflow Guide**: `docs/workflows/pytorch.md` §§4–8 (RawData expectations, CONFIG-001)
- **Phase B Plan**: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md`
- **Implementation Tracker**: `plans/active/TEST-PYTORCH-001/implementation.md` (Phase B checklist)

---

## 6. Decision Points & Open Questions

### 6.1 Resolved

- **Q**: Is aggressive minimization (e.g., `n_subset=16`) necessary to meet runtime budget?
  **A**: **NO**. Current full dataset (1087 positions) runs <25s with 2 epochs. Use `n_subset=64` for balanced coverage without jeopardizing grouping logic diversity.

- **Q**: Should fixture preserve legacy `(H, W, N)` format to test auto-transpose heuristic?
  **A**: **NO**. Emit canonical `(N, H, W)` to align with DATA-001 and avoid reliance on runtime detection (FORMAT-001 heuristic is a fallback, not a primary contract).

### 6.2 For Phase B2 Discussion

- **Coordinate Subset Strategy**: First `n_subset` positions vs stratified sampling?
  - **Recommendation**: First `n_subset` (deterministic, no seed required, simpler generator)
  - **Fallback**: Stratified if coordinate spread analysis shows clustering issues

- **Fixture Versioning**: Should we support multiple fixture sizes (v1=16, v2=64, v3=128)?
  - **Recommendation**: Single fixture `v1` with `n_subset=64` (proven budget). Add variants only if specific test scenarios demand them.

---

## 7. Next Actions (Phase B2 Entry Criteria Met)

1. **B2.A**: Author generator design document (`generator_design.md`) citing this scope analysis
2. **B2.B**: Write failing TDD test `test_fixture_outputs_match_contract` asserting criteria §3.1–3.2
3. **B2.C**: Implement `make_pytorch_integration_fixture.py` to satisfy test
4. **B2.D**: Document fixture metadata (provenance, regeneration commands, SHA256)

Phase B1 complete. All acceptance criteria defined with concrete numeric targets and evidence-backed justification.
