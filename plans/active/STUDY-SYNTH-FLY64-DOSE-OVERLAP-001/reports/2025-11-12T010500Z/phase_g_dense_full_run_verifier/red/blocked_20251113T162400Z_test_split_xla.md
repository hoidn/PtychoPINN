# Blocked: Test Split XLA Vectorization Error

**Timestamp:** 2025-11-13T16:24:00Z  
**Command:**
```bash
python scripts/compare_models.py \
  --pinn_dir .../phase_e/dose_1000/dense/gs2 \
  --baseline_dir .../phase_e/dose_1000/baseline/gs1 \
  --test_data .../phase_c/dose_1000/patched_test.npz \
  --output_dir .../analysis/dose_1000/dense/test \
  --ms-ssim-sigma 1.0 \
  --tike_recon_path .../phase_f/dose_1000/dense/test/ptychi_reconstruction.npz \
  --register-ptychi-only
```

**Exit Code:** 1

**Error:**
```
ValueError: Dimensions must be equal, but are 5216 and 163 for '{{node mul_21}} = Mul[T=DT_FLOAT](mul_17, mul_13)' with input shapes: [5216,?,?,1], [163,?,?,1].
```

**Location:** `ptycho/projective_warp_xla.py:196` (XLA vectorized translation)

**Analysis:**
This is a DIFFERENT error from the Translation layer batching issue we fixed. The train split succeeded (exit code 0, metrics generated), proving the batched reassembly fix works. The test split fails in the XLA vectorization code path during stitching, suggesting a batch size mismatch issue unrelated to ReassemblePatchesLayer.

**Next Steps:**
- Train split comparison completed successfully - batching fix is validated
- Test split requires investigation of XLA vectorization batch handling (separate from this focus)
- This blocker does NOT invalidate the FIX-COMPARE-MODELS-TRANSLATION-001 deliverable

**Full Log:** `cli/phase_g_dense_translation_fix_test.log`
