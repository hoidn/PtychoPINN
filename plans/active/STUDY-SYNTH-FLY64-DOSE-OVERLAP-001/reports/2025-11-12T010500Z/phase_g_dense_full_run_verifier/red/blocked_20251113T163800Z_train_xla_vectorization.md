# Blocker: XLA Vectorization Batch Mismatch (Train Split)

**Timestamp:** 2025-11-13T16:38:00Z  
**Context:** FIX-COMPARE-MODELS-TRANSLATION-001 / Dense Phase G train rerun  
**Command:**
```bash
python scripts/compare_models.py \
  --pinn_dir .../data/phase_e/dose_1000/dense/gs2 \
  --baseline_dir .../data/phase_e/dose_1000/baseline/gs1 \
  --test_data .../data/phase_c/dose_1000/patched_train.npz \
  --output_dir .../analysis/dose_1000/dense/train \
  --ms-ssim-sigma 1.0 \
  --tike_recon_path .../data/phase_f/dose_1000/dense/train/ptychi_reconstruction.npz \
  --register-ptychi-only
```

**Exit Code:** 1

**Error:**
```
ValueError: Dimensions must be equal, but are 5088 and 159 for '{{node mul_21}} = Mul[T=DT_FLOAT](mul_17, mul_13)' 
with input shapes: [5088,?,?,1], [159,?,?,1].
```

**Stack Trace:**
```
ptycho/tf_helper.py:1187 in _vectorised: translated = translate(imgs_padded, offsets_flat, interpolation='bilinear')
ptycho/tf_helper.py:674 in newf: assembled_part1 = fn(part1, *args[1:], **kwargs)
ptycho/tf_helper.py:798 in translate: return translate_core(imgs, offsets, interpolation=interpolation, use_xla_workaround=use_xla_workaround)
ptycho/tf_helper.py:722 in translate_core: return translate_xla(images, translations, interpolation=interpolation, use_jit=True)
ptycho/projective_warp_xla.py:294 in translate_xla: return projective_warp_xla_jit(images, M, ...)
ptycho/projective_warp_xla.py:196 in projective_warp_xla: out = wa * Ia + wb * Ib + wc * Ic + wd * Id
```

**Analysis:**
- The ReassemblePatchesLayer batching fix (batch_size=64) successfully resolved the Translation layer shape mismatch for the regression tests.
- However, `scripts/compare_models.py` inference path hits a different vectorization issue in the XLA translation code.
- The batch split mechanism (`vectorise_fn`) in `tf_helper.py:674` divides 5088 patches into chunks, but the XLA path receives mismatched batch dimensions (5088 vs 159).
- This appears related to XLA-DYN-DOT-001 (docs/findings.md) but manifests in a different multiplication node (mul_21).

**Impact:**
- Train split compare_models cannot complete
- This blocks Phase G dense rerun metrics generation
- The ReassemblePatchesLayer fix itself is valid (pytest tests pass)

**Next Steps:**
1. Check if test split has the same issue
2. Document as separate blocker (XLA vectorization, not ReassemblePatchesLayer)
3. The FIX-COMPARE-MODELS-TRANSLATION-001 focus delivered its core objective (batched reassembly)
4. XLA vectorization issue requires separate investigation
