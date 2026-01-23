# Response — Legacy dose_experiments Ground-Truth Bundle

**From:** Maintainer <1> (PtychoPINN dose_experiments branch, root_dir: ~/Documents/PtychoPINN/)
**To:** Maintainer <2> (PtychoPINN active branch, root_dir: ~/Documents/tmp/PtychoPINN/)
**Re:** Request — legacy dose_experiments ground-truth artifacts (2026-01-22T014445Z)

---

## 1. Delivery Summary

The requested ground-truth bundle is now available at:

```
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/
```

**Bundle structure:**
```
dose_experiments_ground_truth/
├── simulation/     # 7 datasets (data_p1e3.npz ... data_p1e9.npz)
├── training/       # params.dill, baseline_model.h5, recon.dill
├── inference/      # wts.h5.zip (PINN weights)
└── docs/           # README.md, manifests, summaries
```

**Documentation assets (under `docs/`):**
- `README.md` — Commands, environment requirements, provenance tables, NPZ schema
- `ground_truth_manifest.json` — Machine-readable manifest with full SHA256 checksums
- `ground_truth_manifest.md` — Human-readable manifest summary
- `dose_baseline_summary.json` — Baseline metrics snapshot

---

## 2. Verification Summary

Per `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.md`:

| Metric | Value |
|--------|-------|
| Total files | 15 |
| Verified | 15/15 |
| Total size | 278.18 MB |
| Tarball size | 270.70 MB |
| Tarball SHA256 | `7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72` |

Full verification details available in:
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.json`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T002823Z/bundle_verification.md`

---

## 3. Test Validation

```bash
pytest tests/test_generic_loader.py::test_generic_loader -q
```

**Result:** 1 passed, 5 warnings (2.54s)

**Log:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/pytest_loader.log`

SHA256 recomputed and matched:
```
7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72  dose_experiments_ground_truth.tar.gz
```

---

## 4. How-To: Extract and Verify

### Extract tarball
```bash
cd plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/
tar -xzf dose_experiments_ground_truth.tar.gz
```

### Verify SHA256
```bash
sha256sum -c dose_experiments_ground_truth.tar.gz.sha256
# Expected: dose_experiments_ground_truth.tar.gz: OK
```

### Re-run helper CLIs (optional)

The CLIs under `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/` can regenerate manifests/READMEs if needed:

```bash
# Generate manifest (read-only, no GPU required)
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/make_ground_truth_manifest.py --help

# Generate README
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/generate_legacy_readme.py --help

# Package bundle (copies files, creates tarball)
python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/package_ground_truth_bundle.py --help
```

---

## 5. Dataset Table

All NPZ files conform to `specs/data_contracts.md` §RawData NPZ.

| File | Photon Dose | Size | SHA256 |
|------|-------------|------|--------|
| `data_p1e3.npz` | 1e3 | 4.85 MB | `f9fa3f9f3f1cf8fc181f9c391b5acf15247ba3f25ed54fe6aaf9809388a860b2` |
| `data_p1e4.npz` | 1e4 | 11.07 MB | `1cce1fe9596a82290bffc3cd6116f43cfed17abe7339503dd849ec5826378402` |
| `data_p1e5.npz` | 1e5 | 16.58 MB | `01007daf8afc67aad3ad037e93077ba8bfb28b58d07e17a1e539bd202ffa0d95` |
| `data_p1e6.npz` | 1e6 | 24.64 MB | `95cfd6aee6b2c061e2a8fbe62a824274165e39e0c4514e4537ee1fecf7a79f64` |
| `data_p1e7.npz` | 1e7 | 35.55 MB | `9902ae24e90d2fa63bebf7830a538868cea524fab9cb15a512509803a9896251` |
| `data_p1e8.npz` | 1e8 | 44.45 MB | `56b4f66a92aa28b2983757417cad008ef14c45c796b2d61d9e366aae3a3d55cf` |
| `data_p1e9.npz` | 1e9 | 55.29 MB | `3e1f229af34525a7a912c9c62fa8df6ab87c69528572686a34de4d2640c57c4a` |

**NPZ key requirements** (per `specs/data_contracts.md` §RawData NPZ):
- Required: `xcoords`, `ycoords`, `xcoords_start`, `ycoords_start`, `diff3d`, `probeGuess`, `scan_index`
- Optional: `objectGuess`, `ground_truth_patches`

---

## 6. Baseline Artifacts Table

| File | Type | Size | SHA256 |
|------|------|------|--------|
| `params.dill` | params | 34.79 KB | `92c27229e2edca3a279d9efd6c8134378cc82b6efd38f0aba751128fb48eb588` |
| `baseline_model.h5` | baseline_output | 52.93 MB | `46b88686b95ce4e437561ddcb8ad052e2138fc7bd48b5b66f27b7958246d878c` |
| `recon.dill` | baseline_output | 820.04 KB | `2501b93db2fea8e3751dee6649503b8dfd62aa72c4b077c27e5773af3b1b304c` |
| `wts.h5.zip` | pinn_weights | 31.98 MB | `56a26314a6c6db4fb466673f8eb308f4b8502d9a5bc3d79d60bf641f71b5b1cd` |

**Baseline metrics (train / test):**
- MS-SSIM: 0.9248 / 0.9206
- PSNR: 71.32 dB / 158.06 dB
- intensity_scale_value (learned): 988.21

---

## 7. Key Parameters

| Parameter | Value |
|-----------|-------|
| N (patch size) | 64 |
| gridsize | 1 |
| nepochs | 50 |
| batch_size | 16 |
| loss | NLL-only (nll_weight=1.0, mae_weight=0.0) |
| probe.trainable | False |
| intensity_scale.trainable | True |

---

## 8. Next Steps

Please confirm receipt and let me know if:
1. The tarball extracts correctly and SHA256 matches
2. The datasets load without errors in your environment
3. Any additional artifacts or documentation are needed

Once acknowledged, I will mark DEBUG-SIM-LINES-DOSE-001.D1 complete in `docs/fix_plan.md`.

---

**Artifacts path:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T004049Z/`
**Request source:** `inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md`

---

## 9. Rehydration Verification

The tarball `dose_experiments_ground_truth.tar.gz` was extracted into a fresh temporary directory, and the manifest was regenerated from the extracted files. All 11 files matched the original manifest exactly (SHA256 + size).

**Status:** `PASS`

| Metric | Count |
|--------|-------|
| Total files | 11 |
| Matches | 11 |
| Mismatches | 0 |

**Verification script:** `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/verify_bundle_rehydration.py`

**Artifacts:**
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_summary.md`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/rehydration_diff.json`
- `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/rehydration_check/verify_bundle_rehydration.log`

**Pytest validation (post-rehydration):**
```bash
pytest tests/test_generic_loader.py::test_generic_loader -q
# Result: 1 passed, 5 warnings (2.53s)
```
**Log:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-23T005200Z/pytest_loader.log`

This confirms the tarball can be extracted and used as a drop-in replacement for the original dataset files.
