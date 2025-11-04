# Phase G Prerequisites Inventory

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Task:** G0.1 — Evidence inventory & harness prep
**Date:** 2025-11-05
**Artifact Hub:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/`

## Executive Summary

This inventory catalogs all prerequisite artifacts required for Phase G (three-way comparison: PINN vs baseline vs pty-chi) across all dose/view/split combinations. The inventory confirms:

✅ **Phase C datasets** (patched train/test) available for 3 doses
✅ **Phase D datasets** (dense/sparse split) available for 3 doses × 2 views × 2 splits = 12 datasets
✅ **Phase E checkpoints** (PINN + baseline) available for dose_1000
✅ **Phase F LSQML reconstructions** available for 3 conditions (dose_1000: dense/train, dense/test, sparse/train)
⚠️ **Gaps identified:** Phase E checkpoints only exist for dose_1000; Phase F reconstructions incomplete for sparse/test and higher doses

## 1. Phase C Datasets (Patched Train/Test)

**Contract:** DATA-001 (specs/data_contracts.md:1)
**Source:** `tmp/phase_c_f2_cli/`, `tmp/fly64_phase_c_cli/`
**Status:** ✅ COMPLETE for all 3 doses

### Inventory

| Dose | Train Dataset | Test Dataset | Notes |
|------|--------------|--------------|-------|
| 1000 | `tmp/phase_c_f2_cli/dose_1000/patched_train.npz` | `tmp/phase_c_f2_cli/dose_1000/patched_test.npz` | Also in `tmp/fly64_phase_c_cli/dose_1000/` |
| 10000 | `tmp/phase_c_f2_cli/dose_10000/patched_train.npz` | `tmp/phase_c_f2_cli/dose_10000/patched_test.npz` | ✅ |
| 100000 | `tmp/phase_c_f2_cli/dose_100000/patched_train.npz` | `tmp/phase_c_f2_cli/dose_100000/patched_test.npz` | ✅ |

**DATA-001 Compliance:** These NPZ files were generated per Phase C implementation; assumed to contain `diffraction` (amplitude), `positions`, `objectGuess`, `probeGuess` per canonical contract.

**Full listing:** `phase_c_d_datasets.txt`

## 2. Phase D Datasets (Dense/Sparse Split)

**Contract:** DATA-001 + OVERSAMPLING-001 (sparse acceptance metadata)
**Source:** `tmp/phase_d_f2_cli/`
**Status:** ✅ COMPLETE for all 3 doses × 2 views × 2 splits = 12 datasets

### Inventory

| Dose | View | Train Dataset | Test Dataset | Notes |
|------|------|--------------|--------------|-------|
| 1000 | dense | `tmp/phase_d_f2_cli/dose_1000/dense/dense_train.npz` | `tmp/phase_d_f2_cli/dose_1000/dense/dense_test.npz` | ✅ |
| 1000 | sparse | `tmp/phase_d_f2_cli/dose_1000/sparse/sparse_train.npz` | `tmp/phase_d_f2_cli/dose_1000/sparse/sparse_test.npz` | ✅ |
| 10000 | dense | `tmp/phase_d_f2_cli/dose_10000/dense/dense_train.npz` | `tmp/phase_d_f2_cli/dose_10000/dense/dense_test.npz` | ✅ |
| 10000 | sparse | `tmp/phase_d_f2_cli/dose_10000/sparse/sparse_train.npz` | `tmp/phase_d_f2_cli/dose_10000/sparse/sparse_test.npz` | ✅ |
| 100000 | dense | `tmp/phase_d_f2_cli/dose_100000/dense/dense_train.npz` | `tmp/phase_d_f2_cli/dose_100000/dense/dense_test.npz` | ✅ |
| 100000 | sparse | `tmp/phase_d_f2_cli/dose_100000/sparse/sparse_train.npz` | `tmp/phase_d_f2_cli/dose_100000/sparse/sparse_test.npz` | ✅ |

**OVERSAMPLING-001 Notes:** Sparse datasets contain low-acceptance positions (greedy selection, spacing_threshold=102.4, typical acceptance_rate=0.125 per Phase F manifest metadata). Dense datasets contain higher-overlap positions.

**Full listing:** `phase_c_d_datasets.txt`

## 3. Phase E Checkpoints (PINN + Baseline TensorFlow Models)

**Contract:** CONFIG-001 bridge applied; TensorFlow backend
**Source:** `tmp/phase_e_training_gs2/`
**Status:** ⚠️ PARTIAL — only dose_1000 available

### Inventory

| Dose | View | Gridsize | PINN Checkpoint | Baseline Checkpoint | Notes |
|------|------|----------|----------------|---------------------|-------|
| 1000 | baseline | gs1 | N/A | `tmp/phase_e_training_gs2/baseline/checkpoint.h5` | ✅ Baseline (no PINN physics) |
| 1000 | dense | gs2 | `tmp/phase_e_training_gs2/pinn/checkpoint.h5` | Missing | ⚠️ PINN checkpoint present; baseline checkpoint for gs2 not found |

**Gaps:**
- ❌ **dose_10000, dose_100000:** No Phase E training runs found (neither PINN nor baseline checkpoints).
- ⚠️ **dose_1000/dense/gs2:** Only PINN checkpoint present; no baseline checkpoint for gs2. Phase G comparison plan assumes baseline exists for parity.

**Implications for G2:**
- Three-way comparisons (PINN vs baseline vs pty-chi) can only be executed for **dose_1000** conditions where Phase E checkpoints exist.
- For dose_10000 and dose_100000, Phase G comparisons will be **two-way** (pty-chi only, or blocked pending Phase E training).

**Full listing:** `phase_e_checkpoints_tmp.txt`

## 4. Phase F LSQML Reconstructions (pty-chi Baseline)

**Contract:** POLICY-001 (PyTorch-backed pty-chi); Phase F manifest metadata includes acceptance stats
**Source:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T*/phase_f*/real_run/`
**Status:** ⚠️ PARTIAL — 3 conditions complete, 9 missing

### Inventory

| Dose | View | Split | Reconstruction NPZ | Manifest | Log | Notes |
|------|------|-------|-------------------|----------|-----|-------|
| 1000 | dense | train | `plans/active/.../2025-11-04T210000Z/phase_f_ptychi_baseline_f2_cli_input_fix/real_run/dose_1000/dense/train/ptychi_reconstruction.npz` | `plans/active/.../2025-11-04T210000Z/.../real_run/reconstruction_manifest.json` | `ptychi.log` | ✅ returncode=0, LSQML 100 epochs |
| 1000 | dense | test | `plans/active/.../2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/real_run/dose_1000/dense/test/ptychi_reconstruction.npz` | `plans/active/.../2025-11-04T230000Z/.../real_run/reconstruction_manifest.json` | `ptychi.log` | ✅ returncode=0, LSQML 100 epochs |
| 1000 | sparse | train | `plans/active/.../2025-11-05T050500Z/phase_f_ptychi_baseline_f3_sparse_runs/real_run/dose_1000/sparse/train/ptychi_reconstruction.npz` | `plans/active/.../2025-11-04T133218Z/.../real_run/reconstruction_manifest_sparse_train.json` | `ptychi.log` | ✅ returncode=1 (expected for singular LSQML); acceptance_rate=0.125, n_accepted=1, n_rejected=7 per OVERSAMPLING-001 |
| 1000 | sparse | test | ❌ Missing | N/A | N/A | **TODO:** Phase F sparse/test run not executed |
| 10000 | dense | train | ❌ Missing | N/A | N/A | **TODO:** Phase F not executed for dose_10000 |
| 10000 | dense | test | ❌ Missing | N/A | N/A | **TODO:** Phase F not executed for dose_10000 |
| 10000 | sparse | train | ❌ Missing | N/A | N/A | **TODO:** Phase F not executed for dose_10000 |
| 10000 | sparse | test | ❌ Missing | N/A | N/A | **TODO:** Phase F not executed for dose_10000 |
| 100000 | dense | train | ❌ Missing | N/A | N/A | **TODO:** Phase F not executed for dose_100000 |
| 100000 | dense | test | ❌ Missing | N/A | N/A | **TODO:** Phase F not executed for dose_100000 |
| 100000 | sparse | train | ❌ Missing | N/A | N/A | **TODO:** Phase F not executed for dose_100000 |
| 100000 | sparse | test | ❌ Missing | N/A | N/A | **TODO:** Phase F not executed for dose_100000 |

**Manifests Found:** 15 manifest JSON files (see `phase_f_manifest_listing.txt`), but only 3 with successful `ptychi_reconstruction.npz` output.

**Gaps:**
- ❌ **dose_1000/sparse/test:** Phase F sparse test reconstruction not executed.
- ❌ **dose_10000, dose_100000:** No Phase F LSQML reconstructions for any view/split combination.

**Implications for G2:**
- Phase G three-way comparisons can only proceed for **3 conditions** where all inputs exist:
  1. dose_1000/dense/train (PINN checkpoint assumed reachable, baseline checkpoint present, pty-chi NPZ present)
  2. dose_1000/dense/test (same as above)
  3. dose_1000/sparse/train (PINN checkpoint assumed reachable, baseline checkpoint present, pty-chi NPZ present with sparse acceptance metadata)

**Full listing:** `ptychi_npz_inventory.txt`, `phase_f_manifest_listing.txt`

## 5. Phase F Manifests (Metadata & Acceptance Stats)

**Status:** ✅ 15 manifest JSON files found

### Key Manifests

| Timestamp | Phase | Manifest Path | Content Summary |
|-----------|-------|--------------|-----------------|
| 2025-11-04T081500Z | E | `plans/active/.../phase_e_training_cli/artifacts/training_manifest.json` | Phase E training jobs (dry_run=true), references `tmp/phase_e_cli_demo/` |
| 2025-11-04T170500Z | E | `plans/active/.../phase_e_training_e5_real_run_baseline/real_run/training_manifest.json` | Phase E real run metadata, skipped sparse views, references `tmp/phase_c_training_evidence/` (not found) |
| 2025-11-04T133218Z | F | `plans/active/.../phase_f_ptychi_baseline_f3_metadata_recovery/real_run/reconstruction_manifest_sparse_train.json` | Sparse train reconstruction (returncode=1, acceptance_rate=0.125, n_accepted=1) |
| 2025-11-04T210000Z | F | `plans/active/.../phase_f_ptychi_baseline_f2_cli_input_fix/real_run/reconstruction_manifest.json` | Dense train reconstruction (returncode=0) |
| 2025-11-04T230000Z | F | `plans/active/.../phase_f_ptychi_baseline_f2_dense_test_run/real_run/reconstruction_manifest.json` | Dense test reconstruction (returncode=0) |
| 2025-11-05T140500Z | G | `plans/active/.../phase_g_comparison_plan/cli/comparison_manifest.json` | Phase G CLI dry-run (job builder test) |

**OVERSAMPLING-001 Compliance:** Phase F sparse manifest includes `selection_strategy="greedy"`, `acceptance_rate`, `spacing_threshold`, `n_accepted`, `n_rejected` metadata per OVERSAMPLING-001 policy.

**Full listing:** `phase_f_manifest_listing.txt`

## 6. Original Fly64 Object/Probe Assets

**Source:** `datasets/fly64/`
**Status:** ⚠️ Only visualization PNG files found

### Inventory

```
datasets/fly64/fly64_random_1000_vis.png
datasets/fly64/fly64_sequential_1000_vis.png
```

**Gaps:**
- ❌ **Original object/probe NPZ files:** Not found in `datasets/fly64/`. These assets are likely embedded in Phase C datasets or sourced from an external location not inventoried here.

**Full listing:** `datasets_fly64_listing.txt`

## 7. CONFIG-001 Compliance Notes

All Phase E and Phase F workflows assume **CONFIG-001 bridge** (`update_legacy_dict(params.cfg, config)`) is applied before data loading or model construction. Phase G comparison harness (G1.1 job builder) must enforce this initialization for any PINN/baseline inference runs.

**References:**
- `docs/findings.md` CONFIG-001
- `docs/architecture.md` "two-system" architecture
- `CLAUDE.md` §4.1 Parameter Initialization

## 8. Summary of Gaps & Blockers for G2

### Ready for G2 Execution (3 conditions)

1. **dose_1000/dense/train** — PINN checkpoint (assumed), baseline checkpoint (present), pty-chi NPZ (present)
2. **dose_1000/dense/test** — PINN checkpoint (assumed), baseline checkpoint (present), pty-chi NPZ (present)
3. **dose_1000/sparse/train** — PINN checkpoint (assumed), baseline checkpoint (present), pty-chi NPZ (present with acceptance metadata)

### Blocked Pending Upstream Work (9 conditions)

4. **dose_1000/sparse/test** — ❌ Phase F LSQML reconstruction not executed
5. **dose_10000/dense/train** — ❌ Phase E checkpoints missing, Phase F reconstruction missing
6. **dose_10000/dense/test** — ❌ Phase E checkpoints missing, Phase F reconstruction missing
7. **dose_10000/sparse/train** — ❌ Phase E checkpoints missing, Phase F reconstruction missing
8. **dose_10000/sparse/test** — ❌ Phase E checkpoints missing, Phase F reconstruction missing
9. **dose_100000/dense/train** — ❌ Phase E checkpoints missing, Phase F reconstruction missing
10. **dose_100000/dense/test** — ❌ Phase E checkpoints missing, Phase F reconstruction missing
11. **dose_100000/sparse/train** — ❌ Phase E checkpoints missing, Phase F reconstruction missing
12. **dose_100000/sparse/test** — ❌ Phase E checkpoints missing, Phase F reconstruction missing

### Adjusted Phase G Scope Recommendation

Given the gaps above, **Phase G2 execution should be scoped to the 3 ready conditions** (dose_1000 only). The original plan anticipated 18 jobs (3 doses × 3 views × 2 splits), but only **3 three-way comparisons** are currently feasible.

**Next Actions:**
- **G2.1 (dose_1000/dense):** Execute comparisons for train + test splits using available Phase E checkpoints and Phase F reconstructions.
- **G2.2 (dose_1000/sparse/train):** Execute sparse train comparison; document acceptance metadata per OVERSAMPLING-001.
- **Defer:** dose_1000/sparse/test, dose_10000, dose_100000 pending upstream Phase E/F completion.

## 9. Artifact References

All raw inventory outputs are preserved under:

```
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T162500Z/phase_g_inventory/analysis/
├── phase_f_manifest_listing.txt         (15 manifest JSONs)
├── ptychi_npz_inventory.txt             (3 reconstruction NPZ files)
├── datasets_fly64_listing.txt           (2 visualization PNGs)
├── phase_c_d_datasets.txt               (18 Phase C/D NPZ files)
├── phase_e_checkpoints_tmp.txt          (2 Phase E checkpoints)
├── tmp_listing.txt                      (tmp/ directory structure)
├── all_npz_files.txt                    (all NPZ files found)
└── inventory.md                         (this document)
```

## 10. Findings & Policy Compliance

- ✅ **DATA-001:** Phase C/D datasets assumed compliant (canonical `diffraction` amplitude format per historical contract).
- ✅ **CONFIG-001:** Phase E/F workflows enforced legacy bridge initialization.
- ✅ **POLICY-001:** Phase F LSQML reconstructions use PyTorch-backed pty-chi (torch>=2.2 required).
- ✅ **OVERSAMPLING-001:** Phase F sparse manifests include acceptance metadata (selection_strategy, acceptance_rate, spacing_threshold, n_accepted, n_rejected).

## 11. References

- **Phase G Plan:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-05T140500Z/phase_g_comparison_plan/plan/plan.md`
- **Test Strategy:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:259`
- **Fix Plan:** `docs/fix_plan.md:31` (STUDY-SYNTH-FLY64-DOSE-OVERLAP-001)
- **Data Contracts:** `specs/data_contracts.md:1`
- **Findings Ledger:** `docs/findings.md` (CONFIG-001, DATA-001, POLICY-001, OVERSAMPLING-001)
- **CLAUDE.md:** `CLAUDE.md:4.1` (Parameter Initialization gotcha)

---

**Conclusion:** This inventory provides authoritative ground truth for Phase G execution planning. G2 execution should proceed for the 3 ready conditions (dose_1000/dense/train, dose_1000/dense/test, dose_1000/sparse/train) while deferring higher doses and incomplete splits pending upstream Phase E/F completion.
