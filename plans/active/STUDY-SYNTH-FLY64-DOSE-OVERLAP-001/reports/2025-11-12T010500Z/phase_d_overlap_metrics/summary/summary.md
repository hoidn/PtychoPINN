# Phase D Overlap Metrics Hub — Turn Summary

**Loop timestamp:** 2025-11-11T161300Z
**Agent:** Ralph (implementation)
**Focus:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 — Phase D overlap metrics CLI evidence capture
**Status:** COMPLETE — gs1/gs2 CLI runs executed with Metric 1/2/3 JSON bundles

---

## What Was Done

Executed Phase D overlap metrics CLI twice (gs1 and gs2) against `data/phase_c/dose_1000` to capture real metrics artifacts after commit `d94f24f7` landed the API/CLI/tests.

1. **Pytest selector rerun:** `test_overlap_metrics_bundle` passed in 1.54s → `green/pytest_phase_d_overlap_bundle_rerun.log`
2. **GS1 CLI:** `gridsize=1, s_img=1.0, n_groups=512` → `metrics/gs1_s100_n512/*.json` + `cli/phase_d_overlap_gs1.log`
3. **GS2 CLI:** `gridsize=2, s_img=0.8, n_groups=512` → `metrics/gs2_s080_n512/*.json` + `cli/phase_d_overlap_gs2.log`
4. **Artifact inventory:** Updated with commands + metrics summary → `analysis/artifact_inventory.txt`

---

## Metrics Summary (dose_1000)

### GS1 (gridsize=1, s_img=1.0, n_groups=512)
- **Metric 1 (group-based):** N/A (gridsize=1)
- **Metric 2 (image-based):** Train 0.8908, Test 0.8917
- **Metric 3 (group↔group):** Train 0.2662, Test 0.2658
- Images: 5088 train (100%), 5216 test (100%)
- Groups: 5088 train, 5216 test (one image per group)

### GS2 (gridsize=2, s_img=0.8, n_groups=512)
- **Metric 1 (group-based):** Train 0.8909, Test 0.8942
- **Metric 2 (image-based):** Train 0.8777, Test 0.8794
- **Metric 3 (group↔group):** Train 0.2706, Test 0.2673
- Images: 4070/5088 train (80%), 4173/5216 test (80%)
- Groups: 512 train, 512 test (4 images per group with KNN)

**Key observations:**
- Metric 2 (image-level overlap) consistently high (~0.88-0.89) across both configurations
- Metric 3 (group↔group COM-based) moderate (~0.27), indicating spatial clustering
- GS2 Metric 1 matches Metric 2 within-group (validates disc overlap calculation)
- All metrics comply with `specs/overlap_metrics.md` schema

---

## Evidence Files

**Tests:**
- `green/pytest_phase_d_overlap_bundle_rerun.log` (1 passed)

**CLI Logs:**
- `cli/phase_d_overlap_gs1.log`
- `cli/phase_d_overlap_gs2.log`

**Metrics Artifacts:**
- `metrics/gs1_s100_n512/{train,test,metrics_bundle}.json`
- `metrics/gs2_s080_n512/{train,test,metrics_bundle}.json`

**Inventory:**
- `analysis/artifact_inventory.txt` (updated with commands + summary)

---

## Unblocked Work

- **Phase E:** Training CLI can now align manifests with Phase D outputs
- **Phase G:** Dense orchestrator can validate overlap-driven filtering and resume SSIM grid generation

---

## References

- Spec: `specs/overlap_metrics.md` — Metric 1/2/3 definitions, disc overlap, CLI contract
- Guide: `docs/GRIDSIZE_N_GROUPS_GUIDE.md` — unified `n_groups` semantics
- Initiative Plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`
- Fix Plan Ledger: `docs/fix_plan.md`
- Selector: `tests/study/test_dose_overlap_overlap.py::test_overlap_metrics_bundle`

---

### Turn Summary
Reran pytest selector and executed overlap CLI for gs1 (`s_img=1.0,n_groups=512`) and gs2 (`s_img=0.8,n_groups=512`) against Phase C dose_1000 data.
Generated `train_metrics.json`, `test_metrics.json`, and `metrics_bundle.json` for both configurations with Metric 1/2/3 values matching spec requirements (Metric 1 omitted for gs1 as expected).
Next: Phase E/G can proceed using these metrics artifacts for training manifest alignment and dense pipeline verification.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_d_overlap_metrics/ (pytest_phase_d_overlap_bundle_rerun.log, phase_d_overlap_gs{1,2}.log, metrics JSONs, artifact_inventory.txt)
