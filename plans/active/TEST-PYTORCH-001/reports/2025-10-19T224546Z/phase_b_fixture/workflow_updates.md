# Phase B3.C Workflow Documentation Updates

## Loop Metadata
- **Date**: 2025-10-19T224546Z
- **Phase**: TEST-PYTORCH-001 Phase B3.C (Documentation Close-out)
- **Engineer**: Ralph
- **Directive**: input.md B3.C — Refresh docs/workflows/pytorch.md §11 with minimal fixture runtime evidence

---

## Documentation Changes

### Target File: `docs/workflows/pytorch.md` Section 11

Updated three subsections to reflect Phase B3 minimal fixture integration outcomes:

#### 1. Runtime Performance (Lines 258-274)

**Old State:**
- Reported Phase D1 baseline: 35.9s ± 0.5s (canonical dataset, 1087 positions)
- CI Budget: ≤90s (2.5× baseline)
- Warning Threshold: 60s (1.7× baseline)

**New State:**
- **Current Performance (Phase B3):**
  - Smoke Test Runtime: 3.82s (7 fixture validation tests)
  - Integration Test Runtime: 14.53s (full train→save→load→infer cycle)
  - Test Dataset: `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` (64 positions, 25 KB)
  - CI Budget: ≤90s (integration runs at 16% of budget)
  - Warning Threshold: 45s (3.1× current runtime)
- **Historical Baselines:**
  - Phase D1: 35.9s ± 0.5s
  - Phase B1: 21.91s (n_groups=64 override)
  - Phase B3 Improvement: 33.7% faster vs Phase B1

**Rationale:**
- Reflects evidence from `plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/summary.md` (14.53s integration runtime, 3.82s smoke runtime)
- Preserves historical baseline context while highlighting current performance
- Adjusts warning threshold from 60s → 45s to reflect tighter performance envelope (3× current runtime vs 1.7× old baseline)
- Documents fixture path and characteristics (64 positions, 25 KB size)

#### 2. Data Contract Compliance (Lines 291-296)

**Old State:**
- Generic "Dataset: Canonical format per specs/data_contracts.md §1"

**New State:**
- **Test Dataset:** Minimal fixture at `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` (64 scan positions, stratified sampling, canonical (N,H,W) format, float32/complex64 dtypes per DATA-001)
- **Fixture Generation:** Reproducible command with SHA256 checksum (6c2fbea0dcadd950385a54383e6f5f731282156d19ca4634a5a19ba3d1a5899c)

**Rationale:**
- Documents specific fixture used in regression (not generic canonical dataset)
- Provides regeneration command per Phase B2.D fixture_notes.md
- Includes SHA256 checksum for provenance/verification
- Notes stratified sampling strategy (94.8% / 96.8% coordinate coverage)

#### 3. CI Integration Notes (Lines 298-305)

**Old State:**
- Recommended Timeout: 120s (3.3× baseline)
- Generic retry/marker guidance

**New State:**
- **Recommended Timeout:** 90s (6.2× current runtime, conservative buffer)
- **CPU-Only Enforcement:** Explicit `CUDA_VISIBLE_DEVICES=""` guidance
- **Reference:** Updated to include Phase B3 summary.md artifact link

**Rationale:**
- Reduces timeout from 120s → 90s (still 6.2× current runtime, very conservative)
- Clarifies CPU-only enforcement mechanism (CUDA_VISIBLE_DEVICES environment variable)
- Adds Phase B3 artifact reference for traceability

---

## Changes Summary

| Section | Lines | Old Value | New Value | Evidence Source |
| :--- | :--- | :--- | :--- | :--- |
| Runtime Performance | 260-270 | "Baseline: 35.9s" | "Integration: 14.53s, Smoke: 3.82s" | `summary.md` (2025-10-19T233500Z) |
| Runtime Performance | 264 | "CI Budget: ≤90s (2.5× baseline)" | "CI Budget: ≤90s (16% utilization)" | Phase B3 14.53s / 90s = 16.1% |
| Runtime Performance | 265 | "Warning: 60s (1.7× baseline)" | "Warning: 45s (3.1× current)" | 3.1 × 14.53s = 45.04s |
| Data Contract | 295-296 | Generic dataset note | Fixture path + SHA256 checksum | `fixture_notes.md` (2025-10-19T225900Z) |
| CI Integration | 300 | "Timeout: 120s (3.3× baseline)" | "Timeout: 90s (6.2× current)" | 6.2 × 14.53s = 90.09s |
| CI Integration | 305 | Generic implementation.md ref | Added summary.md Phase B3 ref | Artifact discipline per plan.md |

---

## Validation

### ASCII Formatting Preserved
- All existing `<doc-ref>` tags intact (POLICY-001, FORMAT-001, DATA-001 references)
- Markdown structure preserved (headers, lists, code blocks)
- Line wrapping consistent with existing style

### No Regressions Introduced
- Did not alter §§1-10 (configuration, training, inference guidance)
- Did not modify §12 (backend selection — separate initiative)
- Only touched §11 subsections as directed by input.md

### Evidence Traceability
- All new numeric values cite Phase B3 artifacts (summary.md, fixture_notes.md)
- SHA256 checksum matches `tests/fixtures/pytorch_integration/minimal_dataset_v1.json` metadata
- Runtime values match `pytest_integration_fixture.log` timestamps

---

## Conflicts/Pitfalls Avoided

✅ **Did not regress CUDA_VISIBLE_DEVICES requirement** — Added explicit CI guidance (line 303)
✅ **Preserved canonical dataset references elsewhere** — Only swapped regression fixture narrative in §11
✅ **Did not introduce new runtime numbers without evidence** — All values cite 2025-10-19T233500Z logs
✅ **Did not delete existing artifact directories** — No changes to `reports/` hierarchy
✅ **Maintained ASCII formatting** — No emoji, no formatting corruption
✅ **Avoided touching data/memmap/meta.json** — Per input.md pitfall guidance

---

## Next Actions (B3.C Completion Checklist)

- [x] Edit `docs/workflows/pytorch.md` Section 11 with new dataset/runtime info
- [x] Capture narrative of changes in `workflow_updates.md` (this file)
- [ ] Update `implementation.md` B3 row (mark `[x]`, add artifact links)
- [ ] Append `docs/fix_plan.md` Attempt with B3.C completion summary
- [ ] Author loop `summary.md` for 2025-10-19T224546Z artifact hub

**Exit Criteria:**
- Workflow guide §11 reflects minimal fixture (14.53s runtime, 3.82s smoke, fixture path)
- CI thresholds updated (<45s warning, 90s cap, 60s investigation threshold documented)
- Historical baselines preserved for context
- Artifact hub complete with workflow_updates.md + summary.md

---

**Status:** ✅ Workflow documentation refresh COMPLETE — awaiting plan/ledger updates (B3.C steps 2-4).
