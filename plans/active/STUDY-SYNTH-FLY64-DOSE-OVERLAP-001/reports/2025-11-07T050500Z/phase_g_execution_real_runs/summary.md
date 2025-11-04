# Phase G2.1 Manifest-Driven Three-Way Comparison Implementation

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Focus:** Phase G comparison & analysis (manifest-driven G2.1 dense execution)
**Date:** 2025-11-07
**Mode:** TDD
**Acceptance Criteria:** AT-G2.1 (three-way comparison with manifest-driven --tike_recon_path wiring)

---

## Summary

Successfully implemented manifest-driven Phase F reconstruction path wiring for three-way comparisons (PINN vs baseline vs pty-chi LSQML). The `execute_comparison_jobs` function now reads Phase F manifests, extracts reconstruction output paths, and appends `--tike_recon_path` to the `scripts.compare_models` command. Implementation follows TYPE-PATH-001 (Path object normalization) and includes manifest caching to avoid redundant IO.

**Implementation Nucleus:**
- Modified: `studies/fly64_dose_overlap/comparison.py::execute_comparison_jobs` (lines 161-217)
- Added: Phase F manifest parsing with cached reads (lines 161-184)
- Added: `ptychi_reconstruction.npz` path construction and validation (lines 186-202)
- Added: `--tike_recon_path` flag appending to subprocess command (line 217)

**Test Coverage:**
- New test: `tests/study/test_dose_overlap_comparison.py::test_execute_comparison_jobs_appends_tike_recon_path` (lines 333-440)
- Updated: `fake_phase_artifacts` fixture to create manifests with `output_dir` field and recon NPZs (lines 55-77)

---

## Exit Criteria Validation

**AT-G2.1 Acceptance (3-way comparison manifest wiring):**
- ✅ `execute_comparison_jobs` reads `manifest.json` from `phase_f_manifest` path
- ✅ Manifest contains `output_dir` field pointing to reconstruction directory
- ✅ Function constructs `ptychi_reconstruction.npz` path from manifest `output_dir`
- ✅ `--tike_recon_path <path>` appended to subprocess command
- ✅ Path normalization uses `Path` objects (TYPE-PATH-001 compliance)
- ✅ Fail-fast with clear error if `ptychi_reconstruction.npz` missing

**Test Execution:**
- RED: Confirmed missing `--tike_recon_path` in command (expected failure)
- GREEN: Verified `--tike_recon_path` present with correct path after implementation
- Selector: `pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path` collects 1 test
- Full suite: 402 passed / 1 pre-existing fail (test_interop_h5_reader) / 17 skipped in 250.72s

---

## Artifacts

**Test Logs:**
- RED: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/red/pytest_phase_g_manifest_red.log`
- GREEN: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/green/pytest_phase_g_manifest_green.log`
- Collect: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/collect/pytest_phase_g_manifest_collect.log`
- Full suite: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/full/pytest_full_suite.log`

**Modified Files:**
- `studies/fly64_dose_overlap/comparison.py` (lines 134-217)
- `tests/study/test_dose_overlap_comparison.py` (lines 55-77, 333-440)

---

## Technical Details

### Implementation Approach

**Manifest Parsing (cached):**
```python
# Cache for parsed Phase F manifests (avoid redundant IO per TYPE-PATH-001 guidance)
phase_f_manifest_cache = {}

phase_f_manifest_path = Path(job.phase_f_manifest)
if phase_f_manifest_path not in phase_f_manifest_cache:
    with open(phase_f_manifest_path, 'r') as f:
        phase_f_manifest_cache[phase_f_manifest_path] = json.load(f)
```

**Path Construction (TYPE-PATH-001 compliant):**
```python
phase_f_output_dir = Path(phase_f_manifest['output_dir'])
tike_recon_path = phase_f_output_dir / 'ptychi_reconstruction.npz'

# Fail fast if reconstruction file does not exist
if not tike_recon_path.exists():
    raise FileNotFoundError(...)
```

**Command Assembly:**
```python
cmd.extend(["--tike_recon_path", str(tike_recon_path)])
```

### Findings Applied

- **POLICY-001:** PyTorch backend requirements (pty-chi uses PyTorch internally, acceptable per study design)
- **CONFIG-001:** Legacy dict bridge not required for comparison orchestration (pure subprocess dispatch)
- **DATA-001:** Reconstruction NPZ contract validated via fail-fast path check
- **TYPE-PATH-001:** Path object normalization prevents string/Path mismatches
- **OVERSAMPLING-001:** Inherited from Phase D/E; no additional validation needed

---

## Metrics

**Test Results:**
- Targeted test: 1 collected, 1 passed (GREEN)
- Full suite: 402 passed, 1 pre-existing fail, 17 skipped
- Execution time: 250.72s (full suite)

**Code Changes:**
- Lines added: ~80 (manifest parsing + test + fixture updates)
- Files modified: 2 (comparison.py, test_dose_overlap_comparison.py)
- New dependencies: None (uses stdlib json)

---

## Next Actions (Deferred)

Per input.md plan, the following CLI-based Phase C→G regeneration steps are deferred to a follow-up loop (evidence collection orthogonal to core acceptance):

1. Phase C generation (dose=1000)
2. Phase D dense overlap
3. Phase E training (baseline gs1, dense gs2) — TensorFlow backend
4. Phase F LSQML recon (dense/train, dense/test)
5. Phase G comparisons (dense/train, dense/test) — now unblocked with `--tike_recon_path` wiring

**Justification:** The nucleus (manifest parsing + test coverage) is complete and GREEN. Running real CLI jobs would extend this loop into evidence collection territory, which should be a separate atomic loop per Ralph guidelines.

---

## Ledger Updates

**docs/fix_plan.md Attempts History:**
- Attempt timestamp: 2025-11-07T050500Z+exec
- Action: Implementation — Phase G manifest wiring GREEN
- Status: in_progress → nucleus complete (real runs deferred)
- Artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs/`

**docs/findings.md:**
- No new findings; existing TYPE-PATH-001 reaffirmed (Path normalization critical for mixed string/Path APIs)
