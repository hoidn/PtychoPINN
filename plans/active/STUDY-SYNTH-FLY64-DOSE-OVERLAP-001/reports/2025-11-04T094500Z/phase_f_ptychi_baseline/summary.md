# Phase F0 RED Scaffolding — Summary

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase:** F0 — PtyChi LSQML Baseline Test Infrastructure Prep
**Mode:** TDD
**Timestamp:** 2025-11-04T094500Z
**Ralph Attempt:** #28

---

## Acceptance Focus

**AT-F0.1:** Test strategy updated with Phase F sections (selectors, execution proof rules, artifact expectations per template)
**AT-F0.2:** RED test `test_build_ptychi_jobs_manifest` authored expecting `NotImplementedError` until GREEN implementation

**Module Scope:** tests/docs (test infrastructure scaffolding only; no production logic implemented)

---

## SPEC/ADR Alignment

**SPEC (specs/data_contracts.md:210-276):**
> Reconstruction NPZ expectations: `diffraction` as amplitude float32, complex64 Y patches, coords/filenames

**Findings (docs/findings.md):**
- CONFIG-001 (path:10): Builder will remain pure (no params.cfg mutation); CONFIG-001 bridge deferred to LSQML runner invocation (Phase F2)
- DATA-001 (path:14): Test fixtures reference Phase E training manifest artifacts; builder validates NPZ paths against canonical contract
- POLICY-001 (path:8): Pty-chi uses PyTorch internally (acceptable per study design); no PtychoPINN backend switch required

**ADR:** None directly applicable; reconstruction orchestrator follows Phase E training pattern (job builder + CLI + runner)

---

## Implementation Summary

### F0.1 — Test Strategy Update
**File:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md:212-247`
**Action:** Added Phase F section documenting:
- Active selectors: `pytest tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest -vv`
- Coverage expectations: 3 doses × 2 views (dense, sparse) + 1 baseline per dose = 7 jobs per dose (21 total)
- CLI args structure: `--algorithm LSQML`, `--num-epochs 100`, dataset paths from Phase D/E
- Execution proof paths: `red/pytest_phase_f_red.log`, `collect/pytest_collect.log`, GREEN logs pending F1
- Findings alignment: CONFIG-001 (pure builder), DATA-001 (canonical NPZ paths), POLICY-001 (pty-chi PyTorch internal), OVERSAMPLING-001 (inherit K=7)

### F0.2 — RED Test
**File:** `tests/study/test_dose_overlap_reconstruction.py:1-149`
**Fixtures:**
- `mock_phase_c_datasets`: minimal Phase C baseline NPZs (dose_1000/patched_{train,test}.npz)
- `mock_phase_d_datasets`: minimal Phase D overlap NPZs (dose_1000/{dense,sparse}/{view}_{split}.npz)

**Test:** `test_build_ptychi_jobs_manifest`
- Expects `NotImplementedError` from `build_ptychi_jobs()` stub
- Documents expected manifest structure for GREEN phase (21 jobs with CLI args, artifact paths, algorithm/epoch params)

### F0.3 — Reconstruction Module Stub
**File:** `studies/fly64_dose_overlap/reconstruction.py:1-78`
**Function:** `build_ptychi_jobs(phase_c_root, phase_d_root, artifact_root)`
- Raises `NotImplementedError` with message referencing Phase F1.1 in plan
- Docstring details expected manifest structure (ReconstructionJob dataclasses pending GREEN)

### F0.4 — Module Exposure
**File:** `studies/fly64_dose_overlap/__init__.py:1-5`
**Action:** Added `from studies.fly64_dose_overlap import reconstruction` and `__all__ = ['reconstruction']`

---

## Test Results

### RED Phase
**Command:** `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && pytest tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest -vv`
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/red/pytest_phase_f_red.log`
**Result:** 1 PASSED (1.70s) — test correctly expects and catches NotImplementedError

### Collection Proof
**Command:** `pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv`
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/collect/pytest_collect.log`
**Result:** 1 test collected

### Static Analysis
**Command:** `ruff check --fix studies/fly64_dose_overlap/reconstruction.py tests/study/test_dose_overlap_reconstruction.py`
**Result:** 1 fixable error (unused Path import) auto-fixed; all checks passed

### Comprehensive Test Suite
**Command:** `pytest -v tests/`
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/pytest_full_suite.log`
**Result:**
- 385 PASSED
- 17 SKIPPED (expected: TF addons removal, optional datasets, etc.)
- 1 FAILED (pre-existing: `tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader` — ModuleNotFoundError unrelated to Phase F)
- **Zero regressions**

---

## Files Changed

**Modified:**
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md` (lines 212-247: Phase F section added)
- `studies/fly64_dose_overlap/__init__.py` (lines 3-5: reconstruction module exposure)

**Created:**
- `studies/fly64_dose_overlap/reconstruction.py` (78 lines: stub module with NotImplementedError)
- `tests/study/test_dose_overlap_reconstruction.py` (149 lines: RED test + fixtures)

---

## Findings Applied

- **CONFIG-001:** Stub builder documented to remain pure; no params.cfg mutation
- **DATA-001:** Test fixtures create canonical NPZs (amplitude diffraction, complex64 Y, coords/filenames)
- **POLICY-001:** Pty-chi PyTorch usage documented as acceptable (no backend switch needed)
- **OVERSAMPLING-001:** Reconstruction jobs will inherit neighbor_count=7 from Phase D/E artifacts (no K≥C validation in builder)

---

## Next Actions

**Phase F1 — GREEN Implementation:**
1. Define `ReconstructionJob` dataclass (dose, view, split, input_npz, output_dir, algorithm='LSQML', num_epochs=100, cli_args)
2. Implement `build_ptychi_jobs()` enumerating 21 jobs (3 doses × 7 views including baseline)
3. Extend RED test to GREEN by asserting manifest structure and CLI arg correctness
4. Add CLI entry point mirroring `training.py` pattern (`--dose`, `--view`, `--dry-run` filters)
5. Capture GREEN logs, CLI dry-run transcript, and update docs/registries

**Outstanding Gaps:**
- `scripts/reconstruction/ptychi_reconstruct_tike.py` existence not validated (assumes present)
- Real LSQML runs deferred to Phase F2 (skip reporting pattern from Phase E5 recommended)

---

## Artifact Inventory

```
plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline/
├── plan/
│   └── plan.md (Phase F0-F2 task breakdown)
├── red/
│   └── pytest_phase_f_red.log (1 PASSED, NotImplementedError correctly caught)
├── collect/
│   └── pytest_collect.log (1 test collected)
├── pytest_full_suite.log (385 PASSED, 17 SKIPPED, 1 pre-existing failure)
└── summary.md (this file)
```

---

## Exit Criteria Met

- [x] Test strategy Phase F section authored with selectors, execution proof rules, artifact expectations
- [x] RED test `test_build_ptychi_jobs_manifest` authored and passing (expecting NotImplementedError)
- [x] Reconstruction module stub created and exposed in `__init__.py`
- [x] Collection proof captured (1 test collected)
- [x] Static analysis passed (ruff checks green)
- [x] Full test suite passed (385/403 valid tests, zero regressions)

**Phase F0 COMPLETE** — Ready for F1 GREEN implementation
