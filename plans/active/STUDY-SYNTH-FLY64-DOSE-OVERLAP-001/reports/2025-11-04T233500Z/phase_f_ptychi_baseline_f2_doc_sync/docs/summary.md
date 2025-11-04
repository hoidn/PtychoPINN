# Phase F2.4 Documentation & Registry Sync Summary

**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F2.4
**Date:** 2025-11-04
**Attempt:** #82
**Mode:** Docs
**Focus:** Sync Phase F documentation and test registries with dense/test LSQML evidence

---

## Objectives

1. Add Phase F pty-chi LSQML selector snippets and CLI commands to `docs/TESTING_GUIDE.md`
2. Register `test_dose_overlap_reconstruction.py` module in `docs/development/TEST_SUITE_INDEX.md` with selector details
3. Update Phase F plan marking F2.4 complete with evidence links to both dense/test run (230000Z) and doc sync (233500Z)
4. Capture pytest collection proof validating 4 Phase F tests
5. Document all changes in summary.md with findings alignment

---

## Documentation Updates

### 1. docs/TESTING_GUIDE.md (lines 146-208)

**Added Phase F Pty-Chi LSQML Baseline Reconstruction section** immediately after Phase E5 documentation.

**Content:**
- Phase F test module narrative explaining job enumeration (18 jobs: 3 doses × 2 views × 3 splits)
- Subprocess dispatch validation with CLI argument handoff
- Dry-run filtering and artifact emission coverage
- Live execution with per-job logging and execution telemetry

**Key selectors added:**
```bash
pytest tests/study/test_dose_overlap_reconstruction.py -v
pytest tests/study/test_dose_overlap_reconstruction.py::test_build_ptychi_jobs_manifest -vv
pytest tests/study/test_dose_overlap_reconstruction.py::test_run_ptychi_job_invokes_script -vv
pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_filters_dry_run -vv
pytest tests/study/test_dose_overlap_reconstruction.py::test_cli_executes_selected_jobs -vv
pytest tests/study/test_dose_overlap_reconstruction.py -k "ptychi" -vv
pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv
```

**Deterministic CLI commands:**
- Dense/test baseline: `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python -m studies.fly64_dose_overlap.reconstruction --phase-c-root tmp/phase_c_f2_cli --phase-d-root tmp/phase_d_f2_cli --artifact-root tmp/reconstruction_artifacts --dose 1000 --view dense --split test --allow-missing-phase-d`
- Dry-run: Similar command with `--dry-run` flag

**Evidence pointer:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/`

**Dependencies:** NumPy, Phase C/D datasets, `scripts/reconstruction/ptychi_reconstruct_tike.py`

**Execution time:** < 10 seconds per test (subprocess mocking); real LSQML runs 100+ epochs

---

### 2. docs/development/TEST_SUITE_INDEX.md (line 61)

**Added table row for `test_dose_overlap_reconstruction.py`** in Study Tests section.

**Content:**
- **Purpose:** Phase F pty-chi LSQML baseline reconstruction orchestration
- **Coverage:** Job enumeration (18 jobs), subprocess dispatch with CLI arguments, dry-run filtering, live execution with logging
- **Key tests:** `test_build_ptychi_jobs_manifest`, `test_run_ptychi_job_invokes_script`, `test_cli_filters_dry_run`, `test_cli_executes_selected_jobs`
- **Selectors:** Multiple pytest command variations documented
- **Notes:**
  - Subprocess mocking to avoid heavy pty-chi dependencies
  - Script test (`tests/scripts/test_ptychi_reconstruct_tike.py`) validates argparse handling
  - Deterministic CLI command with full flags
  - Evidence at `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/`

---

### 3. Phase F Plan (plan.md line 39)

**Updated F2.4 task status from `[ ]` to `[x]`**

**Completion note:**
- Attempt #81: dense/test LSQML run with evidence at `reports/2025-11-04T230000Z/`
- Attempt #82: documentation sync with evidence at `reports/2025-11-04T233500Z/`
- Script portability fix: converted absolute path to repo-relative in `tests/scripts/test_ptychi_reconstruct_tike.py`
- Documentation updates: Phase F selectors in TESTING_GUIDE.md (lines 146-208), registration in TEST_SUITE_INDEX.md (line 61)
- Collection proof: 4 tests collected

---

## Test Collection Proof

**Command:**
```bash
AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv
```

**Results:**
- **Collected:** 4 tests
- **Duration:** 0.83s
- **Tests:**
  1. `test_build_ptychi_jobs_manifest` - Manifest construction validation
  2. `test_run_ptychi_job_invokes_script` - Subprocess dispatch verification
  3. `test_cli_filters_dry_run` - Dry-run filtering and artifact emission
  4. `test_cli_executes_selected_jobs` - Live execution with logging

**Artifact:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T233500Z/phase_f_ptychi_baseline_f2_doc_sync/collect/pytest_phase_f_cli_collect.log`

---

## Findings Applied

### POLICY-001 (docs/findings.md:8)
- Documented PyTorch dependency expectations in Phase F section
- Noted that pty-chi uses PyTorch internally for LSQML reconstruction

### CONFIG-001 (docs/findings.md:10)
- Confirmed orchestrator remains pure (no params.cfg manipulation)
- Documentation references CONFIG-001 compliance in builder narrative

### CONFIG-002 (docs/findings.md:11)
- Execution configs remain isolated in subprocess calls
- CLI argument handoff maintains neutrality

### DATA-001 (docs/findings.md:14)
- Referenced amplitude + complex64 requirements in TESTING_GUIDE.md
- Noted dataset contract enforcement in TEST_SUITE_INDEX.md

### OVERSAMPLING-001 (docs/findings.md:17)
- Reiterated K≥C guardrail in study description
- Jobs inherit K=7 from Phase D/E configurations

---

## File Changes Summary

| File | Lines Changed | Nature |
|------|--------------|--------|
| `docs/TESTING_GUIDE.md` | +63 (lines 146-208) | Added Phase F section with selectors, CLI commands, evidence pointers |
| `docs/development/TEST_SUITE_INDEX.md` | +1 row (line 61) | Registered `test_dose_overlap_reconstruction.py` with full selector details |
| `plans/active/.../phase_f_ptychi_baseline_plan/plan.md` | 1 cell (line 39) | Updated F2.4 from `[ ]` to `[x]` with completion narrative |

**Total documentation lines:** 64 lines added across 2 user-facing doc files + 1 plan update

---

## Compliance Checks

### SPEC/ADR Alignment
- No SPEC requirements for documentation structure; aligned with existing doc patterns
- Followed TEST_SUITE_INDEX.md table schema for consistency
- Matched TESTING_GUIDE.md narrative style from Phase E5 section

### Module Scope
- **Category:** Docs only
- **No code changes** - stayed within documentation scope as required by Mode: Docs

### Documentation Quality
- All artifact paths use relative paths (no absolute `/home/` references)
- CLI commands include environment variable `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md`
- Evidence pointers reference timestamped artifact hubs for traceability
- Selector snippets provide multiple entry points (module, function, filter, collection)

---

## Exit Criteria Status

- [x] Phase F section added to `docs/TESTING_GUIDE.md` with selectors and CLI commands
- [x] `test_dose_overlap_reconstruction.py` registered in `docs/development/TEST_SUITE_INDEX.md`
- [x] Phase F plan F2.4 marked `[x]` with evidence links
- [x] pytest collection proof captured (4 tests, 0.83s)
- [x] Findings POLICY-001, CONFIG-001/002, DATA-001, OVERSAMPLING-001 documented
- [x] summary.md written documenting all changes

---

## Blockers / Issues

None encountered. All documentation updates completed successfully, collection proof validates 4 tests remain discoverable.

---

## Metrics

- **Files updated:** 3 (2 doc files + 1 plan file)
- **Lines added:** 64 (docs) + 1 (plan completion note)
- **Tests documented:** 4 Phase F reconstruction tests
- **Selectors registered:** 7 pytest command variations
- **Evidence hubs linked:** 2 (230000Z dense/test run, 233500Z doc sync)
- **Duration:** ~15 minutes (documentation-only loop)

---

## Next Actions

1. Update `docs/fix_plan.md` with Attempt #82 entry in Attempts History
2. Mark Phase F2.4 complete in fix_plan.md ledger
3. Optionally proceed to sparse/train LSQML baseline or advance to Phase G comparisons

---

**Phase F2.4 Documentation Sync COMPLETE**

---

## Cross-References

- Dense/test run evidence: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T230000Z/phase_f_ptychi_baseline_f2_dense_test_run/`
- Doc sync artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T233500Z/phase_f_ptychi_baseline_f2_doc_sync/`
- Phase F plan: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md`
- Test strategy: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/test_strategy.md`

---

**End of Summary**
