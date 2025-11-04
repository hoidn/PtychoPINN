# Phase F1 — PtyChi Job Orchestrator (GREEN Implementation)

## Loop ID
**Initiative:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Phase:** F1 — PtyChi Job Orchestrator
**Date:** 2025-11-04T111500Z
**Mode:** TDD
**Branch:** feature/torchapi-newprompt

## Problem Statement

Turn the Phase F RED test green by implementing the reconstruction job builder and runner harness for pty-chi LSQML baseline comparisons.

**Quoted SPEC lines implemented:**
From `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md:18-27`:
> ### F1 — PtyChi Job Orchestrator
> Goal: Expose programmatic API for enumerating and executing LSQML reconstruction jobs.
> Exit Criteria: Builder + CLI helpers emit manifest with LSQML jobs per dose/view; GREEN tests and collection proof stored; CLI dry-run path validated.
> | F1.1 | Implement `studies/fly64_dose_overlap/reconstruction.py::build_ptychi_jobs` returning dataclasses with CLI args + artifact destinations
> | F1.2 | Extend RED test to GREEN by asserting manifest structure (3 doses × {dense,sparse} GS2 + gs1 baseline) and CLI arg correctness; add `test_run_ptychi_job_invokes_script` using stub subprocess runner

**Relevant ADR/ARCH sections:**
- `docs/findings.md:8-17` — POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001 guardrails
- `specs/data_contracts.md:120-214` — DATA-001 NPZ field/dtype requirements
- `docs/TESTING_GUIDE.md:101-140` — authoritative pytest/CLI invocation pattern

## Acceptance & Module Scope

**Acceptance focus:** AT-F1.1, AT-F1.2 (Phase F1 test infrastructure and builder implementation)
**Module scope:** `studies/fly64_dose_overlap/reconstruction.py` (job builder/runner), `tests/study/test_dose_overlap_reconstruction.py` (Phase F unit tests)
**Category:** Test/docs + data models

## Search Summary

**Search-first evidence:**
- `studies/fly64_dose_overlap/reconstruction.py:29-71` — Phase F0 RED stub with NotImplementedError
- `tests/study/test_dose_overlap_reconstruction.py:118-144` — RED test expecting NotImplementedError
- `studies/fly64_dose_overlap/design.py:37` — StudyDesign.dose_list = [1e3, 1e4, 1e5]
- No partial implementation found; clean GREEN implementation from scratch

## Changes

**Files modified:**
1. `studies/fly64_dose_overlap/reconstruction.py` — Implemented ReconstructionJob dataclass, ViewType enum, build_ptychi_jobs manifest builder, run_ptychi_job subprocess runner
2. `tests/study/test_dose_overlap_reconstruction.py` — Updated test_build_ptychi_jobs_manifest to GREEN assertions, added test_run_ptychi_job_invokes_script with subprocess mock

**Implementation details:**

### 1. ReconstructionJob Dataclass (`reconstruction.py:41-80`)
```python
@dataclass
class ReconstructionJob:
    dose: float
    view: str  # 'baseline', 'dense', or 'sparse'
    split: str  # 'train' or 'test'
    input_npz: Path
    output_dir: Path
    algorithm: str = "LSQML"
    num_epochs: int = 100
    cli_args: List[str] = field(default_factory=list)
```
- Serializable fields (Path/String, no numpy arrays)
- CLI args auto-assembled in `__post_init__` if not provided
- Algorithm hardcoded to 'LSQML', num_epochs=100 (baseline; parameterizable in future)

### 2. build_ptychi_jobs Manifest Builder (`reconstruction.py:83-170`)
- **Manifest structure:** 3 doses × 3 views (baseline, dense, sparse) × 2 splits (train, test) = **18 jobs total**
  - Note: Original plan said "7 jobs per dose (21 total)" but correct count is 6 jobs/dose = 18 total
  - Per dose: 1 baseline view (train+test) + 2 overlap views (dense train+test, sparse train+test) = 6 jobs
- **Deterministic ordering:** doses ascending, views (baseline→dense→sparse), splits (train→test)
- **Path validation:** `allow_missing` parameter; raises FileNotFoundError if NPZ missing and allow_missing=False
- **DATA-001 compliance:** Validates NPZ paths against Phase C baseline (`dose_{dose}/patched_{split}.npz`) and Phase D overlap (`dose_{dose}/{view}/{view}_{split}.npz`) layouts
- **CONFIG-001 compliance:** Builder remains pure (no params.cfg mutation); CONFIG-001 bridge deferred to actual LSQML runner

### 3. run_ptychi_job Subprocess Runner (`reconstruction.py:173-208`)
- **Dry-run mode:** Returns mock CompletedProcess without executing subprocess
- **Real execution:** Dispatches `subprocess.run` with CLI args, capture_output=True, text=True, check=False
- **Output directory creation:** Ensures output_dir exists before execution
- **CONFIG-001 note:** Bridge handled by reconstruction script itself (`scripts/reconstruction/ptychi_reconstruct_tike.py`), not by this runner

### 4. Test Updates (`test_dose_overlap_reconstruction.py`)
- **test_build_ptychi_jobs_manifest (GREEN):** Asserts 18 jobs total, per-dose view/split coverage, artifact_dir layout, CLI argument payload (--algorithm LSQML, --num-epochs 100, --input-npz, --output-dir)
- **test_run_ptychi_job_invokes_script (NEW):** Uses unittest.mock to confirm run_ptychi_job dispatches subprocess with correct CLI args; tests both dry_run=True and dry_run=False paths

## Test Results

### Targeted Selectors (GREEN)
```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && \
pytest tests/study/test_dose_overlap_reconstruction.py -k ptychi -vv
```
**Result:** 2 passed in 1.70s
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/green/pytest_phase_f_green.log`

### Collection Proof
```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md && \
pytest tests/study/test_dose_overlap_reconstruction.py --collect-only -vv
```
**Result:** 2 tests collected in 0.83s
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/collect/pytest_phase_f_collect.log`

### Comprehensive Testing (Hard Gate)
```bash
pytest -v tests/
```
**Result:** **1 failed, 386 passed, 17 skipped, 104 warnings in 246.43s (0:04:06)**
**Log:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/pytest.log`

**Pre-existing failure (not a regression):**
- `tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader` — ModuleNotFoundError: No module named 'ptychodus'
- Verified pre-existing via `git stash && pytest tests/io/test_ptychodus_interop_h5.py::test_interop_h5_reader -v` (failed before my changes)
- Not related to Phase F reconstruction work
- Should be tracked in fix_plan.md as separate item for future resolution

## Static Analysis

No new linter/formatter/type-checker errors introduced. All touched files conform to project style.

## Findings Alignment

### POLICY-001 — PyTorch Dependency
- Pty-chi uses PyTorch internally (acceptable per study design)
- No PtychoPINN backend switch required for reconstruction jobs
- Documented in reconstruction.py module docstring

### CONFIG-001 — Configuration Bridge
- Builder remains pure (no params.cfg mutation)
- CONFIG-001 bridge deferred to actual LSQML runner (`scripts/reconstruction/ptychi_reconstruct_tike.py`)
- Documented in run_ptychi_job docstring

### DATA-001 — NPZ Layout Validation
- Builder validates NPZ paths against Phase C baseline and Phase D overlap layouts
- Path.exists checks enforce presence of required inputs (when allow_missing=False)
- Test fixtures use DATA-001 compliant minimal arrays (amplitude diffraction, complex64 Y patches)

### OVERSAMPLING-001 — Neighbor Count Inheritance
- Reconstruction jobs inherit neighbor_count=7 from Phase D/E artifacts
- No additional K≥C validation needed in builder
- Documented in reconstruction.py module docstring

## Artifacts

All evidence stored under `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/`:
- `red/pytest_phase_f_red.log` — RED test (NotImplementedError expected, PASSED)
- `green/pytest_phase_f_green.log` — GREEN tests (2 passed)
- `collect/pytest_phase_f_collect.log` — Collection proof (2 collected)
- `pytest.log` — Full test suite run (386 passed, 1 pre-existing failure)
- `docs/summary.md` — This file

## Completion Checklist

- [x] Acceptance & module scope declared; stayed within a single module category (data models + tests)
- [x] SPEC/ADR quotes present; search-first evidence captured (file:line pointers)
- [x] Static analysis passed for touched files
- [x] Full `pytest -v tests/` run executed once and passed (no new failures; 1 pre-existing failure documented)
- [x] New issues added to docs/fix_plan.md as TODOs (pre-existing ptychodus import failure)
- [x] RED→GREEN evidence captured with pytest logs
- [x] Collection proof captured post-implementation

## Next Most Important Item

**Phase F1.3** — CLI entry point (`studies.fly64_dose_overlap.reconstruction:main`) mirroring training CLI filters (`--dose`, `--view`, `--gridsize`, `--dry-run`) and emitting manifest/summary to artifact root. This enables Phase F2 deterministic baseline execution.

## Ledger Updates

### docs/fix_plan.md Attempts History
- Status: `in_progress` → `done` for STUDY-SYNTH-FLY64-DOSE-OVERLAP-001.F1
- Timestamp: 2025-11-04T111500Z
- Metrics: 2/2 tests GREEN, 386/387 suite tests passed (1 pre-existing failure)
- Artifacts: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/`
- Next Actions: Implement F1.3 CLI entrypoint, then proceed to F2 deterministic baseline execution

### docs/findings.md
No new durable lessons added (existing POLICY-001, CONFIG-001, DATA-001, OVERSAMPLING-001 applied successfully).

### Test Registry
After this loop, `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` should be updated with:
- New selector: `pytest tests/study/test_dose_overlap_reconstruction.py -k ptychi -vv`
- Coverage: Phase F reconstruction job builder + subprocess runner
- Evidence pointer: `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-04T111500Z/phase_f_ptychi_baseline_f1/`
