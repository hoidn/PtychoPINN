# Phase G Dense Execution — Loop Summary (2025-11-07T070500Z)

## Objective
Implement Phase C→G pipeline orchestrator script and execute real evidence run for dose=1000 dense/train,test comparisons.

## Outcomes

### ✓ Script Implementation (COMPLETED)
Created `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py`:
- **TYPE-PATH-001 compliance**: All filesystem paths normalized to `Path` objects
- **AUTHORITATIVE_CMDS_DOC propagation**: Environment variable checked and set if missing
- **Fail-fast behavior**: Non-zero subprocess returns halt execution and write `analysis/blocker.log`
- **Comprehensive logging**: Per-phase CLI logs with command headers and stdout/stderr capture
- **Collect-only mode**: Dry-run verification without execution (`--collect-only`)

**Script features**:
- Sequential orchestration of 8 commands (Phase C→D→E→F→G)
- Phase E training for both baseline (gs1) and view-specific (gs2) models
- Phase F reconstruction for each split (train/test)
- Phase G comparison for each split with manifest-driven `--tike_recon_path` wiring
- Artifact organization: `{HUB}/data/{phase_c,phase_d,phase_e,phase_f}`, `{HUB}/analysis`, `{HUB}/cli`

### ✓ Pytest Validation (GREEN)
**Targeted test**: `tests/study/test_dose_overlap_comparison.py -k tike_recon_path`
- **Result**: PASSED (1.78s)
- **Validates**: Phase G manifest parsing and `--tike_recon_path` flag appending
- **Evidence**: `green/pytest_phase_g_manifest_green.log`

### ✗ Pipeline Execution (BLOCKED)
**Command**:
```bash
export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
python plans/active/.../bin/run_phase_g_dense.py --hub ... --dose 1000 --view dense --splits train test
```

**Blocker**: Phase C dataset generation failed at command 1/8
- **Error**: `TypeError: object of type 'float' has no len()` in `ptycho/raw_data.py:227`
- **Root cause**: `studies/fly64_dose_overlap/generation.py` passes `xcoords` as `float` instead of array to `RawData.from_simulation`
- **Impact**: Cannot proceed to Phase D/E/F/G without Phase C datasets
- **Scope**: **Pre-existing bug in Phase C generation**, outside Phase G orchestration scope

**Blocker artifacts**:
- `analysis/blocker.log`: Captured failing command and return code
- `cli/phase_c_generation.log`: Full Phase C CLI transcript with traceback

**Script behavior validation**:
- ✓ Detected non-zero return code (exit 1)
- ✓ Halted execution immediately (did not proceed to Phase D)
- ✓ Wrote blocker.log with command, return code, and log path
- ✓ Propagated exit code to caller

### ✓ Full Test Suite (PASSED)
**Command**: `pytest -v tests/`
- **Result**: 402 passed, 1 pre-existing fail (test_interop_h5_reader), 17 skipped
- **Duration**: 464.31s (7m44s)
- **Baseline parity**: Matches expected test health
- **Evidence**: `full/pytest_full_suite.log`

**New test coverage** (from prior Phase G manifest wiring loop):
- `test_execute_comparison_jobs_appends_tike_recon_path` validates manifest parsing + `--tike_recon_path` flag

## Acceptance Status

**Phase G Orchestration Script (AT-G1)**: ✓ COMPLETE
- Script implements all required CLI command sequencing
- TYPE-PATH-001 enforcement via `Path()` normalization
- Fail-fast + blocker logging behavior validated
- Collect-only mode provides dry-run verification

**Phase G Dense Evidence (AT-G2)**: ✗ BLOCKED
- Cannot generate Phase C→G evidence due to Phase C generation bug
- Blocker recorded in `analysis/blocker.log` per Ralph protocol
- Phase G comparison logic and manifest wiring are GREEN (prior loop)

## Artifacts

### Implemented Code
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py` (262 lines, executable)

### Test Evidence
- `green/pytest_phase_g_manifest_green.log` — Targeted test GREEN (1.78s, 1 passed)
- `full/pytest_full_suite.log` — Full suite GREEN (464s, 402 passed / 1 pre-existing fail / 17 skipped)

### Execution Logs
- `cli/phase_c_generation.log` — Phase C CLI transcript with error traceback
- `analysis/blocker.log` — Failing command capture (per Ralph protocol)

### Collect-Only Output
Script dry-run shows 8 planned commands:
1. Phase C: Dataset Generation
2. Phase D: Overlap View Generation
3. Phase E: Training Baseline (gs1)
4. Phase E: Training Dense (gs2)
5. Phase F: Reconstruction dense/train
6. Phase F: Reconstruction dense/test
7. Phase G: Comparison dense/train
8. Phase G: Comparison dense/test

## Findings Applied
- **POLICY-001**: PyTorch remains required dependency (not exercised; TensorFlow backend used)
- **CONFIG-001**: Training CLI bridges legacy config (not reached; blocked at Phase C)
- **DATA-001**: Dataset contracts (not validated; Phase C failed before generation)
- **OVERSAMPLING-001**: Dense overlap spacing (not reached; blocked at Phase C)
- **TYPE-PATH-001**: Path normalization enforced in new script (✓ implemented)

## Next Actions

### Immediate (Unblock Phase G Evidence)
1. **Fix Phase C generation bug**: `studies/fly64_dose_overlap/generation.py` must pass array `xcoords`/`ycoords` to `RawData.from_simulation`, not float scalars
   - **Error signature**: `TypeError: object of type 'float' has no len()` at `raw_data.py:227`
   - **Context**: Line 170 in `generation.py` calls `simulate_fn()` which eventually reaches `raw_data.py:227` expecting `len(xcoords)`
   - **Expected fix scope**: Type validation or coordinate generation logic in Phase C generation

2. **Rerun orchestrator**: Once Phase C fix lands, re-execute:
   ```bash
   export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
   python plans/active/.../bin/run_phase_g_dense.py --hub <new-hub> --dose 1000 --view dense --splits train test
   ```

3. **Capture metrics**: After successful run, extract MS-SSIM/MAE from `analysis/` outputs and document in follow-up summary

### Follow-Up (Post-Dense Evidence)
- Execute sparse view comparisons (`--view sparse --dose 1000`)
- Execute multi-dose sweeps (500, 2000, 10000, 100000)
- Consolidate cross-dose/view metrics into initiative-level report

## Acceptance Focus & Module Scope

**Acceptance Focus**: Phase G orchestration script implementation (AT-G1 partial: script complete, evidence blocked)

**Module Scope**: Scripts/orchestration — no production module changes

**Stop Rule Compliance**: ✓ Stayed within scripts + orchestration; no edits to core modules (model.py, diffsim.py, tf_helper.py)

## SPEC/ADR Alignment

**SPEC Sections Implemented**:
- TYPE-PATH-001 (docs/findings.md:21): Path normalization in orchestrator script
- AUTHORITATIVE_CMDS_DOC guard (docs/TESTING_GUIDE.md:183): Environment variable propagation

**ADR References**:
- Phase pipeline structure per `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md` Phase C→G sequence
- CLI argument contracts per `studies/fly64_dose_overlap/{generation,overlap,training,reconstruction,comparison}.py` main() argparse definitions

**Blocker Documentation**:
- Pre-existing Phase C bug: `studies/fly64_dose_overlap/generation.py:170` → `ptycho/raw_data.py:227` type mismatch
- Outside Phase G orchestration scope; marked for follow-up fix-plan item
