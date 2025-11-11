# Ralph Loop Report: Phase G Dense Pipeline Execution

**Loop Timestamp:** 2025-11-11T194104Z
**Focus ID:** STUDY-SYNTH-FLY64-DOSE-OVERLAP-001
**Mode:** Perf
**Hub:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/`

## Problem Statement

Execute the counted Phase C→G dense pipeline run with `--clobber` to regenerate all artifacts (Phase C manifests, Phase D overlap metrics, Phase E training models, Phase F baseline reconstructions, Phase G comparison metrics), followed immediately by the `--post-verify-only` sweep to populate the `analysis/` directory with SSIM grid, verification, highlights, preview, metrics summary/digest, and artifact inventory evidence.

**SPEC Lines Implemented:**
- From `input.md:14-18`: Execute `run_phase_g_dense.py::main` with `--hub "$HUB" --dose 1000 --view dense --splits train test --clobber`, then immediately run `--post-verify-only` mode.
- Pytest guards from `input.md:8-16`: Collect-only and execution tests for post-verify automation.

**ADRs/ARCH Aligned:**
- TEST-CLI-001: All CLI outputs archived via tee to `$HUB/cli/`
- TYPE-PATH-001: Hub-relative path printing enforced in success banners
- PREVIEW-PHASE-001: Preview artifact must contain only phase deltas (no amplitude text)
- DATA-001: Phase C regenerates NPZ datasets with proper metadata compliance

## Search Summary

**Existing Implementation:**
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py:43`: Main orchestrator with Phase C→G command chain
- `tests/study/test_phase_g_dense_orchestrator.py:437`: Pytest guards for collect-only and execution verification

**Missing Prior to This Loop:**
- Dense Phase C→G execution evidence from this repository (`/home/ollie/Documents/PtychoPINN`)
- All `analysis/` artifacts (SSIM grid, verification report, highlights checker output, metrics summary/digest, artifact inventory)

## Changes Implemented

### 1. Pre-Execution Guards (Steps 0-2)

**Git Hygiene:**
- Stashed unstaged prompt/doc changes
- Executed `timeout 30 git pull --rebase` successfully (already up to date)
- Restored stashed changes

**Hub Preparation:**
- Created/verified hub directory structure: `$HUB/{analysis,archive,cli,collect,data,green,red,summary}`
- Workspace guard: Confirmed `pwd -P` = `/home/ollie/Documents/PtychoPINN`

### 2. Pytest Guards (TEST-CLI-001)

**Collect-Only Verification:**
```bash
pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv
```
- **Result:** ✅ 1 test collected (17 deselected)
- **Evidence:** `$HUB/collect/pytest_collect_post_verify_only.log`

**GREEN Test Execution:**
```bash
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv
```
- **Result:** ✅ PASSED in 0.88s
- **Evidence:** `$HUB/green/pytest_post_verify_only.log`

### 3. Phase C→G Dense Pipeline Execution (In Progress)

**Command:**
```bash
PYTHONPATH=/home/ollie/Documents/PtychoPINN:$PYTHONPATH \
  python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
  --hub "$HUB" \
  --dose 1000 \
  --view dense \
  --splits train test \
  --clobber
```

**Status:** Running in background (bash_id: 15310e)

**Progress (as of 2025-11-11T194648Z):**
- ✅ **Phase C dose=1000:** Dataset generation complete
  - Generated: `simulated_raw.npz` → `canonical.npz` → `patched.npz`
  - Split: `patched_train.npz` (5088 scans), `patched_test.npz` (5216 scans)
  - Validated: DATA-001 compliance passed for both splits

- 🔄 **Phase C dose=10000:** In progress (currently simulating diffraction patterns)

- ⏳ **Phase C dose=100000:** Pending

- ⏳ **Phases D-G:** Pending (overlap computation, training, baseline reconstruction, comparison)

**Evidence:** `$HUB/cli/run_phase_g_dense_stdout.log` (actively logging)

### 4. Technical Resolution: PYTHONPATH Fix

**Issue:** Initial execution failed with `ModuleNotFoundError: No module named 'ptycho'`

**Root Cause:** The `run_phase_g_dense.py` script imports from `ptycho.*` but when invoked directly, Python doesn't include the project root in the module search path. The project uses an editable install, but the module is located at `/home/ollie/Documents/PtychoPINN/ptycho/`.

**Resolution:** Prefixed command with `PYTHONPATH=/home/ollie/Documents/PtychoPINN:$PYTHONPATH` to ensure the `ptycho` package is discoverable.

**Verification:** Second execution started successfully, GPU initialized, TensorFlow loaded, Phase C simulation proceeding.

## Test Results

### Targeted Tests

1. **Collect-Only Guard:**
   - Selector: `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv`
   - Result: ✅ 1 test collected
   - Log: `$HUB/collect/pytest_collect_post_verify_only.log`

2. **Execution Chain Test:**
   - Selector: `pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv`
   - Result: ✅ PASSED (0.88s)
   - Log: `$HUB/green/pytest_post_verify_only.log`

### Comprehensive Suite

**Not Yet Run:** The full `pytest -v tests/` suite will execute after the pipeline completes and all artifacts are validated.

## Artifacts Generated (Partial)

### Existing (from this loop):
- `$HUB/collect/pytest_collect_post_verify_only.log` — Collect-only output (1 test)
- `$HUB/green/pytest_post_verify_only.log` — GREEN test execution (PASSED)
- `$HUB/cli/run_phase_g_dense_stdout.log` — Dense pipeline stdout (in progress)
- `$HUB/cli/phase_c_generation.log` — Phase C dataset generation log
- `$HUB/data/phase_c/dose_1000/{simulated_raw,canonical,patched,patched_train,patched_test}.npz` — Dose 1000 datasets
- `$HUB/summary/progress_note.md` — Execution progress tracking
- `$HUB/summary/ralph_loop_report.md` — This report

### Pending (upon pipeline completion):
- `$HUB/cli/run_phase_g_dense_post_verify_only.log` — Post-verify-only mode output
- `$HUB/analysis/ssim_grid_summary.md` — SSIM grid summary
- `$HUB/analysis/ssim_grid.log` — SSIM grid computation log
- `$HUB/analysis/verification_report.json` — Verification results JSON
- `$HUB/analysis/verify_dense_stdout.log` — Verifier stdout
- `$HUB/analysis/check_dense_highlights.log` — Highlights checker output
- `$HUB/analysis/metrics_summary.json` — Metrics summary JSON
- `$HUB/analysis/metrics_delta_highlights_preview.txt` — Preview with phase-only deltas (PREVIEW-PHASE-001)
- `$HUB/analysis/metrics_digest.md` — Human-readable metrics digest
- `$HUB/analysis/artifact_inventory.txt` — Complete artifact inventory

## Documentation Updates

### Pending (upon completion):
- `docs/fix_plan.md` Attempts History: Add entry with timestamp, metrics deltas (MS-SSIM ±0.000 / MAE ±0.000000), preview verdict, artifact paths, pytest selectors
- `docs/findings.md`: No new findings to add (existing findings applied correctly)
- `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/implementation.md`: Phase G checklist items remain for post-completion verification

## Next Actions

### Immediate (within this background process):
1. ⏳ Complete Phase C for remaining dose levels (10000, 100000)
2. ⏳ Execute Phase D: Overlap computation
3. ⏳ Execute Phase E: PINN training (dose=1000, view=dense)
4. ⏳ Execute Phase F: Baseline reconstruction
5. ⏳ Execute Phase G: Three-way comparison with MS-SSIM metrics
6. ⏳ Execute `--post-verify-only` mode immediately after Phase G

### Follow-Up (next loop):
1. Validate all expected artifacts exist in `$HUB/analysis/`
2. Extract MS-SSIM ±0.000 / MAE ±0.000000 deltas from `metrics_summary.json`
3. Verify `metrics_delta_highlights_preview.txt` contains only phase deltas (no amplitude text) per PREVIEW-PHASE-001
4. Run full pytest suite: `pytest -v tests/`
5. Update `docs/fix_plan.md` Attempts History with complete results
6. Commit with message: `STUDY-SYNTH-FLY64-DOSE-OVERLAP-001 Phase G: Dense pipeline execution evidence (tests: pytest tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only)`

## Findings Applied

- ✅ **POLICY-001:** PyTorch ≥2.2 is mandatory; environment already configured
- ✅ **CONFIG-001:** `update_legacy_dict(params.cfg, config)` runs inside orchestrator (no manual intervention needed)
- ✅ **DATA-001:** Phase C validates NPZ compliance automatically
- ✅ **TYPE-PATH-001:** Hub-relative paths enforced in orchestrator success banners
- ✅ **TEST-CLI-001:** All pytest logs and CLI outputs archived via tee
- ✅ **PREVIEW-PHASE-001:** Preview validation will check for phase-only deltas post-execution
- ✅ **PHASEC-METADATA-001:** Phase C manifests regenerated automatically by `--clobber` mode

## Completion Checklist

### Completed This Loop:
- ✅ Acceptance & module scope declared: Phase G orchestrator execution (module scope: CLI/orchestration)
- ✅ SPEC quotes present: `input.md:14-18` (Do Now), `input.md:8-16` (pytest guards)
- ✅ Search-first evidence: File pointers to orchestrator (`bin/run_phase_g_dense.py:43`) and tests (`tests/study/test_phase_g_dense_orchestrator.py:437`)
- ✅ Static analysis: N/A (no code changes this loop; execution-only)
- ✅ Targeted tests passed: Both collect-only and execution guards GREEN

### Pending Next Loop:
- ⏳ Full `pytest -v tests/` run after pipeline completion
- ⏳ Artifact validation and metrics extraction
- ⏳ Update `docs/fix_plan.md` with complete results
- ⏳ Git commit with evidence links

## Risk Assessment

### Current Status: LOW RISK

**Successful Indicators:**
- Phase C dose=1000 completed successfully (5088 train + 5216 test scans)
- DATA-001 validation passed for both train/test splits
- GPU detected and initialized (NVIDIA GeForce RTX 3090, 22GB)
- TensorFlow XLA compilation succeeded
- No errors or warnings in Phase C execution logs

**Monitoring:**
- Pipeline running in background (bash_id: 15310e)
- Logs actively capturing all stdout/stderr to `$HUB/cli/`
- Progress can be monitored via `BashOutput` tool

**Mitigation:**
- If pipeline fails mid-execution, blocker logs will capture exact failure point
- `--clobber` mode ensures clean restart capability
- All Phase C outputs preserved under `$HUB/data/phase_c/` for debugging

## Ralph Loop Summary

Initiated the counted Phase C→G dense pipeline execution after verifying pytest guards (collect-only and GREEN test both passed). The pipeline is running successfully in the background with Phase C dose=1000 already complete and dose=10000 in progress. All CLI outputs are being archived to `$HUB/cli/` per TEST-CLI-001. Upon completion, the `--post-verify-only` mode will populate `$HUB/analysis/` with SSIM grid, verification, highlights, preview, metrics, and inventory artifacts required for this focus. Next loop will validate the complete artifact bundle, extract MS-SSIM ±0.000 / MAE ±0.000000 deltas, and update `docs/fix_plan.md` Attempts History before committing.

**Artifacts:** `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier/` (collect/, green/, cli/, data/phase_c/dose_1000/, summary/)
