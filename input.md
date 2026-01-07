Mode: Implementation
Focus: STUDY-SYNTH-DOSE-COMPARISON-001 — Synthetic Dose Response & Loss Comparison Study
Branch: feature/torchapi-newprompt-2
Mapped tests: pytest --collect-only -q tests/ (registry check)
Artifacts: plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/

## Summary

Execute the synthetic dose response study now that all blockers are resolved. The FIX-GRIDSIZE-TRANSLATE-BATCH-001 fix (commit 13599565) added XLA-compatible batch broadcast to `translate_xla`, and all 8 translation/model tests pass.

## Current State

- **FIX-GRIDSIZE-TRANSLATE-BATCH-001**: ✅ DONE (commit 13599565, 8/8 tests pass)
- **REFACTOR-MODEL-SINGLETON-001**: ✅ DONE (lazy loading prevents import-time model creation)
- **Prior run artifacts**: `dose_comparison.png` shows "No Data" for all 4 reconstruction panels (from BEFORE the fix)
- **Study status**: Ready for execution

## Do Now

### Phase A: Verify Environment

1. **Quick sanity check** - verify the XLA fix is active:
```bash
pytest tests/tf_helper/test_translation_shape_guard.py -v -k "gridsize_broadcast" --tb=short 2>&1 | tee plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/pytest_sanity.log

# Expected: 2 passed
```

### Phase B: Execute Dose Response Study

2. **Run the study** with moderate epochs (5) for quick validation:
```bash
cd /home/ollie/Documents/PtychoPINN

# Run dose response study
timeout 1800 python scripts/studies/dose_response_study.py \
    --output-dir plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/study_outputs \
    --nepochs 5 \
    2>&1 | tee plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/dose_study_run.log

# Check for success
grep -i "error\|exception\|shape.*mismatch" plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/dose_study_run.log && echo "ERRORS FOUND" || echo "NO ERRORS"
```

### Phase C: Verify Outputs

3. **Check artifacts** were produced:
```bash
# Check for trained models
ls -la plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/study_outputs/*/wts.h5.zip 2>/dev/null || echo "No model weights found"

# Check for figure
ls -la plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/study_outputs/dose_comparison.png 2>/dev/null || echo "No figure found"

# Copy figure to reports root for easy access
cp plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/study_outputs/dose_comparison.png \
   plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/dose_comparison_v3.png 2>/dev/null || true
```

### Phase D: Test Registry Check

4. **Run pytest collect** to verify no regressions:
```bash
pytest --collect-only -q tests/ 2>&1 | tee plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/pytest_collect.log

# Check for errors
grep -i "error" plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z/pytest_collect.log && echo "Collection errors found" || echo "Collection OK"
```

## How-To Map

```bash
# Artifacts directory
ARTIFACTS=plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T200000Z

# Full study run
python scripts/studies/dose_response_study.py --output-dir $ARTIFACTS/study_outputs --nepochs 5

# If OOM or timeout, try smaller batch:
python scripts/studies/dose_response_study.py --output-dir $ARTIFACTS/study_outputs --nepochs 3 --batch-size 8
```

## Pitfalls To Avoid

1. **DO NOT** modify the translation layer code - the fix is complete
2. **DO NOT** add XLA workarounds (`USE_XLA_TRANSLATE=0`) - XLA should work now
3. **DO** ensure the study runs with default XLA settings (no env var overrides)
4. **DO** capture full logs for debugging if failures occur
5. **DO** check that reconstruction panels show actual data (not "No Data")
6. **DO** verify all 4 arms produce model weights
7. **Environment Freeze:** Do not install packages or modify environment

## If Blocked

1. If XLA errors persist: Log the full traceback, note any shape values
2. If OOM: Reduce batch_size to 8 or nepochs to 3
3. If timeout: Reduce nepochs to 2 for quick validation
4. Record blocker in `$ARTIFACTS/blocked_<timestamp>.md` with error signature

## Findings Applied

- **MODULE-SINGLETON-001:** ✅ Resolved - lazy loading active, no XLA workarounds needed
- **CONFIG-001:** `update_legacy_dict` must be called before legacy modules - handled in dose_response_study.py
- **BUG-TF-001:** `params.cfg['gridsize']` must be set before data generation - handled
- **TF-NON-XLA-SHAPE-001:** Non-XLA batch broadcast - reference for XLA fix design (commit 13599565)

## Pointers

- Implementation plan: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/implementation.md`
- XLA fix commit: 13599565 (`FIX-GRIDSIZE-TRANSLATE-BATCH-001`)
- Dose response script: `scripts/studies/dose_response_study.py`
- Prior blocker analysis: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/blockers/2026-01-06_xla_shape_mismatch_hypothesis.md`
- fix_plan: `docs/fix_plan.md` (STUDY-SYNTH-DOSE-COMPARISON-001 entry)

## Exit Criteria

1. `dose_response_study.py` completes without exceptions
2. All 4 arms produce `wts.h5.zip` model weights
3. `dose_comparison.png` shows image data (not "No Data") in all 6 panels
4. `pytest --collect-only` shows no regressions
5. Ledger updated with results

## Next Up

- If successful: Update fix_plan.md to mark study `done`, archive artifacts
- If blocked: Document specific failure mode, determine if fix needs refinement
