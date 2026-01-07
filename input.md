Mode: Implementation
Focus: STUDY-SYNTH-DOSE-COMPARISON-001 — Synthetic Dose Response & Loss Comparison Study
Branch: feature/torchapi-newprompt-2
Selector: scripts/studies/dose_response_study.py (direct execution)
Artifacts: plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/

## Summary

Execute the dose response study now that REFACTOR-MODEL-SINGLETON-001 is complete. This will verify D4 (dose_response_study.py runs with varying N/gridsize) and produce scientific results.

## Goal

Run `dose_response_study.py` end-to-end with 4 experimental arms (High Dose NLL, High Dose MAE, Low Dose NLL, Low Dose MAE) to:
1. Verify REFACTOR-MODEL-SINGLETON-001 Phase D4 exit criterion (varying N/gridsize works)
2. Train models and produce reconstructions
3. Generate the 6-panel comparison figure

## Tasks

### Task 1: Verify study script structure (A0 from plan)

**File:** `scripts/studies/dose_response_study.py`

Check that the script is properly structured:
- No XLA workarounds at the top (Phase C removed them)
- Uses `TrainingConfig` and `update_legacy_dict`
- Has entry point for execution

Run a quick import test:
```bash
python -c "from scripts.studies.dose_response_study import main; print('Script imports OK')"
```

### Task 2: Execute the study

Run the dose response study with reduced epochs for initial verification:
```bash
# Create artifacts directory
mkdir -p plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/

# Run with reduced epochs for verification
python scripts/studies/dose_response_study.py \
    --output-dir tmp/dose_study \
    --nepochs 5 \
    2>&1 | tee plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/dose_study_run.log
```

**Expected:**
- 4 models trained (high_nll, high_mae, low_nll, low_mae)
- Figure `dose_comparison.png` generated
- No shape mismatch errors (confirms REFACTOR-MODEL-SINGLETON-001 fix)

### Task 3: Capture evidence

If successful:
```bash
# Copy figure to artifacts
cp tmp/dose_study/dose_comparison.png plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/

# Record completion
echo "STUDY-SYNTH-DOSE-COMPARISON-001: VERIFIED" >> plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/status.txt
```

If blocked:
- Capture full error traceback
- Note the specific failure point (simulation, training, inference, visualization)
- Record in `blocked_<timestamp>.md`

## Implement

- Execute `scripts/studies/dose_response_study.py` with reduced epochs
- Verify no shape mismatch errors occur
- Capture output figure and logs

## How-To Map

```bash
# Quick test that script loads
python -c "import scripts.studies.dose_response_study"

# Full run with reduced epochs
python scripts/studies/dose_response_study.py --output-dir tmp/dose_study --nepochs 5

# Check for shape errors in log
grep -i "shape\|mismatch\|reshape" plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/reports/2026-01-07T073000Z/dose_study_run.log

# Verify output
ls -la tmp/dose_study/
ls -la tmp/dose_study/dose_comparison.png
```

## Pitfalls To Avoid

1. **DO NOT** add XLA workarounds — they should not be needed with lazy loading
2. **DO NOT** skip `update_legacy_dict` calls between training arms
3. **DO** use reduced epochs (5) for verification; full run can come later
4. **DO** capture the full log including any warnings
5. **DO** check for DeprecationWarnings from lazy singleton access (expected but harmless)
6. **DO** verify the figure shows actual data, not placeholders

## If Blocked

1. If shape mismatch: Check if module is being imported before params.cfg is set
2. If OOM: Reduce n_groups in the study config
3. If inference fails: Check that model was saved correctly to output_dir
4. Log the error and create blocker document with:
   - Error message and traceback
   - Which arm failed (high_nll, etc.)
   - Last successful step

## Findings Applied

- **MODULE-SINGLETON-001**: RESOLVED — Lazy loading prevents import-time model creation
- **CONFIG-001**: Must call `update_legacy_dict(params.cfg, config)` before each training arm
- **NORMALIZATION-001**: Respect intensity_scale symmetry per spec-ptycho-core.md
- **DATA-001**: Generated RawData follows data contract

## Pointers

- Study implementation: `plans/active/STUDY-SYNTH-DOSE-COMPARISON-001/implementation.md`
- Lazy loading fix: `ptycho/model.py:867-890` (`__getattr__`)
- Factory API: `ptycho/model.py::create_compiled_model()`, `create_model_with_gridsize()`
- Fix plan: `docs/fix_plan.md` (STUDY-SYNTH-DOSE-COMPARISON-001 entry)

## Next Up

- If study succeeds: Full production run with 50 epochs
- Update fix_plan with results and metrics
