# DEBUG-SIM-LINES-DOSE-001 — Final Documentation + Archive Prep

## Summary
Finalize documentation for the completed NaN debugging initiative and prepare for archive.

## Focus
DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (**NaN DEBUGGING COMPLETE**)

## Branch
paper

## Mapped Tests
`pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`

## Artifacts
`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143500Z/`

## Do Now (Documentation Tasks)

### 1. Add Knowledge Base Entry (docs/findings.md)

Add a new finding ID `DEBUG-SIM-LINES-DOSE-001-COMPLETE` documenting:
- **Root Cause:** CONFIG-001 violation (stale `params.cfg` values not synced before training/inference)
- **Fix:** C4f bridging — `update_legacy_dict(params.cfg, config)` calls added to `scripts/studies/sim_lines_4x/pipeline.py` and plan-local runner before all training/inference handoffs
- **Verification:** All four scenarios (gs1_ideal, gs1_custom, gs2_ideal, gs2_custom) train without NaN
- **Remaining Issue:** Amplitude bias (~3-6x undershoot) is separate from NaN debugging; may require future investigation

### 2. Update Initiative Summary (plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md)

Prepend a final turn summary:
```markdown
### Turn Summary — 2026-01-20T14:35:00Z (Final)
NaN debugging initiative COMPLETE. A1b ground-truth run blocked by Keras 3.x incompatibility (legacy model uses `tf.shape()` on KerasTensors); documented closure rationale showing A1b is no longer required since CONFIG-001 root cause already confirmed and fixed. Initiative ready for archive after soak period.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143500Z/
```

### 3. Run CLI Smoke Guard

```bash
AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v 2>&1 | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143500Z/pytest_cli_smoke.log
```

## How-To Map

| Step | Command/Action |
|------|----------------|
| Knowledge base entry | Edit `docs/findings.md`, add new entry after existing findings |
| Initiative summary | Edit `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md`, prepend turn summary |
| Pytest guard | Run the CLI smoke selector and archive the log |

## Pitfalls To Avoid
- Do not modify core physics/model modules
- Do not delete any reports or artifacts
- Ensure knowledge base entry cites the correct fix (C4f CONFIG-001 bridging)
- Keep the amplitude bias separate from the NaN debugging scope

## If Blocked
If pytest fails unexpectedly, archive the error log and document the failure signature in the summary without marking the initiative incomplete.

## Findings Applied (Mandatory)
- CONFIG-001: The root cause of all NaN failures — stale params.cfg values
- POLICY-001: PyTorch environment requirements (not directly applicable here)
- NORMALIZATION-001: Amplitude bias is separate; not addressed in this initiative

## Pointers
- Root cause evidence: `docs/fix_plan.md` lines 175-184 (NaN DEBUGGING MILESTONE COMPLETE)
- A1b closure rationale: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143500Z/a1b_closure_rationale.md`
- Implementation plan: `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` (A1b status at lines 99-108)
- Test strategy: `plans/active/DEBUG-SIM-LINES-DOSE-001/test_strategy.md`

## Next Up (Optional)
After this documentation handoff, the initiative is ready to be marked `done` in `docs/fix_plan.md` and moved to archive after a short soak period. Focus can then shift to PARALLEL-API-INFERENCE or FIX-DEVICE-TOGGLE-001.
