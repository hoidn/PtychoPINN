### Turn Summary
Implemented Phase D2 normalization telemetry: extended `write_intensity_stats_outputs()` with stage-to-stage ratios, `normalize_gain`, and "largest drop" marker citing the spec.
Ran gs1_ideal (gridsize=1) and dose_legacy_gs2 (gridsize=2, reduced batch to avoid OOM) through the updated runner; both produce ratio tables showing 44–72% amplitude reduction at the `normalize_data` stage.
Next: Trace the `normalize_data` math to determine if the gain formula `sqrt(((N/2)^2)/mean(sum(X^2)))` violates the spec's symmetry invariants.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/ (bias_summary.md, intensity_stats.json per scenario)

---

### Prior Turn Summary (Supervisor)
Scoped Phase D2 around explicit normalization-ratio telemetry and dose-like replay evidence, updating the plan checklist plus fix_plan attempts history accordingly.
Drafted the new Do Now (input.md) so Ralph adds ratio logging to the runner/analyzer, reruns gs1_ideal and a dose-style override, and archives everything under the fresh 2026-01-20T114126Z hub with the pytest guard.
Next: implement the instrumentation, run both scenarios, generate the analyzer report, and keep the CLI smoke test green.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/
