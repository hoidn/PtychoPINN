### Turn Summary (2026-01-20T120753Z)
Executed Phase D2 evidence reruns: gs1_ideal (gridsize=1, stable profile) and dose_legacy_gs2 (gridsize=2, custom probe, batch=4/group=64) through the existing instrumented runner.
Key findings: both scenarios drop 40-70% amplitude at `normalize_data` (gs1 ratio=0.56, gs2 ratio=0.27); gs1_ideal collapses to NaN during training while gs2 trains healthily but needs ~3.9× scaling to approach ground truth.
Next: inspect `normalize_data()` gain formula to determine if scale is double-applied.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T114126Z/ (bias_summary.md, intensity_stats.md per scenario, pytest_cli_smoke.log)

---

### Prior Turn Summary (Implementation)
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
