### Turn Summary
Built the plan-local analyzer CLI so scenario telemetry (bias, intensity scales, normalization stages, training NaNs) lands in a single JSON/Markdown bundle for Phase C4.
Analyzer results show both gs1_ideal and gs2_ideal still undershoot amplitude by about 2.5 even with matching intensity scales, and gs2 now reports NaNs across every loss metric while normalization stages look identical.
Next: trace the workflow between normalization and the IntensityScaler to find the shared amplitude drop before modifying shared modules.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T121500Z/ (bias_summary.json, bias_summary.md, analyze_intensity_bias.log, pytest_cli_smoke.log)
