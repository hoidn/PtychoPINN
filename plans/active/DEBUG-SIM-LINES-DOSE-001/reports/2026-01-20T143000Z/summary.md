### Turn Summary
Added explicit ratio/least-squares tables plus baseline→scaled MAE/RMSE callouts to analyze_intensity_bias so Phase C4d evidence quantifies the normalization gap.
Regenerated the gs1_ideal/gs2_ideal summaries, capturing gs1_ideal ratios near 2.0 while NaNs leave gs2_ideal with zero finite ratios.
Confirmed the analyzer CLI and the synthetic-helpers smoke guard remain green with logs stored under the 2026-01-20T143000Z hub.
Next: fold these quantified ratios into Phase C4d planning to decide whether loader normalization or downstream losses must change before touching core workflows.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/ (bias_summary.md, pytest_cli_smoke.log)

### Turn Summary
Extended the bias analyzer with prediction↔truth scaling diagnostics so the Phase C4d evidence now quantifies whether a single scalar explains the ≈12× amplitude gap.
Stored scaling ratios/least-squares scalars in JSON + Markdown, updated docs/fix_plan.md, and reran the analyzer plus CLI smoke guard with logs in the C4d hub.
Next: interpret the scaling results to decide whether loader normalization or downstream losses need to change before finishing Phase C4.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T143000Z/ (bias_summary.md, pytest_cli_smoke.log)
