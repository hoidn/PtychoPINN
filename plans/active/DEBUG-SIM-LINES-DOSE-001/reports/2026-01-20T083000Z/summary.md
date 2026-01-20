### Turn Summary
Instrumented `run_phase_c2_scenario.py` with ground-truth dumps, center-crop helpers, and amplitude/phase diff metrics so each gs*_ideal rerun now writes comparison JSON + PNGs alongside cropped reconstruction arrays and metadata pointers.
Replayed gs1_ideal/gs2_ideal with the baked profiles, regenerated reassembly telemetry (still `fits_canvas=true` at 828/826 px), and archived the CLI smoke selector + reassembly CLI logs; gs1 amplitude MAE ≈2.48 with phase MAE ≈0.16 while gs2 lands at ≈2.48/0.13.
Next: dig into why gs1 training still collapses relative to gs2 despite matching telemetry, or pivot to Phase C4 fixes once the comparison artifacts are reviewed.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T083000Z/ (gs*_ideal_runner.log, gs*_ideal/comparison_metrics.json, amplitude/phase diff PNGs, reassembly_cli.log, pytest logs)

### Turn Summary
Mapped the next Phase C3b increment to extend the Phase C2 runner with ground-truth comparison artifacts so gs1_ideal vs gs2_ideal can be quantified instead of relying on screenshots.
Scoped the code touch to plan-local runner helpers (ground-truth dumps, center-crop diff metrics, metadata updates) plus reassembly/test reruns, updated the implementation plan/fix ledger, and rewrote input.md pointing Ralph at the new artifacts hub.
Next: Ralph updates the runner, reruns gs1_ideal and gs2_ideal with the new comparison outputs, refreshes reassembly telemetry, and archives the pytest evidence.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T083000Z/ (planning notes placeholder)
