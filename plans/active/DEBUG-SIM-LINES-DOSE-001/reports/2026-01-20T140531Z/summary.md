### Turn Summary
Reopened Phase D3 with explicit D3a completion plus the D3b/D3c checklist and synchronized docs/fix_plan + summary + input around the 60-epoch gs2_ideal retrain.
Captured the 12× nepochs gap findings and queued the concrete `run_phase_c2_scenario.py --scenario gs2_ideal --nepochs 60` + analyzer + pytest sequence so Ralph can validate H-NEPOCHS.
Next: Ralph executes the 60-epoch rerun, archives training histories/analyzer outputs, and reports whether amplitude/pearson_r improve enough to adjust sim_lines defaults.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T140531Z/ (planning hub)
