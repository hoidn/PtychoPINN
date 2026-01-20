### Turn Summary
Implemented CONFIG-001 bridging in `scripts/studies/sim_lines_4x/pipeline.py` (run_scenario + run_inference) and `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py` (main + run_inference_and_reassemble).
Reran gs1_ideal + gs2_ideal with `--prediction-scale-source least_squares` and captured refreshed analyzer outputs confirming bundle/legacy intensity_scale delta=0.
gs2_ideal now healthy (no NaNs, pearson_r=0.135, least_squares=1.91) but gs1_ideal still collapses at epoch 3 (all metrics NaN), suggesting a gridsize=1 numeric instability unrelated to CONFIG-001 drift.
Next: investigate gs1_ideal's NaN source or pivot to core workflow normalization audit if amplitude bias persists.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T160000Z/ (bias_summary.md, gs*_ideal/**, pytest_cli_smoke.log)
