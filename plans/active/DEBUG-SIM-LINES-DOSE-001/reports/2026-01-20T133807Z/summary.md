### Turn Summary
Scoped the Phase D3 hyperparameter audit by updating docs/fix_plan + the initiative summary, reserving the 2026-01-20T133807Z hub, and defining the CLI changes needed to capture sim_lines vs dose_experiments training knobs.
Rewrote input.md with the new Do Now so Ralph can extend compare_sim_lines_params.py, rerun the diff with `--default-sim-lines-nepochs 5`, and prove the CLI smoke selector.
Next: Ralph implements the CLI updates, regenerates hyperparameter_diff artifacts, and archives the pytest log under the new hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/ (summary.md)
