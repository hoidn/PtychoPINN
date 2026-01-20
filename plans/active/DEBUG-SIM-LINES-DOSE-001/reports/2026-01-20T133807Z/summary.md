### Turn Summary
Extended compare_sim_lines_params.py with `--default-sim-lines-nepochs` flag and `get_training_config_snapshot()` helper to capture nepochs, batch_size, and probe.trainable in the diff.
Discovered **critical finding:** nepochs diverges 60 (legacy) vs 5 (sim_lines) — a 12× training length reduction that plausibly explains the amplitude collapse.
Next: schedule gs2_ideal retrain with nepochs=60 to verify whether training length alone closes the amplitude gap (H-NEPOCHS hypothesis).
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/ (hyperparam_diff.md, analysis.md, pytest_cli_smoke.log)

---

### Turn Summary (prior: Galph scope)
Scoped the Phase D3 hyperparameter audit by updating docs/fix_plan + the initiative summary, reserving the 2026-01-20T133807Z hub, and defining the CLI changes needed to capture sim_lines vs dose_experiments training knobs.
Rewrote input.md with the new Do Now so Ralph can extend compare_sim_lines_params.py, rerun the diff with `--default-sim-lines-nepochs 5`, and prove the CLI smoke selector.
Next: Ralph implements the CLI updates, regenerates hyperparameter_diff artifacts, and archives the pytest log under the new hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T133807Z/ (summary.md)
