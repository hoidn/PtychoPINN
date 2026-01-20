### Turn Summary
Reopened Phase D1 after the reviewer showed the loss diff misread dose_experiments' conditional MAE assignments; plan, fix plan, and initiative summary now require capturing actual runtime cfg snapshots instead of assuming an inversion.
Documented the D1a–D1c tasks (capture both loss_fn branches, fix the comparison CLI, rerun the diff) and rewrote input.md so Ralph stubs ptycho.params, records both cfg snapshots, emits the new artifacts, and runs the CLI pytest guard.
Next: Ralph updates `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/compare_sim_lines_params.py`, regenerates the diff under the 2026-01-20T112029Z hub, and reports whether H-LOSS-WEIGHT still holds once the real loss weights are known.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T112029Z/ (summary.md)
