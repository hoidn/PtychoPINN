### Turn Summary
Modularized the evaluation alignment pipeline with a shared helper that matches compare_models registration behavior.
Updated the sim_lines_4x evaluation script to use the shared helper and documented the full alignment+registration pattern.
Artifacts: ptycho/image/cropping.py, scripts/studies/sim_lines_4x/evaluate_metrics.py, docs/DEVELOPER_GUIDE.md


### Turn Summary
Aligned evaluation PNGs with the same fine-scale registration used by compare_models, eliminating the visible shift.
Updated the SIM-LINES-4X evaluation script to apply registration before metrics and PNG export, and refreshed the metrics JSON/table.
Artifacts: .artifacts/sim_lines_4x_metrics/2026-01-13T194447Z/, outputs/sim_lines_4x_rerun_20260113T025238Z/{gs1_ideal,gs1_custom,gs2_ideal,gs2_custom}/eval_outputs_nsamples1000_seed7/


### Turn Summary
Regenerated evaluation PNGs for all four SIM-LINES-4X scenarios using nsamples=1000/seed=7 and scan-coordinate alignment.
Saved aligned reconstruction and ground-truth amplitude/phase images under each scenario's eval_outputs directory.
Artifacts: outputs/sim_lines_4x_rerun_20260113T025238Z/{gs1_ideal,gs1_custom,gs2_ideal,gs2_custom}/eval_outputs_nsamples1000_seed7/


### Turn Summary
Recomputed SIM-LINES-4X metrics using the true simulated object as ground truth and random test subsampling (nsamples=1000, seed=7).
Added a reproducible evaluation script and refreshed the paper metrics table + JSON notes to reflect scan-coordinate alignment.
Artifacts: .artifacts/sim_lines_4x_metrics/2026-01-13T191658Z/, .artifacts/sim_lines_4x_metrics/2026-01-13T191911Z/, .artifacts/sim_lines_4x_metrics/2026-01-13T192119Z/, .artifacts/sim_lines_4x_metrics/2026-01-13T192409Z/


### Turn Summary
Updated sim_lines_4x inference presentation: square crop + nonzero vmin scaling for amplitude/phase colormaps.
Regenerated inference figures for gs1/gs2 and ideal/custom scenarios in outputs/sim_lines_4x_rerun_20260113T025238Z.
Artifacts: outputs/sim_lines_4x_rerun_20260113T025238Z/{gs1_ideal,gs1_custom,gs2_ideal,gs2_custom}/inference_outputs/


### Turn Summary
Note: Phase C validation should be rerun after the SYNTH-HELPERS-001 refactor updates sim_lines_4x helpers.
Adjusted SIM-LINES-4X counts so gs2 uses 8000 total images with 4000/4000 splits and group_count=1000.
Reran gs2 ideal and integration scenarios; outputs saved under .artifacts/sim_lines_4x with logs captured.
Stitch warnings appeared during gs2 inference, but reconstructions were still written.
Next: rerun gs1 scenarios if you want refreshed outputs with the same scaling rules.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-11T083629Z/ (run_gs2_ideal.log, run_gs2_integration_probe.log)


### Turn Summary
Implemented SIM-LINES-4X pipeline and four scenario runner scripts, plus README/docs entries for the new workflow.
Verified static analysis and integration workflow with ruff and pytest, recording logs in the report directory.
Next: execute the four scenarios to generate bundles and reconstruction images for validation.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-11T081911Z/ (ruff_check.log, pytest_integration.log)
### Turn Summary
Drafted a SIM-LINES-4X quantitative metrics table for the paper with a script-backed LaTeX generator and JSON source data.
Inserted the table input into the Overlap-Free Reconstruction section of the manuscript for a draft quantitative comparison.
Artifacts: paper/tables/sim_lines_4x_metrics.tex, paper/tables/scripts/generate_sim_lines_4x_metrics.py, paper/data/sim_lines_4x_metrics.json
### Turn Summary
Refactored nongrid simulation caching into a reusable memoization decorator for RawData results.
Updated `simulate_nongrid_raw_data` to use the shared decorator with `.artifacts/synthetic_helpers/cache` as the default store.
Artifacts: .artifacts/synthetic_helpers/cache/ (generated on next run)
### Turn Summary
Validated the new nongrid cache decorator with a two-call smoke test (cache hit confirmed by unchanged mtime).
Ran synthetic helper unit and CLI smoke tests to guard the caching change.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-13T193346Z/ (pytest_synthetic_helpers.log, pytest_synthetic_helpers_cli.log)
### Turn Summary
Updated gs2 SIM-LINES-4X runners to pass explicit probe_scale values, matching the new ideal/custom scale targets.
Reran all four scenarios with probe_scale=10.0 (ideal) and probe_scale=4.0 (custom), writing outputs to `.artifacts/sim_lines_4x_probe_scale_2026-01-13T204359Z`.
Captured run logs and ruff evidence; stitch warnings still appear in logs but reconstruction PNGs were emitted.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-13T204359Z/ (ruff_check.log, run_gs2_custom_probe.log)
### Turn Summary
Reran SIM-LINES-4X gs2 ideal/custom cases for 50 epochs and wrote outputs to `.artifacts/sim_lines_4x_probe_scale_2026-01-13T205123Z_gs2_50`.
gs2_ideal training hit NaN losses early while gs2_custom completed after a longer rerun.
Next: decide whether to rerun gs2_ideal with diagnostics or adjusted settings to eliminate NaNs.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-13T205123Z/ (run_gs2_ideal_50epochs.log, run_gs2_custom_probe_50epochs.log)
### Turn Summary
Added a sim_seed override in the SIM-LINES-4X pipeline and gs2_ideal runner to rerun gs2_ideal with different simulation seeds without touching core modules.
Reran gs2_ideal with seeds 37, 53, 101, 149, and 211, aborting each run once NaNs appeared; no seed completed cleanly.
Next: decide whether to keep trying new seeds or adjust training settings to prevent NaNs.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-13T212359Z/ (ruff_check.log, run_gs2_ideal_seed37.log, run_gs2_ideal_seed211_rerun.log)
### Turn Summary
Added NaN-safe handling in the gs2 ideal probe_scale sweep so invalid metrics are sanitized and registration failures can fall back cleanly.
Ran a 20-epoch sweep for probe_scale 2/4/6/8/10; best amplitude SSIM is 0.2493 at probe_scale=6.0, while scales 8 and 10 produced NaN recon amplitudes (metrics recorded as null).
Next: confirm whether to rerun the NaN scales with adjusted seeds/settings or accept the current sweep.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-13T220032Z/ (run_probe_scale_sweep_20epochs.log, ruff_check.log), .artifacts/sim_lines_4x_probe_scale_sweep_2026-01-13T220032Z/probe_scale_sweep.json

