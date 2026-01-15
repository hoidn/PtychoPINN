### Turn Summary
Added NaN-safe handling in the gs2 ideal probe_scale sweep so invalid metrics are sanitized and registration failures can fall back cleanly.
Ran a 20-epoch sweep for probe_scale 2/4/6/8/10; best amplitude SSIM is 0.2493 at probe_scale=6.0, while scales 8 and 10 produced NaN recon amplitudes (metrics recorded as null).
Next: confirm whether to rerun the NaN scales with adjusted seeds/settings or accept the current sweep.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-13T220032Z/ (run_probe_scale_sweep_20epochs.log, ruff_check.log), .artifacts/sim_lines_4x_probe_scale_sweep_2026-01-13T220032Z/probe_scale_sweep.json
