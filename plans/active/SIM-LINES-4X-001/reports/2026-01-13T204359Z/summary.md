### Turn Summary
Updated gs2 SIM-LINES-4X runners to pass explicit probe_scale values, matching the new ideal/custom scale targets.
Reran all four scenarios with probe_scale=10.0 (ideal) and probe_scale=4.0 (custom), writing outputs to `.artifacts/sim_lines_4x_probe_scale_2026-01-13T204359Z`.
Captured run logs and ruff evidence; stitch warnings still appear in logs but reconstruction PNGs were emitted.
Artifacts: plans/active/SIM-LINES-4X-001/reports/2026-01-13T204359Z/ (ruff_check.log, run_gs2_custom_probe.log)
