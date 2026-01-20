### Turn Summary
Extended the plan-local Phase C2 runner with CLI arg serialization + artifact pointers, then reran gs1_ideal (512 imgs/256 groups) and gs2_ideal (256 imgs/128 groups) to produce fresh amplitude/phase npy+PNG dumps with run_metadata + stats under the 2026-01-20T061530Z hub.
Captured manual inspection notes, reran the jitter-aware reassembly_limits CLI for both ideal probes, and updated run_metadata to record the reduced loads that keep training stable (no NaNs).
Validated the synthetic helpers CLI import guard via pytest and refreshed pytest/reassembly logs in the same hub for review.
Next: fold these scenario results into the Phase C exit assessment or spin follow-up experiments if we still need gs2 custom telemetry.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T061530Z/ (gs1_ideal_runner.log, gs2_ideal_runner.log, reassembly_cli.log, pytest_cli_smoke.log)

### Turn Summary
Scoped the Phase C2 verification handoff: promoted the plan to Phase C status, opened the 2026-01-20T061530Z artifacts hub, and defined the `run_phase_c2_scenario.py` runner + CLI/test matrix for gs1_ideal and gs2_ideal evidence collection.
Captured the Do Now details in docs/fix_plan.md and input.md (runner implementation, scenario reruns, reassembly telemetry, visual notes, pytest guard) so Ralph can execute without touching production modules.
Next: Ralph builds the runner, executes both scenarios with PNG/NaN evidence, reruns the reassembly_limits CLI, and archives pytest output under the new hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T061530Z/ (planning notes)
