### Turn Summary
Scoped the D2b CLI so we can capture dose_experiments-style normalization stats (RawData→container + intensity scales) without rerunning the full sim_lines pipeline.
Drafted the Do Now covering the new capture script, execution commands, doc updates, and the pytest guard so Ralph can move straight into implementation.
Next: implement `capture_dose_normalization.py`, run it for the dose_legacy_gs2 profile, update the plan/fix-plan entries, and archive the CLI + pytest logs under this hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T124212Z/ (input.md, summary.md)
