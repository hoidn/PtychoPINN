### Turn Summary
Scoped the next D4 increment around dataset-derived vs fallback intensity-scale telemetry so we can prove or disprove a gain mismatch before editing normalization code.
Updated docs/fix_plan.md, the implementation plan (D4a checklist), the initiative summary, and input.md with the new runner/analyzer work plus gs2 baseline + 60-epoch reruns under `.../2026-01-20T231745Z/`.
Next: Ralph lands the dataset-scale instrumentation, reruns both gs2 profiles, regenerates `bias_summary.*`, and archives the CLI smoke pytest log in the new hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/
