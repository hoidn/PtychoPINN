### Turn Summary (2026-01-20T232800Z)
Implemented dataset-derived intensity scale telemetry in `run_phase_c2_scenario.py` to compare against the fallback formula per `specs/spec-ptycho-core.md §Normalization Invariants`.
The key finding: dataset-derived scale=577.74 vs fallback=988.21 (ratio=0.585), revealing the pipeline uses the fallback instead of dataset-derived mode, causing a ~1.7× scale mismatch.
Next: investigate where `intensity_scale` is computed and determine whether switching to dataset-derived mode corrects the amplitude bias.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/ (gs2_ideal/*, gs2_ideal_nepochs60/*, bias_summary.md, pytest_cli_smoke.log)

---

### Turn Summary (planning)
Scoped the next D4 increment around dataset-derived vs fallback intensity-scale telemetry so we can prove or disprove a gain mismatch before editing normalization code.
Updated docs/fix_plan.md, the implementation plan (D4a checklist), the initiative summary, and input.md with the new runner/analyzer work plus gs2 baseline + 60-epoch reruns under `.../2026-01-20T231745Z/`.
Next: Ralph lands the dataset-scale instrumentation, reruns both gs2 profiles, regenerates `bias_summary.*`, and archives the CLI smoke pytest log in the new hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/
