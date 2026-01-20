### Turn Summary (2026-01-20T234500Z) — D4b ROOT CAUSE IDENTIFIED
Traced `intensity_scale` computation through the codebase and identified root cause in `ptycho/train_pinn.py:calculate_intensity_scale()`.
The function uses the closed-form fallback (`sqrt(nphotons)/(N/2)`) instead of the dataset-derived scale, even though it receives the data container as input. Dead code at lines 173-175 contains an unimplemented TODO.
**Key Finding:** `diffsim.py:scale_nphotons()` correctly computes dataset-derived scale; `train_pinn.py:calculate_intensity_scale()` incorrectly uses fallback only. This 1.7× scale mismatch (988.21/577.74) violates `specs/spec-ptycho-core.md §Normalization Invariants`.
Next: Obtain approval to modify `ptycho/train_pinn.py` (core module), implement fix in D4c, rerun scenarios to verify amplitude bias reduction.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T231745Z/ (D4a telemetry + code analysis)

---

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
