Summary: Pull the TensorFlow/Keras intensity-scaler weights from the gs1_ideal/gs2_ideal checkpoints so we can explain the shared ~2.47 amplitude bias before touching workflow code.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
Branch: paper
Mapped tests:
- pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts:
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103000Z/

Do Now:
- DEBUG-SIM-LINES-DOSE-001 C3d — Implement: `plans/active/DEBUG-SIM-LINES-DOSE-001/bin/inspect_intensity_scaler.py::main` to load an existing scenario run directory, read its saved TensorFlow checkpoint + `params.cfg`, and dump the `IntensityScaler`/`IntensityScaler_inv` weights (gain/bias) plus the recorded `intensity_scale` into JSON + Markdown artifacts. Run it for `gs1_ideal` and `gs2_ideal` from `reports/2026-01-20T093000Z/`, write outputs under the new artifacts hub, and add summary pointers into the hub’s `summary.md`. Validate with `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`.

How-To Map:
1. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/inspect_intensity_scaler.py --scenario-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T093000Z/gs1_ideal --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103000Z/intensity_scaler_gs1_ideal.json --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103000Z/intensity_scaler_gs1_ideal.md
2. Repeat for gs2: AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md python .../inspect_intensity_scaler.py --scenario-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T093000Z/gs2_ideal --output-json plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103000Z/intensity_scaler_gs2_ideal.json --output-markdown plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103000Z/intensity_scaler_gs2_ideal.md
3. AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T103000Z/pytest_cli_smoke.log

Pitfalls To Avoid:
- Do not modify `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` yet; keep this loop inspector-only.
- Reuse the existing Phase A snapshot + run outputs; do NOT retrain the scenarios.
- Ensure inspector loads weights via TensorFlow’s saved model API so the bias/gain reflects the trained values (no manual recomputation).
- Capture both gain and offset for `IntensityScaler` and `IntensityScaler_inv`; omitting one makes bias math unusable.
- Keep computations device-neutral (CPU is fine) and don’t force GPU-only operations inside the inspector.
- Preserve the original `comparison_metrics` artifacts; write new files under the fresh hub only.
- Don’t derive fixes yet—this step is evidence collection only.
- Avoid hard-coding absolute paths; accept `--scenario-dir` so future hubs can reuse the tool.
- If TensorFlow warns about missing CUDA backends, note it but continue unless the inspector actually fails.
- Always prefix commands with AUTHORITATIVE_CMDS_DOC per CLAUDE instructions.

If Blocked:
- If the inspector cannot load the saved model (e.g., missing files), capture the stack trace + offending path under the new hub, log the blocker in docs/fix_plan.md Attempts, and skip straight to updating summary.md/input.md so we can triage in the next loop.

Findings Applied:
- CONFIG-001 — The inspector must respect `update_legacy_dict` ordering by reading params via the scenario’s saved `params.json` rather than reinitializing legacy modules.
- POLICY-001 — Use the repo’s default `python` executable (no custom wrappers) when running the inspector + pytest.
- NORMALIZATION-001 — Keep the analysis read-only so we do not accidentally mix physics/statistical/display scaling paths while extracting the intensity scaler state.

Pointers:
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:142-167 (Phase C3 checklist / new C3d item)
- docs/fix_plan.md:90-104 (Attempts history + latest hub reference)
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T093000Z/gs1_ideal/comparison_metrics.json (recorded amplitude bias stats)
- plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T093000Z/gs2_ideal/comparison_metrics.json (gs2 bias mirror for comparison)

Next Up:
- After the inspector proves where the offset is coming from, design the actual intensity-scaling fix (Phase C4) and rerun gs1_ideal/gs2_ideal to verify the bias disappears.
