Summary: Prototype the Phase C4e amplitude-rescaling hook so we can prove whether applying a deterministic scalar (bundle `intensity_scale` or least-squares) eliminates the ≈12× amplitude drop before touching shared workflows.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy (Phase C4e rescale prototype)
Branch: paper
Mapped tests: pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/

Do Now:
- Implement: scripts/studies/sim_lines_4x/pipeline.py::run_inference and plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py::main — add a `--prediction-scale-source` flag (`none`, `recorded`, `least_squares`), compute the requested scalar (recorded intensity_scale or best-fit vs ground truth), multiply stitched amplitude/phase by that scalar before saving, and persist the scalar + mode in `run_metadata` and analyzer inputs; rerun gs1_ideal + gs2_ideal with `least_squares` to capture rescaled outputs under the new hub.
- Validate:
  1. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs1_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs1_ideal --prediction-scale-source least_squares --group-limit 64 --nepochs 5 > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs1_ideal_runner.log
  2. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/run_phase_c2_scenario.py --scenario gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs2_ideal --prediction-scale-source least_squares --group-limit 64 --nepochs 5 > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs2_ideal_runner.log
  3. python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/analyze_intensity_bias.py --scenario gs1_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs1_ideal --scenario gs2_ideal=plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/gs2_ideal --output-dir plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z > plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/analyze_intensity_bias.log
  4. pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v | tee plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/pytest_cli_smoke.log
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/

How-To Map:
1. export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
2. Add `--prediction-scale-source {none,recorded,least_squares}` to the runner CLI; when `recorded`, use the bundle/legacy `intensity_scale` recorded in `intensity_info`, and when `least_squares`, compute the scalar via Σ(pred·truth)/Σ(pred²) using the ground-truth amplitude before writing amplitude/phase; persist the scalar + mode under `run_metadata['prediction_scale']` and `stats.json`.
3. Mirror the same option in scripts/studies/sim_lines_4x/pipeline.py::run_inference (pass a scalar through to downstream consumers) so future workflow changes can share the hook; default to `none` unless explicitly requested.
4. Ensure amplitude/phase PNGs, `.npy`, and stats that land under each scenario directory clearly state whether a scale was applied; keep the pre-scale arrays available (e.g., `amplitude_unscaled.npy`) so analyzer comparisons remain reproducible.
5. After rebuilding, rerun gs1_ideal and gs2_ideal with `--prediction-scale-source least_squares`, capturing CLI output and storing the runner logs plus training/inference metadata under the new hub.
6. Regenerate bias summaries via `bin/analyze_intensity_bias.py` pointing to the new gs1_ideal/gs2_ideal directories so the scaling section shows ΔMAE/ΔRMSE after applying the scalar, and archive the log in the hub.
7. Re-run the synthetic helpers CLI smoke selector to guard the SIM-LINES plan-local scripts and stash the pytest log in the same artifacts directory.

Pitfalls To Avoid:
- Do not overwrite existing amplitude files without also storing the unscaled version (downstream comparisons still rely on baseline tensors).
- Keep the scalar metadata JSON-serializable (cast numpy types to float) and include the mode so analyzer consumers know which path was used.
- Guard against zero-division in the least-squares solver; skip rescaling when Σ(pred²)=0 and emit a warning instead of crashing.
- Maintain CONFIG-001 hygiene: runner changes must continue calling `update_legacy_dict` before touching legacy modules.
- Respect normalization boundaries from NORMALIZATION-001 — the new hook is for display/evaluation only; never feed rescaled tensors back into training.
- Update both runner and pipeline code paths so evidence stays consistent regardless of entrypoint.
- Keep logs lightweight; capture CLI output to files under the artifacts hub instead of printing megabytes to stdout.
- Leave `AUTHORITATIVE_CMDS_DOC` set for every command invocation.
- Avoid modifying stable physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Ensure pytest output is archived even on failure for reviewer traceability.

If Blocked:
- Capture the failing command/traceback in `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T150500Z/blocker.log`, mark the blocker + rationale inside docs/fix_plan.md Attempts and the initiative summary, then stop after updating galph_memory with `<status>blocked</status>` so we can reassess before touching production modules.

Findings Applied (Mandatory):
- CONFIG-001 (docs/findings.md:14) — keep legacy params synchronized before running runner/pipeline helpers.
- NORMALIZATION-001 (docs/findings.md:22) — keep physics/statistical/display scaling separated; the new hook lives in the display/evaluation layer only.
- POLICY-001 (docs/findings.md:12) — runner changes must stay compatible with the PyTorch-required environment and avoid interpreter wrappers.

Pointers:
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:199 — C4e checklist describing the rescaling prototype expectations.
- specs/spec-ptycho-core.md:86 — Normalization invariant stating stitched canvases must reflect the same scaling used during training/inference.
- docs/DATA_NORMALIZATION_GUIDE.md:9 — delineates physics vs statistical vs display scaling (justifies isolating the new hook to evaluation output).
- docs/TESTING_GUIDE.md:55 — CLI smoke selector requirements for synthetic helper scripts.

Next Up (optional):
1. If rescaling proves insufficient, plan Phase C4f to inspect loader normalization math before touching shared workflows.
2. If gs2_ideal still emits NaNs after rescaling, instrument the runner to dump per-epoch loss tables for the NaN channels and compare against gs1_ideal.

Doc Sync Plan (Conditional): Not needed (selectors unchanged).
Mapped Tests Guardrail: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` collects >0 tests; keep it green.
Normative Math/Physics: Cite specs/spec-ptycho-core.md §Normalization Invariants verbatim when describing intensity-scaling behavior.
