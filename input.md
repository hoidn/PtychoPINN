Summary: Update the workflow helper so grouped datasets inflate `max_position_jitter`/padded size from actual offsets, then prove the fix via pytest and refreshed reassembly telemetry.
Focus: DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy
Branch: paper
Mapped tests: pytest tests/test_workflow_components.py::TestCreatePtychoDataContainer::test_updates_max_position_jitter -v
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060900Z/

Do Now (hard validity contract)
- Implement: ptycho/workflows/components.py::create_ptycho_data_container — derive `max_position_jitter` from grouped `coords_offsets` (train/test aware) so `get_padded_size()` satisfies the spec requirement before reassembly, and plumb the helper into the existing plan-local CLI for verification.
- Pytest: pytest tests/test_workflow_components.py::TestCreatePtychoDataContainer::test_updates_max_position_jitter -v
- Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060900Z/{git_diff.txt,pytest_workflow_components.log,reassembly_cli.log,reassembly_gs1_custom.json,reassembly_gs1_custom.md,reassembly_gs2_custom.json,reassembly_gs2_custom.md}

How-To Map
1. Edit `ptycho/workflows/components.py` inside `create_ptycho_data_container()`: after `dataset = data.generate_grouped_data(...)`, compute per-axis max absolute offsets from `dataset['coords_offsets']`, translate that to a required canvas (`required = math.ceil(config.model.N + 2 * max_abs)`), compare against `params.cfg['max_position_jitter']`, and bump jitter so the derived `get_padded_size()` equals `required` (only increase, never shrink). Log the adjustment for traceability.
2. Export overrides into a helper (e.g., `_update_max_position_jitter_from_offsets(dataset, config)`) so the logic can be unit tested without RawData I/O; ensure both train/test invocations keep the maximum jitter observed in the process.
3. Capture `git diff ptycho/workflows/components.py > "$ARTIFACT_DIR/git_diff.txt"` for review once edits are done.
4. Run `pytest tests/test_workflow_components.py::TestCreatePtychoDataContainer::test_updates_max_position_jitter -v | tee "$ARTIFACT_DIR/pytest_workflow_components.log"` to prove the helper inflates jitter/padded size.
5. Rebuild the SIM-LINES evidence with the updated workflow:
   - `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py --scenario gs1_custom --group-limit 64 --output-json "$ARTIFACT_DIR/reassembly_gs1_custom.json" --output-markdown "$ARTIFACT_DIR/reassembly_gs1_custom.md" | tee "$ARTIFACT_DIR/reassembly_cli.log"`
   - `python plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py --scenario gs2_custom --group-limit 64 --output-json "$ARTIFACT_DIR/reassembly_gs2_custom.json" --output-markdown "$ARTIFACT_DIR/reassembly_gs2_custom.md" | tee -a "$ARTIFACT_DIR/reassembly_cli.log"`
   Confirm the `fits_canvas` flag is True and loss ratios drop ≈0% for both scenarios.

Pitfalls To Avoid
- Do not mutate core physics modules; keep changes inside workflows + plan-local tools.
- Preserve CONFIG-001 ordering: `update_legacy_dict` must run before sampling/offset analysis.
- Only increase `max_position_jitter`; downstream code may rely on prior minimums.
- Keep helper CPU-friendly—no TensorFlow ops or GPU requirements inside workflow components.
- Ensure `dataset['coords_offsets']` exists; raise a clear error if the snapshot is missing offsets.
- Don’t bypass the pytest guard; evidence must show the new selector passes.
- Re-run the SIM-LINES CLI with identical seeds so comparisons remain apples-to-apples.
- Capture every log (pytest + CLI) under the artifacts hub even when failures arise.
- Follow PYTHON-ENV-001: use `python`, not `python3.11` or virtualenv-specific binaries.
- When writing JSON/Markdown, keep the schema identical so downstream automation keeps working.

If Blocked
- If grouped data lacks offsets or jitter math fails, save the traceback to `$ARTIFACT_DIR/blocker.log`, update docs/fix_plan.md Attempts with the command + error, and set this focus to blocked until we know how to extract offsets reliably.

Findings Applied (Mandatory)
- CONFIG-001 — maintain the `update_legacy_dict` bridge before inspecting grouped offsets so gridsize/N stay authoritative.
- MODULE-SINGLETON-001 — workflow helpers must not resurrect legacy singletons; keep logic stateless and scoped per call.
- NORMALIZATION-001 — avoid touching intensity/probe scaling while adjusting jitter; this task only addresses geometry.
- BUG-TF-REASSEMBLE-001 — padding fixes must keep `_reassemble_position_batched` stable; treat any tf_helper regression as a hard stop.

Pointers
- plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md:120 — Phase C checklist + new C1 test requirement.
- docs/fix_plan.md:18 — Active focus metadata + latest Attempts entry (2026-01-16T060156Z).
- docs/specs/spec-ptycho-workflow.md:2 — Normative reassembly requirement (`M ≥ N + 2·max(|dx|,|dy|)`).
- plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py — CLI used for telemetry verification.

Next Up (optional)
- Once jitter inflates correctly, re-run a small gs2 inference smoke test to confirm reconstructions stitch without clipping before moving to visual verification.

Doc Sync Plan — N/A (existing selectors reused; no new pytest nodes introduced beyond the targeted workflow test).
Mapped Tests Guardrail — The workflow-components selector above already collects >0 tests; keep it green to satisfy the gate.
Normative Math/Physics — Cite `docs/specs/spec-ptycho-workflow.md §Reassembly Requirements` whenever referencing the `N + 2·max(|dx|,|dy|)` canvas rule.
