# PtychoPINN Fix Plan Ledger (Condensed)

**Last Updated:** 2026-01-16 (Phase B4 reassembly instrumentation scoped)
**Active Focus:** DEBUG-SIM-LINES-DOSE-001 — Isolate sim_lines_4x vs dose_experiments discrepancy

---

**Housekeeping Notes:**
- Full ledger snapshot archived at `docs/archive/2026-01-13_fix_plan_archive.md`
- Full Attempts History archived in `docs/fix_plan_archive.md` (snapshot 2026-01-06)
- Earlier snapshots: `docs/archive/2025-11-06_fix_plan_archive.md`, `docs/archive/2025-10-17_fix_plan_archive.md`, `docs/archive/2025-10-20_fix_plan_archive.md`
- Each initiative has a working plan at `plans/active/<ID>/implementation.md` and reports under `plans/active/<ID>/reports/`

---

## Active / Pending Initiatives

### [DEBUG-SIM-LINES-DOSE-001] Isolate sim_lines_4x vs dose_experiments discrepancy
- Depends on: None
- Priority: **Critical** (Highest Priority)
- Status: in_progress — Phase B instrumentation (B4 reassembly limits)
- Owner/Date: Codex/2026-01-13
- Working Plan: `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`
- Summary: `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md`
- Reports Hub: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/`
- Spec Owner: `docs/specs/spec-ptycho-workflow.md`
- Test Strategy: `plans/active/DEBUG-SIM-LINES-DOSE-001/test_strategy.md`
- Goals:
  - Identify whether the sim_lines_4x failure stems from a core regression, nongrid pipeline differences, or a workflow/config mismatch.
  - Produce a minimal repro that isolates grid vs nongrid and probe normalization effects.
  - Apply a targeted fix and verify success via visual inspection if metrics are unavailable.
- Exit Criteria:
  - A/B results captured for grid vs nongrid, probe normalization, and grouping parameters.
  - Root-cause statement with evidence (logs + params snapshot + artifacts).
  - Targeted fix or workflow change applied, with recon success and no NaNs.
  - Visual inspection success gate satisfied if metrics are unavailable.
- Attempts History:
  - *2026-01-13T000000Z:* Drafted phased debugging plan, summary, and test strategy. Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md`, `plans/active/DEBUG-SIM-LINES-DOSE-001/summary.md`, `plans/active/DEBUG-SIM-LINES-DOSE-001/test_strategy.md`.
  - *2026-01-15T235900Z:* Reactivated focus, set Phase A evidence capture Do Now, and opened new artifacts hub. Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-15T235900Z/`.
  - *2026-01-16T000353Z:* Reframed Phase A A0/A1/A3 handoff to build `collect_sim_lines_4x_params.py`, inventory `dose_experiments` defaults, and run the pipeline import smoke test. Artifacts hub: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/`.
  - *2026-01-16T002700Z:* Implemented `scripts/tools/collect_sim_lines_4x_params.py` (metadata-only snapshot CLI), captured the JSON snapshot, recorded the legacy `dose_experiments` tree + parameter script, and reran the sim_lines pipeline import smoke test. Metrics: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (pass). Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T000353Z/{sim_lines_4x_params_snapshot.json,dose_experiments_tree.txt,dose_experiments_param_scan.md,pytest_sim_lines_pipeline_import.log}`. Next Actions: Compare sim_lines snapshot vs dose_experiments defaults (Phase A4) and plan the differential experiments.
  - *2026-01-16T003217Z:* Reviewed the captured artifacts, ticked A1/A3 in the plan, and authored the A4 comparison Do Now plus new artifacts hub (`plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/`) so Ralph can implement the diff script with fresh pytest evidence.
  - *2026-01-16T013500Z:* Implemented `bin/compare_sim_lines_params.py`, generated the Markdown + JSON diff artifacts for all four scenarios, and reran the synthetic helpers CLI smoke test to guard imports.
    - Metrics: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (pass)
    - Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T003217Z/{comparison_draft.md,comparison_diff.json,pytest_sim_lines_pipeline_import.log}`
    - Next Actions: Use the diff to scope the Phase B differential experiments (grid vs nongrid, probe normalization) or flag gaps if additional parameters need capture.
  - *2026-01-16T020000Z:* Reviewed the comparison diff, marked Phase A4 complete in the working plan, and scoped Phase B1 instrumentation to capture grouping stats for both the legacy (dose_experiments) and sim_lines parameter regimes. Created new artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/` for the upcoming grouping summaries and refreshed the Do Now with the bin script plan plus pytest guard.
  - *2026-01-16T005400Z:* Added `bin/grouping_summary.py` under the initiative, captured grouping summaries for the SIM-LINES defaults and the legacy gridsize=2 dose constraints (including failure diagnostics), and reran the synthetic helpers CLI smoke guard.
    - Metrics: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
    - Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T020000Z/{grouping_sim_lines_default.json,grouping_sim_lines_default.md,grouping_dose_experiments_legacy.json,grouping_dose_experiments_legacy.md,pytest_sim_lines_pipeline_import.log}`
    - Next Actions: Mine the grouping stats to plan the remainder of Phase B (probe normalization and grouping differentials) or flag if additional parameter overrides are required.
  - *2026-01-16T031700Z:* Implemented the plan-local `probe_normalization_report.py` CLI plus `plans/.../bin/__init__.py`, generated JSON/Markdown stats for gs1/gs2 × custom/ideal probes, captured the CLI log, and reran the synthetic helpers CLI smoke test.
    - Metrics: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
    - Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T031500Z/{probe_stats_gs1_custom.json,probe_stats_gs1_custom.md,probe_stats_gs1_ideal.json,probe_stats_gs1_ideal.md,probe_stats_gs2_custom.json,probe_stats_gs2_custom.md,probe_stats_gs2_ideal.json,probe_stats_gs2_ideal.md,probe_normalization_cli.log,pytest_sim_lines_pipeline_import.log}`
    - Next Actions: Compare the legacy vs sim_lines probe stats to decide if normalization explains the reconstruction gap or if grouping/reassembly experiments must proceed (Phase B3/B4).
  - *2026-01-16T041700Z:* Supervisor review confirmed the probe stats are numerically identical (max delta ≈5e-7), so normalization is no longer a suspect. Scoped Phase B3 to extend `bin/grouping_summary.py` with per-axis offset stats + nn-index ranges and to capture three runs (gs1 default, gs2 default, gs2 neighbor-count=1) plus the CLI smoke guard under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/`.
  - *2026-01-16T043500Z:* Extended `bin/grouping_summary.py` with overall mean/std reporting plus per-axis coordinate stats (when the penultimate dimension is 2) and nn-index min/max telemetry, regenerated the gs1 default, gs2 default, and gs2 `neighbor-count=1` grouping summaries, and archived the combined CLI + pytest logs under the Phase B3 hub.
    - Metrics: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v`
    - Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T041700Z/{grouping_gs1_custom_default.json,grouping_gs1_custom_default.md,grouping_gs2_custom_default.json,grouping_gs2_custom_default.md,grouping_gs2_custom_neighbor1.json,grouping_gs2_custom_neighbor1.md,grouping_cli.log,pytest_sim_lines_pipeline_import.log}`
    - Next Actions: Analyze the richer per-axis telemetry to decide whether B4 needs additional grouping probes or if we can pivot directly to the reassembly experiments.
  - *2026-01-16T050500Z:* Reviewed the B3 telemetry (coords offsets up to ~382 px on gs2) and compared it against the legacy padded-size math (`get_padded_size()` ≈ 78 px when `offset` remains 4 and `max_position_jitter` remains 10), confirming reassembly canvas under-allocation is the likely regression. Scoped Phase B4 around a `reassembly_limits_report.py` helper that contrasts observed offsets vs padded-size requirements and runs a sum-preservation probe via `reassemble_whole_object()`. Opened artifacts hub `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/` for the next evidence batch and refreshed the Do Now accordingly.
  - *2026-01-16T053600Z:* Added `bin/reassembly_limits_report.py`, generated JSON/Markdown + CLI evidence for `gs1_custom` and `gs2_custom`, and proved that the observed max offsets (~382 px) demand canvases ≥828–831 px while the legacy padded size stays at 74/78 px (loss fractions ≥94%). Pytest guard reran clean.
    - Metrics: `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py::test_sim_lines_pipeline_import_smoke -v` (pass)
    - Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T050500Z/{reassembly_cli.log,reassembly_gs1_custom.json,reassembly_gs1_custom.md,reassembly_gs2_custom.json,reassembly_gs2_custom.md,pytest_sim_lines_pipeline_import.log}`
    - Next Actions: Feed the reassembly deltas into Phase C to patch `get_padded_size()`/canvas sizing or open a dedicated backlog item if the fix crosses initiatives.
  - *2026-01-16T060156Z:* Supervisor planning loop to pivot into Phase C: mined the B4 telemetry, marked Phase B complete in the implementation plan, and scoped C1 around updating `create_ptycho_data_container()` (plus a targeted pytest) so grouped offsets automatically expand `max_position_jitter`/padded size before training/inference. Refreshed `plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}` and `input.md` with the new Do Now, no new production artifacts yet. Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060156Z/`.
  - *2026-01-20T011212Z:* Implemented `_update_max_position_jitter_from_offsets()` with padded-size parity handling, wired it into `create_ptycho_data_container`, and updated `reassembly_limits_report.py` to apply the helper while basing the required canvas on `coords_offsets`. Added the regression pytest, refreshed test docs, and confirmed SIM-LINES reassembly telemetry now reports `fits_canvas=True` with zero loss for gs1/gs2 custom runs.
    - Metrics: `ruff check ptycho/workflows/components.py plans/active/DEBUG-SIM-LINES-DOSE-001/bin/reassembly_limits_report.py tests/test_workflow_components.py`, `pytest --collect-only tests/test_workflow_components.py -q`, `pytest tests/test_workflow_components.py::TestCreatePtychoDataContainer::test_updates_max_position_jitter -v`, `pytest -v -m integration`
    - Artifacts: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060900Z/{git_diff.txt,ruff_check.log,pytest_collect_workflow_components.log,pytest_workflow_components.log,pytest_integration.log,reassembly_cli.log,reassembly_gs1_custom.json,reassembly_gs1_custom.md,reassembly_gs2_custom.json,reassembly_gs2_custom.md}`
    - Next Actions: Run the Phase C2 gs1/gs2 ideal telemetry (if required) or proceed to the inference smoke validation once the padded-size update is accepted.

### [REFACTOR-MEMOIZE-CORE-001] Move RawData memoization decorator into core module
- Depends on: None
- Priority: Low
- Status: done — Phase C docs/tests landed; ready for archive after a short soak
- Owner/Date: TBD/2026-01-13
- Working Plan: `plans/active/REFACTOR-MEMOIZE-CORE-001/implementation.md`
- Summary: `plans/active/REFACTOR-MEMOIZE-CORE-001/summary.md`
- Reports Hub: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/`
- Spec Owner: `docs/architecture.md`
- Test Strategy: Inline test annotations (refactor only; reuse existing tests)
- Goals:
  - Move `memoize_raw_data` from `scripts/simulation/cache_utils.py` into a core module under `ptycho/`.
  - Preserve cache hashing and default cache paths used by synthetic helpers.
  - Keep script imports working via direct update or a thin shim.
- Exit Criteria:
  - Core module provides `memoize_raw_data` with unchanged behavior.
  - Synthetic helpers use the core module; shim or removal completed without regressions.
  - Existing synthetic helper tests pass and logs archived.
- Attempts History:
  - *2026-01-13T202358Z:* Drafted implementation plan and initialized initiative summary. Artifacts: `plans/active/REFACTOR-MEMOIZE-CORE-001/implementation.md`, `plans/active/REFACTOR-MEMOIZE-CORE-001/summary.md`.
  - *2026-01-15T225850Z:* Phase A inventory + compatibility design completed; handed off Phase B move/shim work with pytest coverage instructions. Artifacts: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/`
  - *2026-01-15T231710Z:* Added `ptycho/cache.py` with the memoize helpers, updated synthetic_helpers to import it, and converted `scripts/simulation/cache_utils.py` into a DeprecationWarning shim. Tests: `pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v`, `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v`. Artifacts: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/`
  - *2026-01-15T232107Z:* Confirmed Phase B landed in commit `d29efc91` and staged Phase C cleanup: refresh docs (`docs/index.md`, `scripts/simulation/README.md`), rerun the two synthetic helper selectors, and archive logs under `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/`.
  - *2026-01-15T233050Z:* Documented the new `ptycho/cache.py` core helper in `docs/index.md`, refreshed `scripts/simulation/README.md` with cache-root/override guidance, and captured the required pytest evidence (`pytest --collect-only tests/scripts/test_synthetic_helpers.py -q`, `pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v`, `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v`). Artifacts: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/pytest_collect.log`, `.../pytest_synthetic_helpers.log`, `.../pytest_cli_smoke.log`.
  - *2026-01-15T233622Z:* Verified Phase C evidence (docs updated, selectors rerun), checked plan checkboxes, and logged completion so the initiative can be archived after the soak window. Artifacts: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/`, `plans/active/REFACTOR-MEMOIZE-CORE-001/implementation.md`

### [PARALLEL-API-INFERENCE] Programmatic TF/PyTorch API parity
- Depends on: None
- Priority: Medium
- Status: pending — paused while DEBUG-SIM-LINES-DOSE-001 is active
- Owner/Date: TBD/2026-01-09
- Working Plan: `plans/active/PARALLEL-API-INFERENCE/plan.md`
- Summary: `plans/active/PARALLEL-API-INFERENCE/summary.md`
- Reports Hub: `plans/active/PARALLEL-API-INFERENCE/reports/`
- Spec Owner: `specs/ptychodus_api_spec.md`
- Test Strategy: `tests/scripts/test_tf_inference_helper.py`, `tests/scripts/test_api_demo.py`
- Goals:
  - Provide a single programmatic entry point that can train + infer via TensorFlow or PyTorch without shell wrappers.
  - Extract reusable TensorFlow inference helper so `_run_tf_inference_and_reconstruct()` mirrors the PyTorch helper.
  - Update `scripts/pytorch_api_demo.py` to exercise both backends and add smoke tests.
- Exit Criteria:
  - `_run_tf_inference_and_reconstruct()` helper exposed (done) and consumed by new programmatic flows.
  - `scripts/pytorch_api_demo.py` drives both backends, uses core helpers (TF + PyTorch), and captures outputs under `tmp/api_demo/<backend>/`.
  - `tests/scripts/test_api_demo.py` exercises imports/signatures plus marked slow end-to-end runs for both backends; helper tests continue to pass.
- Attempts History:
  - *2026-01-09T010000Z:* Completed exploration + extraction design for TF helper. Artifacts: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T010000Z/extraction_design.md`.
  - *2026-01-09T020000Z:* Implemented `_run_tf_inference_and_reconstruct()` and `extract_ground_truth()`, deprecated `perform_inference`, and added 7 regression tests + integration workflow run (all green). Artifacts: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T020000Z/`.
  - *2026-01-09T030000Z:* Reviewed Task 1 results and scoped Task 2-3 (demo script + smoke test). Artifacts: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T030000Z/`.
  - *2026-01-15T225312Z:* Added initial smoke tests for `scripts/pytorch_api_demo.py` (import + signature) and reran TF helper regression suite; slow execution tests still deselected pending demo parity. Artifacts: `plans/active/PARALLEL-API-INFERENCE/reports/2026-01-15T225312Z/pytest_collect.log`, `pytest_tf_helper_regression.log`, `pytest_api_demo.log`.

### [ORCH-ROUTER-001] Router prompt + orchestration dispatch layer
- Depends on: None
- Priority: Medium
- Status: pending
- Owner/Date: Codex/2026-01-20
- Working Plan: `plans/active/ORCH-ROUTER-001/implementation.md`
- Summary: `plans/active/ORCH-ROUTER-001/summary.md`
- Reports Hub: `.artifacts/orch-router-001/`
- Spec Owner: `scripts/orchestration/README.md`
- Test Strategy: `plans/active/ORCH-ROUTER-001/test_strategy.md`
- Goals:
  - Add a router loop with deterministic routing + optional prompt override.
  - Preserve sync semantics while enforcing allowlist and crash behavior.
- Exit Criteria:
  - Deterministic routing + router override verified.
  - Allowlist and failure behavior documented and tested.
  - Logging/state annotations captured as specified in the plan.
- Attempts History:
  - *2026-01-20T011707Z:* Drafted implementation plan, test strategy, and summary. Artifacts: `plans/active/ORCH-ROUTER-001/{implementation.md,test_strategy.md,summary.md}`.
  - *2026-01-20T012145Z:* Refined the plan to locate the routing function in `scripts/orchestration/router.py`, document YAML-as-parameters-only, and clarify review cadence/state.json decisions. Artifacts: `plans/active/ORCH-ROUTER-001/{implementation.md,summary.md}`.
  - *2026-01-20T012303Z:* Clarified routing contract to use review cadence every N iterations with actor gating and router precedence, and constrained state.json persistence to the last selected prompt only. Artifacts: `plans/active/ORCH-ROUTER-001/{implementation.md,summary.md}`.
  - *2026-01-20T012542Z:* Drafted the Phase A routing contract artifact and marked A0-A2 complete in the plan (routing contract + state.json field + test strategy linkage). Artifacts: `plans/active/ORCH-ROUTER-001/{routing_contract.md,implementation.md,summary.md}`.
  - *2026-01-20T012735Z:* Expanded plan documentation steps to include docs/index.md updates in Phase A/C. Artifacts: `plans/active/ORCH-ROUTER-001/{implementation.md,summary.md}`.
