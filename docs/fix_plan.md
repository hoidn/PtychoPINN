# PtychoPINN Fix Plan Ledger (Condensed)

**Last Updated:** 2026-01-15 (pivoted back to DEBUG-SIM-LINES-DOSE-001 Phase A evidence capture)
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
- Status: in_progress — Phase A evidence capture kickoff
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
