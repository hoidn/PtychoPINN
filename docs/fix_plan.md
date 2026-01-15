# PtychoPINN Fix Plan Ledger (Condensed)

**Last Updated:** 2026-01-13 (pruned to single active initiative)
**Active Focus:** REFACTOR-MEMOIZE-CORE-001 — Move RawData memoization decorator into core module

---

**Housekeeping Notes:**
- Full ledger snapshot archived at `docs/archive/2026-01-13_fix_plan_archive.md`
- Full Attempts History archived in `docs/fix_plan_archive.md` (snapshot 2026-01-06)
- Earlier snapshots: `docs/archive/2025-11-06_fix_plan_archive.md`, `docs/archive/2025-10-17_fix_plan_archive.md`, `docs/archive/2025-10-20_fix_plan_archive.md`
- Each initiative has a working plan at `plans/active/<ID>/implementation.md` and reports under `plans/active/<ID>/reports/`

---

## Active / Pending Initiatives

### [REFACTOR-MEMOIZE-CORE-001] Move RawData memoization decorator into core module
- Depends on: None
- Priority: Low
- Status: in_progress — Phase B1-B3 delegated to Ralph
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
