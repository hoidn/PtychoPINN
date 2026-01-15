# REFACTOR-MEMOIZE-CORE-001 — Phase B1-B3: Lift memoize_raw_data into ptycho/cache

**Summary:** Promote the RawData memoization decorator into a new core module (`ptycho/cache.py`) plus a scripts shim so caching stays reusable inside the package.

**Focus:** REFACTOR-MEMOIZE-CORE-001 — Move RawData memoization decorator into core module

**Branch:** paper

**Mapped tests:**
- `pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v`
- `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v`

**Artifacts:** `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/`

---

## Do Now

- REFACTOR-MEMOIZE-CORE-001.B1-B3
  - Implement: `ptycho/cache.py::memoize_raw_data` — move `_hash_numpy`, `_normalize_for_hash`, `_hash_payload`, and `memoize_raw_data` into a new `ptycho/cache.py` module (no side effects), switch `scripts/simulation/synthetic_helpers.py::simulate_nongrid_raw_data` to import from the new module, and turn `scripts/simulation/cache_utils.py` into a thin shim that re-exports the decorator (emit a `DeprecationWarning` once so legacy imports keep working).
  - Validate: `pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v` and `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v`
  - Artifacts: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/`

## How-To Map
1. `mkdir -p ptycho && touch ptycho/cache.py`; copy the helper functions from `scripts/simulation/cache_utils.py` verbatim, add a short module docstring, import `RawData` from `ptycho.raw_data`, and expose `__all__ = ["memoize_raw_data"]`.
2. Update `scripts/simulation/synthetic_helpers.py` line 14 to `from ptycho.cache import memoize_raw_data` and ensure the decorator usage stays identical.
3. Replace the body of `scripts/simulation/cache_utils.py` with a shim:
   ```python
   """Compatibility wrapper for legacy imports."""
   from __future__ import annotations
   import warnings
   from ptycho.cache import memoize_raw_data

   warnings.warn(
       "scripts.simulation.cache_utils is deprecated; use ptycho.cache",
       DeprecationWarning,
       stacklevel=2,
   )
   __all__ = ["memoize_raw_data"]
   ```
   Keep the file in place so downstream scripts continue to work.
4. Run `pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v 2>&1 | tee plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/pytest_synthetic_helpers.log`.
5. Run `pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v 2>&1 | tee plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/pytest_cli_smoke.log` to ensure CLI entrypoints and help text still import the shim cleanly.

## Pitfalls To Avoid
1. Do **not** change the excluded-key set (`use_cache`, `cache_dir`, plus caller-provided `exclude_keys`) when copying the decorator.
2. Keep the hashing logic byte-for-byte identical so cache files stay valid; no new kwargs, no dtype coercions.
3. Ensure `ptycho/cache.py` has zero side effects (no `Path.mkdir()` or logging) at import time per ANTIPATTERN-001.
4. The shim should only warn once; avoid re-running the expensive decorator logic there.
5. Preserve typing: `default_cache_dir` remains a `Path`, and the decorator must raise `TypeError` when the wrapped function returns a non-`RawData`.
6. When editing `synthetic_helpers`, avoid touching unrelated helper functions so existing tests remain stable.
7. Keep pytest invocations from the repo root so fixtures resolve correctly (PYTHON-ENV-001).

## If Blocked
- If importing `ptycho.cache` causes a circular import with `RawData`, capture the traceback, revert the new module (leave shim untouched), and note the cycle in `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/blocker.md` before updating `docs/fix_plan.md` Attempts History.
- If the shim’s `DeprecationWarning` disrupts the tests (e.g., treated as error), downgrade the warning within the shim and document the exact pytest output.

## Findings Applied
| Finding ID | Adherence |
|------------|-----------|
| ANTIPATTERN-001 | New `ptycho/cache.py` stays side-effect free and only exposes explicit APIs, avoiding hidden script-side behavior.
| MIGRATION-001 | Moving the decorator into `ptycho/` reduces duplicated script-level helpers and keeps RawData helpers centralized without expanding `params.cfg` usage.

## Pointers
- `docs/fix_plan.md:7-31` — ledger entry + exit criteria for REFACTOR-MEMOIZE-CORE-001
- `plans/active/REFACTOR-MEMOIZE-CORE-001/implementation.md:53-120` — phase checklist and Phase A notes
- `scripts/simulation/cache_utils.py:1-85` — source decorator that needs to move into core
- `scripts/simulation/synthetic_helpers.py:14-99` — only in-repo usage of `memoize_raw_data`
- `docs/TESTING_GUIDE.md:1-120` & `docs/development/TEST_SUITE_INDEX.md:1-120` — canonical guidance for running the mapped script tests

## Next Up
- Phase C: Update documentation references in `docs/index.md` / `scripts/simulation/README.md` once the core module lands.
