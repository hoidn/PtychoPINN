### Turn Summary
Documented `ptycho/cache.py` in docs/index.md so the core RawData memoization helper is discoverable and linked back to the Data Pipeline + shim guidance.
Expanded scripts/simulation/README.md to explain how synthetic_helpers relies on the cache (default `.artifacts/synthetic_helpers/cache`) and how to override or disable it (`--cache-dir`, `use_cache=False`/`--no-cache`).
Captured the required pytest evidence: collect-only listing plus the seeded unit and CLI smoke selectors, with logs archived under the active report hub.
Artifacts: plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/ (pytest_collect.log, pytest_synthetic_helpers.log, pytest_cli_smoke.log)

### Turn Summary
Staged Phase C cleanup for REFACTOR-MEMOIZE-CORE-001 by updating docs/fix_plan + implementation plan and rewriting input.md with the doc/test Do Now.
Documented the landed code (commit d29efc91) and scoped docs/index.md plus scripts/simulation/README.md updates alongside the missing pytest evidence.
Next: Ralph refreshes those docs and reruns the synthetic helper selectors, archiving the logs under the new report timestamp.
Artifacts: plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T232107Z/

### Turn Summary
Moved memoize_raw_data (with helper hashes) into ptycho/cache and pointed synthetic_helpers plus the legacy shim at the new module.
Validated the refactor with pytest tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded -v and pytest tests/scripts/test_synthetic_helpers_cli_smoke.py -v (logs captured).
Next: update docs references for the new cache module and close out Phase C once other imports are audited.
Artifacts: plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/ (pytest_synthetic_helpers.log, pytest_cli_smoke.log)

### Turn Summary
Phase A inventory confirmed memoize_raw_data is only used by synthetic_helpers and now needs a core home (`ptycho/cache.py`) plus a compatibility shim to keep imports stable.
docs/fix_plan.md, the implementation plan, and input.md now capture the move/shim scope along with the mapped pytest selectors and artifacts hub for Phase B1-B3.
Next: Ralph creates `ptycho/cache.py`, converts the scripts shim, updates synthetic_helpers, and runs the two synthetic helper pytest selectors.
Artifacts: plans/active/REFACTOR-MEMOIZE-CORE-001/reports/2026-01-15T225850Z/

### Turn Summary
Drafted an implementation plan to move memoize_raw_data into a core ptycho module with a compatibility shim.
Initialized initiative tracking and documented the expected test selectors for the refactor.
Next: confirm the final module location and shim strategy, then implement Phase B migration.
Artifacts: plans/active/REFACTOR-MEMOIZE-CORE-001/ (implementation.md, summary.md)
