# Implementation Plan (Phased)

## Initiative
- ID: REFACTOR-MEMOIZE-CORE-001
- Title: Move RawData memoization decorator into core module
- Owner: TBD
- Spec Owner: docs/architecture.md
- Status: pending

## Goals
- Relocate the RawData memoization decorator from `scripts/` into a reusable core module under `ptycho/`.
- Preserve hashing behavior, cache directory defaults, and call signature to avoid behavior changes.
- Update script imports (and any docs) to the new core module path.

## Phases Overview
- Phase A — Inventory & Design: confirm usage, pick the core module location, and define the compatibility strategy.
- Phase B — Migration: move the decorator + helpers and update imports.
- Phase C — Verification & Docs: run existing tests and refresh documentation links.

## Exit Criteria
1. `memoize_raw_data` lives under a core module (e.g., `ptycho/cache.py`) with unchanged behavior.
2. `scripts/simulation/synthetic_helpers.py` imports the core decorator; `scripts/simulation/cache_utils.py` is removed or turned into a compatibility shim.
3. Existing synthetic helper tests pass with no behavior regressions.
4. **Test coverage verified:**
   - All cited selectors collect >0 tests (`pytest --collect-only`)
   - All cited selectors pass
   - No regression in existing test suite (full suite green or known-skip documented)
   - Test registry synchronized: `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` updated
   - Logs saved to `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/<timestamp>/`

## Compliance Matrix (Mandatory)
- [ ] **Spec Constraint:** `docs/architecture.md` §7 (stable modules) + `docs/DEVELOPER_GUIDE.md` §2.1 (no import side effects)
- [ ] **Fix-Plan Link:** `docs/fix_plan.md` — add REFACTOR-MEMOIZE-CORE-001 row
- [ ] **Finding/Policy ID:** `ANTIPATTERN-001`, `PYTHON-ENV-001`
- [ ] **Test Strategy:** Inline `Test:` annotations sufficient (refactor only; reuse existing tests)

## Spec Alignment
- **Normative Spec:** `docs/architecture.md`
- **Key Clauses:** stable module policy; explicit dependencies and no import-time side effects

## Testing Integration

**Principle:** Every checklist item that adds or modifies observable behavior MUST specify its test artifact.

**Format for checklist items:**
```
- [ ] <ID>: <implementation task>
      Test: <pytest selector> | N/A: <justification>
```

## Architecture / Interfaces (core cache helper)
- **Module:** `ptycho/cache.py` (new core module)
- **Primary API:** `memoize_raw_data(default_cache_dir: Path, cache_prefix: str, exclude_keys: Iterable[str] | None = None)`
- **Behavior:** stable hash of inputs, disk-backed cache in `.npz`, returns `RawData`
- **Compatibility:** optional shim in `scripts/simulation/cache_utils.py` to avoid broken imports

## Context Priming (read before edits)
- Primary docs/specs to re-read: `docs/architecture.md`, `docs/DEVELOPER_GUIDE.md`, `docs/TESTING_GUIDE.md`
- Required findings/case law: `docs/findings.md` — ANTIPATTERN-001
- Related telemetry/attempts: `scripts/simulation/cache_utils.py`, `scripts/simulation/synthetic_helpers.py`
- Data dependencies to verify: cache root default at `.artifacts/synthetic_helpers/cache`

## Phase A — Inventory & Design
### Checklist
- [x] A1: Inventory memoize_raw_data usage and decide core module location (e.g., `ptycho/cache.py`)
      Test: N/A: analysis
      Notes: Only `scripts/simulation/synthetic_helpers.py::simulate_nongrid_raw_data` uses the decorator, so adding `ptycho/cache.py` keeps the logic close to `RawData` without new dependencies.
- [x] A2: Decide compatibility strategy for `scripts/simulation/cache_utils.py` (shim vs removal)
      Test: N/A: design task
      Notes: Convert `scripts/simulation/cache_utils.py` into a thin shim that re-exports `memoize_raw_data` from `ptycho.cache` and emits a DeprecationWarning so downstream scripts never break.
- [x] A3: Confirm no stable module changes are required; update scope if needed
      Test: N/A: scope check
      Notes: Work only touches new `ptycho/cache.py`, the synthetic helper import, and the shim file—no forbidden modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`) are impacted.

### Dependency Analysis (Required for Refactors)
- **Touched Modules:** `scripts/simulation/cache_utils.py`, `scripts/simulation/synthetic_helpers.py`, new `ptycho/cache.py`
- **Circular Import Risks:** `ptycho/cache.py` should import `RawData` but avoid imports from `scripts/` or `ptycho/loader.py`
- **State Migration:** preserve `CACHE_ROOT` default and hashed key behavior to keep cache hits stable

### Notes & Risks
- Risk: moving helper changes import timing; mitigate by keeping module side-effect free.
- Risk: cache invalidation if hashing changes; keep helper functions identical.

## Phase B — Migration
### Checklist
- [x] B1: Move `memoize_raw_data` + helpers into `ptycho/cache.py` with docstring and type hints  
      Test: `tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded`  
      Notes: Landed in commit `d29efc91`; `_hash_numpy`, `_normalize_for_hash`, `_hash_payload`, and `memoize_raw_data` moved verbatim into the new module with no behavioral drift.
- [x] B2: Update `scripts/simulation/synthetic_helpers.py` to import from `ptycho/cache`  
      Test: `tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded`  
      Notes: `simulate_nongrid_raw_data` now decorates with `from ptycho.cache import memoize_raw_data`, preserving `CACHE_ROOT = .artifacts/synthetic_helpers/cache`.
- [x] B3: Replace `scripts/simulation/cache_utils.py` with a thin shim or remove and update references  
      Test: `tests/scripts/test_synthetic_helpers_cli_smoke.py`  
      Notes: Shim re-exports the decorator and emits a one-time `DeprecationWarning` for legacy imports.

### Notes & Risks
- Keep behavior identical for `use_cache` and `cache_dir` handling.

## Phase C — Verification & Docs
### Checklist
- [x] C1: Re-run synthetic helper selectors and capture both execution and `--collect-only` logs under `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/<timestamp>/`  
      Test: `tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded`, `tests/scripts/test_synthetic_helpers_cli_smoke.py`, `pytest --collect-only tests/scripts/test_synthetic_helpers.py -q`
- [x] C2: Update documentation references to the new core module
      Test: N/A: docs update
- [x] C3: Update ledger + initiative summary with results
      Test: N/A: tracking update

### Notes & Risks
- Doc touchpoints to consider: `docs/index.md`, `docs/architecture.md`, `scripts/simulation/README.md`

## Artifacts Index
- Reports root: `plans/active/REFACTOR-MEMOIZE-CORE-001/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
