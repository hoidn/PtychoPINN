# Implementation Plan (Phased)

## Initiative
- ID: SYNTH-HELPERS-001
- Title: Script-level synthetic simulation helpers
- Owner: TBD
- Spec Owner: docs/DATA_GENERATION_GUIDE.md
- Status: pending

## Goals
- Consolidate synthetic object/probe creation and nongrid simulation into a shared helper module.
- Reduce duplication across `dose_response_study.py`, `sim_lines_4x`, and synthetic lines runner without changing behavior.
- Preserve grid vs nongrid separation and CONFIG-001 ordering requirements.

## Phases Overview
- Phase A — Design & Inventory: confirm touchpoints, API surface, and scope.
- Phase B — Implementation: add helpers and refactor target scripts.
- Phase C — Verification & Docs: validate behavior and update docs if needed.

## Exit Criteria
1. Shared helper module used by `scripts/studies/dose_response_study.py`, `scripts/studies/sim_lines_4x/pipeline.py`, and `scripts/simulation/run_with_synthetic_lines.py`.
2. Nongrid simulation behavior matches prior outputs (seeded trajectories and split logic preserved).
3. CONFIG-001 and ANTIPATTERN-001 compliance preserved (explicit params update, no import side effects).
4. **Test coverage verified:**
   - All cited selectors collect >0 tests (`pytest --collect-only`)
   - All cited selectors pass
   - No regression in existing test suite (full suite green or known-skip documented)
   - Test registry synchronized: `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` updated
   - Logs saved to `plans/active/SYNTH-HELPERS-001/reports/<timestamp>/`

## Compliance Matrix (Mandatory)
- [ ] **Spec Constraint:** `docs/DATA_GENERATION_GUIDE.md` — grid vs nongrid separation; `generate_simulated_data()` usage; CONFIG-001 sequencing
- [ ] **Fix-Plan Link:** `docs/fix_plan.md` — add SYNTH-HELPERS-001 row if tracked
- [ ] **Finding/Policy ID:** `CONFIG-001`, `ANTIPATTERN-001`, `PYTHON-ENV-001`
- [ ] **Test Strategy:** `plans/active/SYNTH-HELPERS-001/test_strategy.md` (link from `docs/fix_plan.md`)

## Spec Alignment
- **Normative Spec:** `docs/DATA_GENERATION_GUIDE.md`
- **Key Clauses:** grid vs nongrid pipeline split; `update_legacy_dict` before legacy usage; grouping required for nongrid output

## Testing Integration

**Principle:** Every checklist item that adds or modifies observable behavior MUST specify its test artifact.

**Format for checklist items:**
```
- [ ] <ID>: <implementation task>
      Test: <pytest selector> | N/A: <justification>
```

## Architecture / Interfaces (helper API)
- **Module:** `scripts/simulation/synthetic_helpers.py` (script-level utility; no new package surface)
- **Functions (proposed signatures + behavior):**
  - `make_lines_object(object_size: int, *, data_source: str = "lines", seed: int | None = None) -> np.ndarray`
    - Returns `complex64` object canvas of shape `(object_size, object_size)` using `sim_object_image`.
    - Temporarily overrides `params.cfg['data_source']` and restores prior value in a `try/finally` to avoid leakage on exceptions.
    - If `seed` is provided, sets NumPy RNG for deterministic object generation.
  - `make_probe(N: int, *, mode: str = "idealized", path: Path | None = None, scale: float = 0.7) -> np.ndarray`
    - `idealized`: uses `probe.get_default_probe(N, fmt="np")` and sets `params.cfg['default_probe_scale']` if missing.
    - `integration`: loads `probeGuess` from NPZ at `path`, validates shape `(N, N)`. Raises `KeyError` if `probeGuess` is missing (no fallback keys).
    - Returns `complex64` probe array, no other side effects.
  - `simulate_nongrid_raw_data(object_guess: np.ndarray, probe_guess: np.ndarray, *, N: int, n_images: int, nphotons: float, seed: int, buffer: float | None = None, sim_gridsize: int = 1) -> RawData`
    - Enforces `sim_gridsize == 1` (raises `ValueError` otherwise) to match `RawData.from_simulation` limitations.
    - Builds `TrainingConfig(ModelConfig(N=N, gridsize=1), n_groups=n_images)` so **`n_images` maps directly to `n_groups`** for parity with existing scripts.
    - If `buffer` is `None`, defaults to `0.35 * min(object_guess.shape)` (matches current `sim_lines_4x` behavior).
    - Calls `update_legacy_dict(params.cfg, config)` internally, then sets NumPy RNG to `seed` before `generate_simulated_data` for deterministic scan positions.
    - Returns `RawData` with `diff3d`, `xcoords`, `ycoords`, `probeGuess`, `objectGuess`.
    - **CONFIG-001 ownership:** this helper is the only location that calls `update_legacy_dict` for simulation; callers must not pre-call it.
  - `split_raw_data_by_axis(raw_data: RawData, *, split_fraction: float = 0.5, axis: str = "y") -> tuple[RawData, RawData]`
    - Deterministic split by sorting `xcoords` or `ycoords` ascending.
    - Test set takes the upper tail (`last` indices) of the sorted axis; train set takes the remaining leading indices.
    - Validates `0 < split_fraction < 1` and ensures non-empty splits.

## Context Priming (read before edits)
- Primary docs/specs to re-read: `docs/DATA_GENERATION_GUIDE.md`, `docs/DEVELOPER_GUIDE.md`, `docs/architecture.md`, `docs/TESTING_GUIDE.md`, `plans/templates/test_strategy_template.md`
- Required findings/case law: `docs/findings.md` — CONFIG-001, ANTIPATTERN-001, PYTHON-ENV-001
- Related telemetry/attempts: N/A
- Data dependencies to verify: integration probe path `ptycho/datasets/Run1084_recon3_postPC_shrunk_3.npz`

## Phase A — Design & Inventory
### Checklist
- [ ] A0: Create `test_strategy.md` from template and link it in `docs/fix_plan.md`
      Test: N/A: planning artifact
- [ ] A1: Define helper API surface and scope (nongrid only; grid left intact)
      Test: N/A: design task
- [ ] A2: Inventory call sites and required behavior parity (seeded scans, split axis)
      Test: N/A: analysis
- [ ] A3: Define helper test cases + selectors (seed determinism, split axis)
      Test: N/A: test design

### Dependency Analysis (Required for Refactors)
- **Touched Modules:** `scripts/studies/dose_response_study.py`, `scripts/studies/sim_lines_4x/pipeline.py`, `scripts/simulation/run_with_synthetic_lines.py`, new `scripts/simulation/synthetic_helpers.py`
- **Circular Import Risks:** Keep helpers under `scripts/` and avoid importing training modules at top level.
- **State Migration:** Move params/probe/object setup into helper functions; keep `update_legacy_dict` inside simulation helper to preserve CONFIG-001 ordering.

### Notes & Risks
- Risk: hidden reliance on global `params.cfg` ordering in scripts; mitigate by explicit helper sequence.
- Risk: seed-based reproducibility drift; keep seeds explicit and shared across arms.
- Risk: nongrid helper enforces `sim_gridsize=1`; any future gridsize>1 simulation needs a deliberate API change + tests.
- Phase B is blocked until A0 completes (test strategy doc + `docs/fix_plan.md` linkage).

## Phase B — Implementation
### Checklist
- [ ] B1: Add `scripts/simulation/synthetic_helpers.py` (object/probe creation, nongrid simulation, split helper)
      Test: `tests/scripts/test_synthetic_helpers.py::test_simulate_nongrid_seeded`
- [ ] B2: Add helper unit tests (seed determinism + split axis + gridsize enforcement)
      Test: `tests/scripts/test_synthetic_helpers.py::test_split_raw_data_by_axis`
- [ ] B3: Add CLI smoke tests for refactored scripts
      Test: `tests/scripts/test_synthetic_helpers_cli_smoke.py`
      Notes: `--help` smoke for `scripts/studies/dose_response_study.py`, `scripts/studies/sim_lines_4x/run_gs1_ideal.py`, `scripts/simulation/run_with_synthetic_lines.py`; import+pure helper smoke for `scripts/studies/sim_lines_4x/pipeline.py` (e.g., `derive_counts`).
- [ ] B4: Refactor `scripts/studies/sim_lines_4x/pipeline.py` to use helpers
      Test: `tests/scripts/test_synthetic_helpers.py`
- [ ] B5: Refactor `scripts/studies/dose_response_study.py` nongrid path to use helpers
      Test: `tests/scripts/test_synthetic_helpers.py`
- [ ] B6: Refactor `scripts/simulation/run_with_synthetic_lines.py` to use helpers
      Test: `tests/scripts/test_synthetic_helpers.py`

### Notes & Risks
- Keep grid-mode path in `dose_response_study.py` unchanged to avoid legacy behavior changes.

## Phase C — Verification & Docs
### Checklist
- [ ] C1: Run helper + CLI smoke tests and archive logs per TEST-CLI-001
      Test: `tests/scripts/test_synthetic_helpers.py`, `tests/scripts/test_synthetic_helpers_cli_smoke.py`
- [ ] C2: Update docs for new helper usage and test registration
      Test: N/A: docs

### Notes & Risks
- Documentation updates to consider:
  - `docs/DATA_GENERATION_GUIDE.md` (add helper usage + nongrid flow reference)
  - `docs/DEVELOPER_GUIDE.md` (explicit helper ordering + CONFIG-001 mention)
  - `docs/architecture.md` (component reference for helpers)
  - `docs/index.md` (link new/updated helper guidance if promoted)
  - `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` (register new tests)
  - `scripts/simulation/README.md`, `scripts/studies/sim_lines_4x/README.md` (usage adjustments if any CLI changes)

## Artifacts Index
- Reports root: `plans/active/SYNTH-HELPERS-001/reports/`
- Latest run: `<YYYY-MM-DDTHHMMSSZ>/`
