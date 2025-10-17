# Phase B.B5 Status Review — Test Harness Alignment

**Timestamp:** 2025-10-17T051750Z
**Initiative:** INTEGRATE-PYTORCH-001
**Focus:** Phase B.B5 (Parity green plan)
**Action Type:** Review / housekeeping

## Current State
- Phase A (torch-optional harness) is complete per Attempt #15; pytest selectors now execute without SKIP when torch is absent.
- Adapter P0 fixes (probe_mask translation, nphotons override enforcement, path normalization) landed in Attempt #17; MVP test passes.
- Parity suite remains red because `TestConfigBridgeParity` still subclasses `unittest.TestCase` while using `@pytest.mark.parametrize`, triggering `TypeError` (see `../2025-10-17T045706Z/pytest_param_failure.md`).
- Parity plan checklist still shows Phase A/B rows as open even though portions are done, causing drift between docs and reality.

## Outstanding Gaps
1. **Parity test harness refactor** — convert `tests/torch/test_config_bridge.py::TestConfigBridgeParity` to pytest style (fixture-based) so parameterization executes.
2. **Test coverage updates** — once pytest style lands, extend assertions for probe_mask/nphotons per field matrix (Phase B.B5.B2/B4).
3. **Plan bookkeeping** — mark completed tasks (Phase A, Phase B.B5.B1/B3) and note blockers for B2/B4 to avoid duplicate effort.
4. **Artifact placement** — ensure future logs land under `reports/` directories (train_debug.log remains at repo root from previous loop).

## Recommended Next Steps
1. Refactor `TestConfigBridgeParity` class to pytest style (drop `unittest.TestCase`, replace `setUp`/`tearDown` with fixture, convert assertions). Validate by re-running `pytest tests/torch/test_config_bridge.py -k parity -vv`.
2. Restore parameterized cases for probe_mask/nphotons once pytest harness works; confirm adapter behaviour by updating expected assertions.
3. Capture new pytest logs under a fresh timestamped directory (e.g., `plans/active/INTEGRATE-PYTORCH-001/reports/<ts>/pytest_parity.log`).
4. Move `train_debug.log` into the relevant reports folder or exclude it before final parity commit to keep repo root clean (non-blocking but note for upcoming loops).

## References
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045706Z/evidence_summary.md` — probe_mask/nphotons gap analysis.
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045936Z/summary.md` — adapter implementation recap.
- `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md` — checklist to update in this loop.
- `tests/torch/test_config_bridge.py` — parity test definitions needing refactor.

