# Phase B.B5 Evidence Summary — Parity Harness Blockers

**Initiative:** INTEGRATE-PYTORCH-001  
**Phase:** B.B5 (Probe Mask & nphotons parity blockers)  
**Timestamp:** 2025-10-17T045706Z

## Key Findings

1. **Parity tests now execute but fail** — `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -vv` raises `TypeError` because `TestConfigBridgeParity` inherits `unittest.TestCase` while using `@pytest.mark.parametrize`. Evidence: `pytest_param_failure.md`.
2. **Bridge leaves path value as `PosixPath`** — `params.cfg['model_path']` remains a path object after calling `update_legacy_dict`, so MVP assertion expecting string fails (see `pytest_phaseA.log` and reproduced in `bridge_probe_mask_check.md`). Need explicit string conversion before updating legacy dict or adjust expectation per spec.
3. **`probe_mask` defaults to `False` regardless of PyTorch tensor** — `to_model_config` unconditionally sets `probe_mask=False`. Need logic to translate tensor/None into boolean per spec §5.1 (P0 item from parity plan). No coverage yet because parity tests blocked.
4. **`nphotons` still sourced from PyTorch default** — Without override enforcement, adapter accepts PyTorch default (1e6) even when spec requires explicit train config override (per parity plan summary). Tests should fail once parity suite runs, so implement gating now.
5. **Torch optional path confirmed** — `TORCH_AVAILABLE` flag is `False`; config bridge imports succeed and tests reach assertions, so Phase B.B5.A exit criteria satisfied.

## Recommended Implementation Steps

1. **Refactor parity test class to pytest style**
   - Drop `unittest.TestCase` inheritance and use fixture helpers for params snapshot.
   - Convert methods to module-level functions or simple classes derived from `object`.
   - Move shared setup into pytest fixtures (`@pytest.fixture`).
   - Ensure markers/parameterization remain intact; update `pytest.ini` marker registration if needed.

2. **Normalize path fields prior to `update_legacy_dict`**
   - In `to_training_config`/`to_inference_config`, convert override values to `Path` for dataclasses but ensure downstream KEY_MAPPINGS convert to string. Evaluate spec expectations: §5.3 says `model_path` becomes `params.cfg['model_path']` (string). If dataclass conversion doesn't enforce this, pre-coerce to string within adapter before returning or update KEY_MAPPINGS? Document rationale.

3. **Implement `probe_mask` conversion logic**
   - Accept PyTorch tensor (`None`→False, non-empty tensor→True) or explicit override.
   - Provide override knob to force `True` when fallback bool insufficient.
   - Update parity tests to cover both default and overridden cases.

4. **Enforce explicit `nphotons` overrides**
   - Fail fast when PyTorch default diverges — require override and include actionable error message referencing parity plan.
   - Update MVP test expectation accordingly and parity plan summary to mark addressed.

5. **Capture re-run logs after fixes**
   - `pytest tests/torch/test_config_bridge.py -m "mvp or parity" -vv` with output stored as `pytest_green.log` per plan.

## Artifacts

- `bridge_probe_mask_check.md` — Runtime reproduction showing probe_mask default False and `model_path` stored as `PosixPath`.
- `pytest_param_failure.md` — Captured pytest failure demonstrating parametrization/type error.
- `pytest_phaseA.log` — (from Attempt #15) baseline failure verifying MVP mismatch; reference for comparison after fix.

## Next Steps for Implementation Agent

- Convert `tests/torch/test_config_bridge.py::TestConfigBridgeParity` to pytest style so parameterization works.
- Adjust adapter logic for `probe_mask`, `nphotons`, and path normalization per recommended steps.
- Re-run targeted pytest selectors and update reports/plan + fix_plan once green.

