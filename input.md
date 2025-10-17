Summary: Add the params.cfg baseline comparison test for the config bridge and capture targeted pytest evidence.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Phase B.B5.D1 baseline comparison
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_params_cfg_matches_baseline -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T061500Z/{summary.md,pytest_baseline.log,params_diff.json}
Do Now:
1. INTEGRATE-PYTORCH-001 Attempt #24 — Implement `test_params_cfg_matches_baseline` (B5.D1 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md, blueprint @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T061152Z/supervisor_summary.md); wire up the canonical PyTorch config inputs, helper to canonicalize `params.cfg`, and JSON baseline loader — tests: pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_params_cfg_matches_baseline -vv
2. INTEGRATE-PYTORCH-001 Attempt #24 — Run the targeted selector after authoring the test, capture output to `pytest_baseline.log`, and if comparison fails store a diff as `params_diff.json`; update docs/fix_plan.md with results — tests: pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_params_cfg_matches_baseline -vv
If Blocked: Record the failing pytest output under the artifact directory and dump the actual vs baseline dictionaries into `params_diff.json`; note the blocker and diff summary in `summary.md` before updating docs/fix_plan.md.
Priorities & Rationale:
- parity_green_plan D1 keeps the config bridge aligned with the canonical TensorFlow baseline; the new supervisor blueprint defines required overrides and assertions (plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T061152Z/supervisor_summary.md).
- `specs/ptychodus_api_spec.md §5.1-§5.3` require legacy params parity before progressing to PyTorch data pipeline work.
- Existing parity suite covers per-field translation; this test closes the loop on holistic params.cfg equality to guard against regression when overrides or KEY_MAPPINGS change.
How-To Map:
- Author helper `canonicalize_params(cfg)` inside the test module to convert Path → str and serialize primitives; reuse the existing `params_cfg_snapshot` fixture to isolate global state.
- Instantiate PyTorch configs and override dictionaries exactly as detailed in the supervisor summary (DataConfig N=128, grid_size=(3,3), nphotons=5e8, probe_scale=2.0; ModelConfig overrides for probe_mask/pad_object/gaussian_smoothing_sigma; Training/Inference overrides for paths, counts, seeds, weights, booleans).
- Load the baseline with `baseline_path = PROJECT_ROOT / 'plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/baseline_params.json'` and compare against the normalized params dict.
- Run the targeted command: `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_params_cfg_matches_baseline -vv 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T061500Z/pytest_baseline.log`.
- Summarize outcomes in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T061500Z/summary.md` (include pass/fail status, unexpected keys, and follow-ups) before updating docs/fix_plan.md Attempts.
Pitfalls To Avoid:
- Do not rely on torch tensors for `probe_mask`; set the override to keep tests torch-optional.
- Remember to clear `params.cfg` before populating to avoid stale keys from previous tests.
- Keep path handling deterministic (convert to `Path` before calling adapter; comparisons expect stringified outputs from `canonicalize_params`).
- Limit pytest execution to the singled-out selector; no full-suite runs this loop.
- Capture artifacts under the specified timestamped directory; avoid littering repo root with logs.
- Preserve existing parity tests — append new helper/tests without breaking current parametrization or markers.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T061152Z/supervisor_summary.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md
- ptycho_torch/config_bridge.py:1
- tests/torch/test_config_bridge.py:1
- specs/ptychodus_api_spec.md:213
Next Up: Draft the override matrix summary for Phase D2 once the baseline test passes.
