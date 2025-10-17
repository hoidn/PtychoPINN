Summary: Add probe_mask parity coverage and verify nphotons override messaging for the config bridge.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Phase B.B5 parity follow-through
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py -k "probe_mask or nphotons" -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T054009Z/{pytest_probe_mask.log,pytest_mvp.log,notes.md}
Do Now: INTEGRATE-PYTORCH-001 Attempt #21 — Add probe_mask parity cases and nphotons override error regression, then run `pytest tests/torch/test_config_bridge.py -k "probe_mask or nphotons" -vv`
If Blocked: Capture failing selector output with current adapter, note obstacle in `notes.md`, and update docs/fix_plan.md Attempts with the log path.
Priorities & Rationale:
- Update probe_mask coverage per field matrix (plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/field_matrix.md) to confirm adapter boolean translation.
- Enforce nphotons override guidance from parity plan (plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md) so physics defaults never regress silently.
- Keep config bridge aligned with reconstructor contract (specs/ptychodus_api_spec.md §5.1-5.3) before proceeding to data pipeline work.
How-To Map:
- Edit `tests/torch/test_config_bridge.py` to add pytest-parametrized probe_mask cases (default False vs override True) using fixtures from `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/fixtures.py`.
- Add a regression that drops the nphotons override and asserts the ValueError mentions `overrides['nphotons']` guidance; pair with a passing case.
- Run `pytest tests/torch/test_config_bridge.py -k "probe_mask or nphotons" -vv 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T054009Z/pytest_probe_mask.log`.
- Re-run `pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg -vv 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T054009Z/pytest_mvp.log` to confirm broader parity after edits.
- Summarize decisions and remaining gaps in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T054009Z/notes.md` for ledger linking.
Pitfalls To Avoid:
- Do not skip parity tests based on torch availability—the harness fallback must stay intact.
- Keep params.cfg snapshot fixture restoring globals; no lingering state between tests.
- Avoid altering adapter defaults beyond probe_mask/nphotons scope this loop.
- Preserve pytest markers (`mvp`) and existing parametrization IDs.
- Ensure logs capture full command output; avoid overwriting earlier evidence directories.
- No blanket pytest runs beyond the specified selectors.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T052500Z/status.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T041908Z/field_matrix.md
- specs/ptychodus_api_spec.md#L1
- tests/torch/test_config_bridge.py#L1
- docs/fix_plan.md#L43
Next Up: Implement params.cfg baseline comparison test (parity plan Phase D1) once probe_mask/nphotons coverage is green.
