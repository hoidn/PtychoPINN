Summary: Add warning coverage for config fields still lacking override enforcement.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 Phase B.B5.D3 override warnings
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py -k "probe_scale or n_groups" -vv; pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T064700Z/{summary.md,pytest_probe_scale.log,pytest_parity.log}
Do Now:
1. INTEGRATE-PYTORCH-001 B.B5.D3 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md — add failing pytest cases covering probe_scale default divergence, missing n_groups override, and missing training test_data_file warning; record red log to plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T064700Z/pytest_probe_scale.log (tests: pytest tests/torch/test_config_bridge.py -k "probe_scale or n_groups or test_data_file" -vv).
2. INTEGRATE-PYTORCH-001 B.B5.D3 @ plans/active/INTEGRATE-PYTORCH-001/implementation.md — implement adapter warnings/errors so new tests pass; rerun targeted selector and overwrite pytest_probe_scale.log (tests: pytest tests/torch/test_config_bridge.py -k "probe_scale or n_groups or test_data_file" -vv).
3. INTEGRATE-PYTORCH-001 B.B5.D3 @ plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md — capture full parity suite after warnings land and save summary.md + pytest_parity.log (tests: pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v).
If Blocked: Capture reproduction script showing which overrides remain `None` using plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T063613Z/train_vs_final_diff.json and log blockers in docs/fix_plan.md Attempts History.
Priorities & Rationale:
- Override gaps flagged in override_matrix.md (plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T063613Z/override_matrix.md) risk silent divergence.
- Spec §§5.1–5.3 (specs/ptychodus_api_spec.md) require explicit handling of lifecycle paths (`train_data_file`, `n_groups`).
- Existing parity plan D3 checklist expects warning coverage before Phase E (plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md).
- CONFIG-001 finding (docs/findings.md) emphasises keeping params.cfg aligned; missing overrides violate this.
How-To Map:
- Clone new tests from existing parity patterns; prefer pytest-style functions with descriptive ids.
- Commands: `pytest tests/torch/test_config_bridge.py -k "probe_scale or n_groups or test_data_file" -vv` (red then green), `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -v`.
- Store logs under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T064700Z/` (red log before fixes, overwrite after green run, plus summary.md capturing warnings implemented and message text).
- Update `summary.md` with warnings added, message strings, and any follow-up gaps.
- When adding warnings, keep adapter torch-optional and reuse existing validation patterns (`pytest.raises` with explicit message checks).
Pitfalls To Avoid:
- Do not hard-import torch inside adapter or tests (keep fallback mode working).
- Avoid mutating global params.cfg outside controlled test fixtures.
- Keep warning/error messages actionable (specify override syntax as in existing nphotons guard).
- Do not regress previously green parity tests; rerun full selector after changes.
- Maintain ASCII in new files; avoid accidental notebook artefacts.
- Do not remove or relax existing validations (probe_mask, nphotons) while adding new ones.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T063613Z/override_matrix.md
- plans/active/INTEGRATE-PYTORCH-001/implementation.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md
- ptycho_torch/config_bridge.py
- tests/torch/test_config_bridge.py
Next Up: After warning coverage lands, move to Phase D.E1 full parity log capture or pivot to data pipeline parity (Phase C) if still pending.
