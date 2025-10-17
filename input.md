Summary: Convert config bridge parity tests to pytest style and rerun probe_mask/nphotons parity selectors.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Phase B.B5
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg -vv; pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -vv; pytest tests/torch/test_config_bridge.py -k "parity and (probe_mask or nphotons)" -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T052500Z/{status.md,pytest_parity.log,pytest_full.log}
Do Now: Refactor `TestConfigBridgeParity` into pytest-form tests (drop unittest.TestCase) per parity_green_plan.md B0, then rerun the targeted selectors above and capture logs in the new artifact directory.
If Blocked: If pytest still raises TypeError after the refactor, capture the stack trace to `$REPORT_DIR/pytest_blocked.log` and note the failing parametrized case in docs/fix_plan.md before exiting.
Priorities & Rationale:
- tests/torch/test_config_bridge.py:151 still subclasses unittest.TestCase, so pytest parametrization fails; removing that inheritance unblocks parity coverage.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md tracks B0 harness refactor as the gate for remaining Phase B tasks.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T051750Z/status_review.md summarizes outstanding work and confirms adapter P0 fixes are complete.
- specs/ptychodus_api_spec.md:213 documents probe_mask and nphotons requirements that the parity assertions must enforce.
How-To Map:
- export REPORT_DIR=plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T052500Z
- mkdir -p "$REPORT_DIR"
- Refactor tests/torch/test_config_bridge.py:
  * Drop `unittest.TestCase` from parity class, replace setUp/tearDown with a pytest fixture that saves/restores params.cfg.
  * Convert `self.assert*` to plain `assert` and use fixtures for shared objects per parity plan guidance.
- Register pytest marker updates if needed in pyproject.toml to keep `mvp`/`parity` markers valid.
- Run `pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg -vv 2>&1 | tee "$REPORT_DIR/pytest_mvp.log"`.
- Run `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -vv 2>&1 | tee "$REPORT_DIR/pytest_parity_direct.log"`.
- Run `pytest tests/torch/test_config_bridge.py -k "parity and (probe_mask or nphotons)" -vv 2>&1 | tee "$REPORT_DIR/pytest_parity.log"`.
- Summarize changes and remaining gaps in `$REPORT_DIR/status.md` and update docs/fix_plan.md Attempts History.
Pitfalls To Avoid:
- Keep torch optional guard intact; do not reintroduce hard torch imports.
- Preserve params.cfg snapshots so tests do not leak global state (use fixtures instead of module globals).
- Maintain pytest markers (`mvp`, `parity`) consistent with pyproject configuration.
- Store every log in the new REPORT_DIR; do not drop additional files in repo root.
- Do not edit adapter behaviour beyond probe_mask/nphotons scope without updating parity plan.
- Ensure ValueError message assertions match spec text; avoid weakening error wording.
- Run commands from repo root; rely on documented selectors only.
- Keep edits ASCII-only and avoid touching TensorFlow core modules.
Pointers:
- tests/torch/test_config_bridge.py:151 — parity class to refactor.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md — Phase B0/B2 guidance.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T051750Z/status_review.md — supervisor notes on outstanding tasks.
- specs/ptychodus_api_spec.md:213 — probe_mask contract details.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045706Z/evidence_summary.md — captures previous failures to compare after refactor.
Next Up: 1) Extend parity assertions for override-required fields (B2/B4) once harness passes; 2) Prepare baseline comparison test per parity plan Phase D.
