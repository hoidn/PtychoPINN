Summary: Unblock Phase B.B5 by fixing parity test harness and implementing probe_mask/nphotons/path handling.
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 — Phase B.B5
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg -vv; pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045936Z/{implementation_notes.md,pytest_green.log}
Do Now: Refactor `TestConfigBridgeParity` to pytest style, then update `config_bridge` to normalize path/probe_mask/nphotons per `reports/2025-10-17T045706Z/evidence_summary.md`; rerun targeted selectors and capture new pytest log.
If Blocked: If pytest still errors on parametrization, capture stack trace to `.../2025-10-17T045936Z/pytest_blocked.log` and document blocker in docs/fix_plan.md Attempts History before exiting.
Priorities & Rationale:
- Align with specs/ptychodus_api_spec.md §5.1-5.3 so params.cfg receives canonical strings/booleans (model_path, probe_mask, nphotons).
- Follow parity_green_plan.md (Phase B tasks B1-B4) to close P0 blockers before advancing to n_subsample work.
- Evidence summary at reports/2025-10-17T045706Z/evidence_summary.md pinpoints harness/type issues that must be resolved first.
- tests/torch/test_config_bridge.py currently mixes unittest + pytest, breaking parameter coverage; converting unlocks the parity matrix assembled in summary.md.
- Ensure compliance with CONFIG-001 finding: reruns must go through update_legacy_dict to validate params bridging.
How-To Map:
- Create artifact dir: `mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045936Z`.
- Convert `TestConfigBridgeParity` to pytest style (drop unittest.TestCase, add fixtures for params snapshot) and adjust imports accordingly.
- Update `ptycho_torch/config_bridge.py`:
  * Convert override paths to str before returning dataclasses so `params.cfg['model_path']` becomes string.
  * Translate `probe_mask` tensor/None to bool or accept override.
  * Require explicit `nphotons` override when PyTorch default differs; raise ValueError with actionable message.
- Re-run targeted tests:
  * `pytest tests/torch/test_config_bridge.py::TestConfigBridgeMVP::test_mvp_config_bridge_populates_params_cfg -vv 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045936Z/pytest_mvp.log`
  * `pytest tests/torch/test_config_bridge.py::TestConfigBridgeParity::test_model_config_direct_fields -vv 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045936Z/pytest_parity_direct.log`
  * If both pass, run `pytest tests/torch/test_config_bridge.py -m "mvp or parity" -vv 2>&1 | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045936Z/pytest_green.log`
- Append implementation summary + remaining risks to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045936Z/implementation_notes.md`.
- Update docs/fix_plan.md Attempts History with results and log references.
Pitfalls To Avoid:
- Do not reintroduce hard torch imports; keep `TORCH_AVAILABLE` guard intact.
- Preserve `update_legacy_dict` sequencing—reset params.cfg between tests via fixtures.
- Avoid mutating spec defaults (pad_object, probe_scale) without plan approval.
- Keep pytest markers (`mvp`, `parity`) consistent; register new markers as needed in pyproject.toml.
- Ensure path normalization happens before returning dataclass to avoid double conversion.
- Capture full command outputs; do not overwrite existing logs from prior attempts.
- Resist changing parity plan scope; focus only on P0 blockers (probe_mask, nphotons, path typing).
- Maintain ASCII-only edits and avoid touching stable TensorFlow core modules.
- Run commands from repo root; no ad-hoc scripts outside documented workflow.
- Do not delete existing artifacts; append new logs with unique filenames.
Pointers:
- specs/ptychodus_api_spec.md §5.1-5.3 — config field contracts.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T050930Z/parity_green_plan.md — phased checklist.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T045706Z/evidence_summary.md — harness issues + next steps.
- tests/torch/test_config_bridge.py — parity/mvp test definitions to refactor.
- ptycho_torch/config_bridge.py — adapter logic to adjust.
Next Up: 1) Phase B.B5.C — resolve n_subsample semantics and error messaging once P0 blockers are green; 2) Phase B.B5.D — implement params.cfg baseline comparison test using baseline_params.json.
