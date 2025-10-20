Summary: Sync PyTorch factory channel counts with gridsize so the integration workflow goes green.
Mode: Parity
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestTrainingPayloadStructure::test_gridsize_sets_channel_count -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070500Z/phase_c4_cli_integration_debug/{analysis.md,pytest_config_factory.log,pytest_integration.log,pytest_cli_train.log,pytest_cli_inference.log}

Do Now:
1. ADR-003-BACKEND-API C4.D3 factory channel sync @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — extend `create_training_payload()` so `PTModelConfig.C_forward` and `C_model` match `pt_data_config.C`, then add pytest `test_gridsize_sets_channel_count` under `TestTrainingPayloadStructure`; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestTrainingPayloadStructure::test_gridsize_sets_channel_count -vv.
2. ADR-003-BACKEND-API C4.D3 integration rerun @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — rerun the PyTorch integration selector and capture fresh log after the fix; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv.
3. ADR-003-BACKEND-API C4.D3 CLI guardrail sweep @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — confirm training/inference CLI tests stay green; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv.
4. ADR-003-BACKEND-API C4.F ledger wrap-up @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — update `summary.md`, plan row C4.D3, and docs/fix_plan Attempt #22 with artifacts; tests: none.

If Blocked: dump failing selector output to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070500Z/phase_c4_cli_integration_debug/blocker.log`, note whether `payload.pt_model_config.C_forward` still diverges from `pt_data_config.C`, flag C4.D3 `[P]`, append blocker note to docs/fix_plan.md, and stop.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T061500Z/phase_c4_cli_integration_debug/coords_relative_investigation.md — documents confirmed C_forward mismatch, provides replication steps.
- ptycho_torch/config_factory.py:200 — factory currently omits channel sync; needs update for CONFIG-001 parity.
- ptycho_torch/helper.py:66 — reassembly helper uses `model_config.C_forward`; mismatch here causes runtime error.
- tests/torch/test_integration_workflow_torch.py:101 — authoritative regression selector that must pass before C4.D3 can close.
- specs/data_contracts.md:28 — reaffirms grouped coordinate contract; parity requires consistent channel counts.

How-To Map:
- Author test then code: add `test_gridsize_sets_channel_count` ensuring `create_training_payload()` returns `pt_model_config.C_forward == gridsize**2` and `pt_model_config.C_model == gridsize**2`; run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py::TestTrainingPayloadStructure::test_gridsize_sets_channel_count -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070500Z/phase_c4_cli_integration_debug/pytest_config_factory.log`.
- After implementation, rerun integration selector and tee output to `pytest_integration.log` in the same report directory.
- Re-run CLI guards with the provided selectors, tee to `pytest_cli_train.log` and `pytest_cli_inference.log`.
- Record observations + config dumps in `analysis.md` inside the new report directory; include `payload.pt_data_config.C` vs `payload.pt_model_config.C_forward` printout.
- When updating plan/ledger, reference both the new report directory and the supervisor investigation note.

Pitfalls To Avoid:
- Do not patch `helper.py` or reassembly logic; fix the factory so configs agree instead.
- Keep `params.cfg` population order unchanged; avoid introducing second calls to `update_legacy_dict`.
- Do not skip adding the regression test before implementation—TDD parity requires the failing test first.
- Avoid guessing pytest selectors; use the ones listed to keep logs aligned with documentation.
- Do not delete or overwrite existing artifacts in `2025-10-20T061500Z`; add new evidence under `2025-10-20T070500Z` only.
- Preserve ASCII formatting in config_factory and tests; follow existing import order.
- No bulk `pytest tests/` reruns—stick to the mapped selectors for this loop.
- Ensure newly added test uses pytest style (no unittest mix-ins) and cleans up temporary files.
- Don’t change default gridsize overrides for other code paths; only synchronize derived channel counts.
- Remember to drop temporary debugging prints before finalizing the diff.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:98
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T061500Z/phase_c4_cli_integration_debug/coords_relative_investigation.md
- ptycho_torch/config_factory.py:170
- ptycho_torch/config_params.py:30
- tests/torch/test_config_factory.py:70

Next Up: 1) C4.D4 manual CLI smoke once integration is green; 2) Begin C4.E documentation updates (workflow guide + spec tables).
