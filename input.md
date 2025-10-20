Summary: Finish ADR-003 Phase C4 by factory-wiring the inference CLI, documenting the training cleanup, and turning the CLI pytest modules green.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv; CUDA_VISIBLE_DEVICES="" pytest tests/ -v
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T044500Z/phase_c4_cli_integration/{pytest_cli_train_green.log,pytest_cli_inference_green.log,pytest_factory_smoke.log,pytest_full_suite_c4.log,refactor_notes.md,summary.md}

Do Now:
1. ADR-003-BACKEND-API C4.C6+C4.C7 implementation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — refactor ptycho_torch/inference.py to build payloads via create_inference_payload(), ensure CONFIG-001 bridging happens before any IO, and pass execution_config through the workflow; tests: none.
2. ADR-003-BACKEND-API C4.C4 documentation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — author refactor_notes.md summarising the removed training hardcodes and updated sourcing; tests: none.
3. Hygiene follow-up (review.md) — restore data/memmap/meta.json to the canonical 34-sample metadata captured before commit ce376dee and stash any diagnostic output under the new artifact directory; tests: none.
4. ADR-003-BACKEND-API C4.D1+C4.D2 validation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — rerun the targeted CLI selectors plus factory smoke (store logs in the new timestamp directory) and confirm all pass; tests: targeted.
5. ADR-003-BACKEND-API C4.D3 + plan sync @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — execute the full pytest suite once targeted checks are green, then update plan/summary rows and append a docs/fix_plan Attempt covering the new evidence; tests: full suite.

If Blocked: If create_inference_payload() surfaces validation errors (missing checkpoint, n_groups, etc.), capture the exact CLI invocation and stack trace to plans/active/ADR-003-BACKEND-API/reports/2025-10-20T044500Z/phase_c4_cli_integration/blocker.log, mark C4.C6 [P] with notes, and log the blocker in docs/fix_plan.md before exiting.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md §C4.C — inference tasks remain open; factory integration is the current blocker.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T041803Z/review.md — documents failing pytest evidence and memmap hygiene issue that must be cleared.
- ptycho_torch/config_factory.py:270-426 — reference implementation for inference payload construction; CLI must delegate here to maintain CONFIG-001 compliance.
- tests/torch/test_cli_inference_torch.py:1-199 — acceptance tests that validate factory wiring; these must go green this loop.
- specs/ptychodus_api_spec.md:70 — backend selection and CONFIG-001 requirements.

How-To Map:
- mkdir -p plans/active/ADR-003-BACKEND-API/reports/2025-10-20T044500Z/phase_c4_cli_integration
- Inference CLI refactor:
  ```python
  from ptycho_torch.config_factory import create_inference_payload
  overrides = {'n_groups': args.n_images}
  payload = create_inference_payload(model_path, test_data_path, output_dir, overrides=overrides, execution_config=execution_config)
  existing_config = (payload.pt_data_config, payload.pt_model_config, payload.pt_training_config, payload.tf_inference_config, payload.pt_datagen_config)
  ```
  Ensure update_legacy_dict has already been invoked inside the factory; avoid duplicating RawData loading prior to payload creation.
- Restore memmap metadata by checking out the previous version:
  ```bash
  git show ce376dee^:data/memmap/meta.json > data/memmap/meta.json
  ```
  Record the restoration in summary.md.
- Targeted tests:
  ```bash
  CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T044500Z/phase_c4_cli_integration/pytest_cli_train_green.log
  CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T044500Z/phase_c4_cli_integration/pytest_cli_inference_green.log
  CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T044500Z/phase_c4_cli_integration/pytest_factory_smoke.log
  ```
- Full regression (run once after targeted tests pass):
  ```bash
  CUDA_VISIBLE_DEVICES="" pytest tests/ -v 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T044500Z/phase_c4_cli_integration/pytest_full_suite_c4.log
  ```
- Update phase_c_execution/plan.md and summary.md, the C4 plan checklist, and append a docs/fix_plan Attempt referencing the new artifacts and test outcomes.

Pitfalls To Avoid:
- Do not open or load NPZ/checkpoint files before the factory populates params.cfg.
- Avoid leaving create_inference_payload() unused; calling RawData directly will keep tests red.
- Keep CUDA disabled during tests (CUDA_VISIBLE_DEVICES="") to match recorded evidence.
- No artifacts at repo root — move any CLI smoke directories under the timestamped report path or delete after capture.
- Ensure data/memmap/meta.json revert is committed with the rest of the changes; no partial working tree.
- Preserve legacy CLI code path behaviour; only modify the new interface branch.
- Do not downgrade pytest_cli_train_green.log — rerun after refactor to capture updated context.
- Use Path objects when passing filesystem arguments to factories to align with type hints.
- When updating docs/fix_plan.md, follow Attempt template and include test selectors + artifact paths.
- Run full suite only once at the end; rerun targeted tests if you iterate.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md#L70 — C4.C6/C4.C7 guidance.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T041803Z/review.md — latest supervisor findings.
- ptycho_torch/inference.py:360-540 — current CLI code to refactor.
- ptycho_torch/config_factory.py:270-426 — inference payload implementation details.
- specs/ptychodus_api_spec.md:70 — backend selection and CONFIG-001 requirements.
- tests/torch/test_cli_inference_torch.py:40 — assertion expectations for execution_config wiring.

Next Up:
- Once the CLI tests are green, proceed to C4.E documentation updates (workflow guide, spec tables, CLAUDE.md) in the following loop.
