Summary: Finish EB1 by fixing Lightning checkpoint callback tests and updating the spec/workflow docs for the new flags.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase E.B1 (checkpoint controls)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k checkpoint -vv | CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k checkpoint -vv | CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-20T160900Z/{summary.md,pytest_cli_checkpoint_green.log,pytest_factory_checkpoint_green.log,pytest_workflows_checkpoint_green.log}

Do Now:
1. EB1.E (reconfirm RED state) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — rerun `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv`, capture the failing log to `.../2025-10-20T160900Z/red/pytest_workflows_checkpoint_red.log`; tests: selector above (expect FAIL).
2. EB1.D+EB1.E (fix callback tests) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — update `tests/torch/test_workflows_components.py::TestLightningCheckpointCallbacks` to patch `lightning.pytorch.Trainer`, assert callbacks list contents, and keep validation for enable/disable paths; adjust any helper assertions as needed without touching production logic; tests: none (implementation step).
3. EB1.E (GREEN validation) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — rerun all mapped selectors (`CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k checkpoint -vv`, `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k checkpoint -vv`, `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv`) and archive logs under `.../2025-10-20T160900Z/green/`; tests: selectors above.
4. EB1.A+EB1.F (docs & ledger sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — update `specs/ptychodus_api_spec.md` checkpoint table, refresh `docs/workflows/pytorch.md` training execution table with new flags/defaults, flip EB1 rows to `[x]`, author Attempt #59 in `docs/fix_plan.md`, and write `summary.md` for the new artifact directory; tests: none.

If Blocked: Record the issue and evidence in `.../2025-10-20T160900Z/blockers.md`, leave relevant EB1 rows `[P]`, and log the blocker in docs/fix_plan.md before stopping.

Priorities & Rationale:
- Test failures live in `tests/torch/test_workflows_components.py:2791-2988`; fixing mocks restores evidence for EB1.D/E.
- `ptycho_torch/workflows/components.py:690-740` already instantiates callbacks; documentation/spec must reflect the new knobs.
- `specs/ptychodus_api_spec.md:256-280` checkpoint/logging table lacks `checkpoint_mode` and CLI exposure details.
- `docs/workflows/pytorch.md:311-320` training execution flag table needs the new checkpoint flags to avoid drift.
- Plan and ledger updates keep `plans/active/ADR-003-BACKEND-API` and `docs/fix_plan.md` authoritative for the next loop.

How-To Map:
- Red log: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-20T160900Z/red/pytest_workflows_checkpoint_red.log`.
- Test fixes: swap the Trainer patch target to `"lightning.pytorch.Trainer"`, assert `mock_trainer_cls.call_args.kwargs['callbacks']` contains the mocked callbacks, and keep `test_disable_checkpointing_skips_callbacks` covering the negative path (no production changes expected).
- Green runs: repeat all three mapped selectors, saving logs to `.../green/pytest_cli_checkpoint_green.log`, `.../green/pytest_factory_checkpoint_green.log`, and `.../green/pytest_workflows_checkpoint_green.log`.
- Spec/doc refresh: add `checkpoint_mode` row + CLI flag linkage in `specs/ptychodus_api_spec.md` checkpoint section, extend the training CLI table in `docs/workflows/pytorch.md` with `--enable-checkpointing/--disable-checkpointing`, `--checkpoint-save-top-k`, `--checkpoint-monitor`, `--checkpoint-mode`, `--early-stop-patience`, and note defaults.
- Planning hygiene: mark EB1 rows in `plan.md` as `[x]`, append Attempt #59 in `docs/fix_plan.md`, and summarize work (tests run, doc deltas, outstanding risks) in `summary.md` inside the new report directory.

Pitfalls To Avoid:
- Do not patch `lightning.Trainer`; use `lightning.pytorch.Trainer` so the mock intercepts `L.Trainer` calls.
- Keep `CUDA_VISIBLE_DEVICES=""` for all selectors to match parity evidence.
- Preserve the RED→GREEN narrative in test docstrings; update expectations to reflect the new GREEN state only after tests pass.
- Do not alter `_train_with_lightning` behavior; goal is to validate existing callbacks, not change runtime logic.
- Store all logs and summaries in the timestamped `2025-10-20T160900Z` directory before committing.
- Maintain ASCII formatting in spec/doc tables when adding new rows.
- Leave `enable_checkpointing` default `True`; ensure tests do not flip the dataclass default.
- Update plan/fix_plan in the same loop—no deferred ledger work.
- Run all three selectors after edits; skipping CLI/config rounds risks missing regressions.
- Avoid removing existing warnings or fixtures from tests unless directly related to the checkpoint coverage.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md:25
- ptycho_torch/workflows/components.py:690
- tests/torch/test_workflows_components.py:2791
- specs/ptychodus_api_spec.md:256
- docs/workflows/pytorch.md:311

Next Up: EB2 scheduler/accumulation knobs once checkpoint controls are fully green.
