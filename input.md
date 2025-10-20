Summary: Surface checkpoint & early-stop knobs via CLI and Lightning callbacks for the PyTorch backend.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase E.B1 (checkpoint controls)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k checkpoint -vv | CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k checkpoint -vv | CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-20T154500Z/{summary.md,pytest_cli_checkpoint_red.log,pytest_cli_checkpoint_green.log,pytest_factory_checkpoint_green.log,pytest_workflows_checkpoint_green.log}

Do Now:
1. EB1.E (author RED tests) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — add checkpoint/early-stop cases to `tests/torch/test_cli_train_torch.py`, `tests/torch/test_config_factory.py::TestExecutionConfigOverrides`, and `tests/torch/test_workflows_components.py` (new class for Lightning callbacks); capture failing selectors (`CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k checkpoint -vv`, `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k checkpoint -vv`, `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv`) into `.../red/`; tests: see selectors (expect FAIL).
2. EB1.A+EB1.B+EB1.C+EB1.D (implement CLI/helper/factory/callback wiring) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — introduce `checkpoint_mode` in `PyTorchExecutionConfig`, expose CLI flags (`--enable-checkpointing/--disable-checkpointing`, `--checkpoint-save-top-k`, `--checkpoint-monitor`, `--checkpoint-mode`, `--early-stop-patience`), map them via `build_execution_config_from_args`, extend factory audit trail, and wire `ModelCheckpoint`/`EarlyStopping` inside `_train_with_lightning`; tests: none (implementation step).
3. EB1.E (GREEN validation) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — rerun checkpoint selectors (`CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py -k checkpoint -vv`, `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k checkpoint -vv`, `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py -k checkpoint -vv`) and store GREEN logs under `.../green/`; if other files changed, finish with `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py tests/torch/test_config_factory.py tests/torch/test_workflows_components.py -vv`; tests: see selectors.
4. EB1.F (docs & ledger sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md — update spec §4.9/§7 tables, `docs/workflows/pytorch.md` execution table, record attempt + artifacts in `docs/fix_plan.md`, and append summary + manifests under the artifact directory; tests: none.

If Blocked: Record details in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-20T154500Z/blockers.md`, leave EB1 rows `[P]`, log blocker in docs/fix_plan.md, and halt.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md (Phase EB1) — authoritative checklist for checkpoint wiring and TDD flow.
- specs/ptychodus_api_spec.md:260-310 — execution config contract; needs new `checkpoint_mode` and updated defaults once implementation lands.
- ptycho_torch/train.py:360-460 — argparse schema + thin wrapper entrypoint that must expose new flags.
- ptycho_torch/cli/shared.py:60-150 — execution config builder to absorb new arguments and validation.
- ptycho_torch/workflows/components.py:640-720 — `_train_with_lightning` area where Lightning callbacks need to honour execution config.

How-To Map:
- RED tests: add pytest cases per plan (use `pytest.param` or new test methods). After authoring, run the three checkpoint selectors with `CUDA_VISIBLE_DEVICES=""` and archive stderr/stdout to `.../red/pytest_*_checkpoint_red.log`.
- Implementation: update `PyTorchExecutionConfig` (add `checkpoint_mode` Literal['min','max'], adjust validation), argparse flags (add enabling/disabling options, default strings), helper to map args (respect `--disable-checkpointing`, defaults to `'min'`), factory audit trail (include new keys), and `_train_with_lightning` to instantiate `ModelCheckpoint`/`EarlyStopping` with metrics pulled from `model.val_loss_name` unless overridden. Patch tests/mocks to import new field names. Ensure `enable_checkpointing=False` disables callbacks gracefully.
- GREEN tests: rerun selectors; on success, rerun combined command if other modules touched. Capture outputs to `.../green/pytest_*_checkpoint_green.log`.
- Docs/spec: update spec tables to include `checkpoint_mode`, CLI defaults. Refresh workflow guide §12 table with new flags/defaults. Add attempt entry + summary referencing artifact logs. Store `summary.md` capturing RED→GREEN evidence and doc updates.

Pitfalls To Avoid:
- Do not bypass `build_execution_config_from_args`; all CLI flags must flow through the helper.
- Keep defaults backwards compatible (`checkpoint_mode='min'`, monitor falls back to `model.val_loss_name`).
- Ensure disabling checkpointing removes callbacks without breaking Trainer instantiation.
- Respect CONFIG-002: execution config changes must not touch `params.cfg`.
- Maintain deterministic CPU runs (`CUDA_VISIBLE_DEVICES=""`) for tests and logs.
- Update spec/workflow docs in same loop; avoid leaving contract drift.
- Use ASCII only in docs/tests; keep tables aligned.
- Capture RED logs before implementing fixes; do not overwrite evidence.
- Avoid importing heavy Lightning modules in tests unnecessarily—use mocking where possible.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/plan.md
- specs/ptychodus_api_spec.md:260-310
- ptycho_torch/train.py:360-460
- ptycho_torch/cli/shared.py:60-150
- ptycho_torch/config_factory.py:200-320
- ptycho_torch/workflows/components.py:640-720
- tests/torch/test_cli_train_torch.py
- tests/torch/test_config_factory.py
- tests/torch/test_workflows_components.py

Next Up: 1. EB2 — scheduler and gradient accumulation knobs (CLI/helper/factory/trainer); 2. EB3 — logger backend decision and implementation.
