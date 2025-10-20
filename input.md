Summary: Refactor PyTorch CLI to use execution-config factories and make the new CLI pytest modules GREEN (ADR-003 C4.C)
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv; pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv; pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv; pytest tests/ -v
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/{pytest_cli_train_green.log,pytest_cli_inference_green.log,pytest_factory_smoke.log,pytest_full_suite_c4.log,manual_cli_smoke.log,refactor_notes.md}

Do Now:
1. ADR-003-BACKEND-API C4.C implementation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — complete C4.C1+C4.C2+C4.C3+C4.C4+C4.C5+C4.C6+C4.C7 (argparse flags, factory wiring, execution_config threading, hardcode removal) for both train.py and inference.py; tests: none.
2. ADR-003-BACKEND-API C4.D validation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — execute C4.D1+C4.D2+C4.D3+C4.D4 (CLI pytest selectors, factory smoke, full regression, manual CLI smoke run) capturing GREEN logs; tests: targeted + full suite.
3. ADR-003-BACKEND-API C4 status sync @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/plan.md — update plan/summary checklists for C4.C+C4.D, append docs/fix_plan Attempt with artifact paths, add refactor_notes.md + manual CLI evidence (C4.F1+C4.F2+C4.F4); tests: none.

If Blocked: If factory calls cannot infer probe size or params.cfg population fails, capture traceback + CLI arguments, store under the artifact directory, set plan rows to `[P]` with blocker notes, and log the condition in docs/fix_plan.md before exiting.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md §C4.C — authoritative checklist for CLI refactor scope.
- ptycho_torch/config_factory.py: create_*_payload() already enforces CONFIG-001; CLI must delegate instead of hand-rolling configs.
- tests/torch/test_cli_train_torch.py & tests/torch/test_cli_inference_torch.py — RED scaffolds that must go GREEN to close TDD loop.
- ptycho_torch/train.py:360 and ptycho_torch/inference.py:300 — current argparse + legacy config blocks to replace.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/summary.md — central status log that must reflect C4 progress.

How-To Map:
- Extend argparse: add `--accelerator` (choices auto/cpu/gpu/cuda/tpu/mps, default auto), paired `--deterministic`/`--no-deterministic`, `--num-workers` (int, default 0), `--learning-rate` (float, default 1e-3) in train.py; add `--accelerator`, `--num-workers`, `--inference-batch-size` (int, default None) in inference.py. Retain `--device` for backward compatibility but emit warning and map to accelerator when `--accelerator` absent.
- Build overrides for factories:
  ```python
  overrides = {
      'n_groups': args.n_images,
      'batch_size': args.batch_size,
      'gridsize': args.gridsize,
      'max_epochs': args.max_epochs,
  }
  if args.test_data_file:
      overrides['test_data_file'] = Path(args.test_data_file)
  ```
  Use `Path` objects for data/output arguments, and let factory manage defaults (no manual nphotons/K/experiment_name overrides). For inference, include `model_path`, `output_dir`, `test_data_file`, `n_groups` in overrides.
- Create execution config from CLI flags:
  ```python
  exec_cfg = PyTorchExecutionConfig(
      accelerator=resolved_accelerator,
      deterministic=args.deterministic,
      num_workers=args.num_workers,
      learning_rate=args.learning_rate,
      inference_batch_size=args.inference_batch_size,
  )
  ```
  Keep other fields at defaults; reuse for both payload construction and audit.
- Training flow: call `create_training_payload(train_data_file, output_dir, overrides=overrides, execution_config=exec_cfg)`; pass payload’s canonical config + execution config to downstream orchestration (either `main(..., existing_config=...)` or `run_cdi_example_torch` per plan §C4.C3) and drop the legacy hand-built configs. Ensure factories populate params.cfg before any data IO.
- Inference flow: call `create_inference_payload(model_path, test_data_path, output_dir, overrides=overrides, execution_config=exec_cfg)`; feed payload outputs into existing inference helpers and remove manual DataLoader wiring.
- Capture logs with tee:
  ```bash
  CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_cli_train_green.log
  CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestExecutionConfigCLI -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_cli_inference_green.log
  CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_config_factory.py -k ExecutionConfig -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_factory_smoke.log
  CUDA_VISIBLE_DEVICES="" pytest tests/ -v 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/pytest_full_suite_c4.log
  python -m ptycho_torch.train --train_data_file datasets/fly/fly001_transposed.npz --output_dir plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/cli_smoke_run --n_images 64 --max_epochs 1 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-4 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/manual_cli_smoke.log
  ```
  Remove or relocate the `cli_smoke_run` directory after capturing results if not needed.
- Document removed hardcodes in `refactor_notes.md` (brief table of old constant → new source).
- Update `phase_c_execution/plan.md`, `phase_c_execution/summary.md`, `plans/.../phase_c4_cli_integration/plan.md`, and `docs/fix_plan.md` once work is complete, citing exact artifact filenames.

Pitfalls To Avoid:
- Do not bypass factory helpers or duplicate CONFIG-001 bridging logic in the CLI.
- Keep argparse incompatibilities clear: reject simultaneous use of legacy flags and new execution-config flags; surface helpful errors.
- Ensure `PyTorchExecutionConfig` fields remain backend-neutral (no TensorFlow assumptions) and avoid mutating global defaults.
- Maintain deterministic default (`deterministic=True`) unless `--no-deterministic` is provided; cover both branches in tests.
- Store every log and temporary run directory under the phase_c4_cli_integration report path; no artefacts at repo root or /tmp left behind.
- Leave RED-era assertions in place; tests must transition to GREEN without loosening expectations.
- When warning about `--device`, use stderr/stdout consistently but do not exit; log once per invocation.
- Preserve existing legacy interface code path untouched except for new accelerator resolution reuse.

Pointers:
- ptycho_torch/train.py:360 — argparse + config-bridge code slated for replacement.
- ptycho_torch/inference.py:300 — manual inference pipeline to refactor through factories.
- ptycho_torch/config_factory.py:170-280 — expected overrides/execution_config semantics.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md:65 — implementation checklist with exit criteria.
- tests/torch/test_cli_train_torch.py:1 — RED tests defining acceptance behaviour.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T004233Z/phase_c_execution/summary.md:20 — status ledger requiring update post-implementation.

Next Up:
- C4.E documentation updates (workflow guide, spec CLI tables, CLAUDE.md examples) once CLI behaviour is verified.
- C4.F3 Phase D prep notes to capture deferred execution knobs for checkpoint/logging governance.
