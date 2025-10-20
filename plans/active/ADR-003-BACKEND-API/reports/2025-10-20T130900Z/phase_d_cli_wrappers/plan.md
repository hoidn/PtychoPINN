## Context
- Initiative: ADR-003-BACKEND-API — Standardize PyTorch backend API
- Phase Goal: Deliver Phase D (“CLI Thin Wrappers”) by collapsing `ptycho_torch/train.py` and `ptycho_torch/inference.py` into lightweight shims that delegate to shared factories + workflow helpers without duplicating business logic.
- Dependencies: `plans/active/ADR-003-BACKEND-API/implementation.md` (Phase D checklist), `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T123500Z/phase_c4f_closeout/summary.md` (Phase C4 exit criteria + deferred knobs), `specs/ptychodus_api_spec.md` §4.8 & §7 (backend selection + CLI contracts), `docs/workflows/pytorch.md` §§11–13 (CLI usage + runtime expectations), Findings `CONFIG-001`, `POLICY-001`, `FORMAT-001`.

### Phase A — Baseline & Design Decisions
Goal: Lock current behaviour, identify required entry points, and document deprecation strategy before editing code.
Prereqs: Phase C4 summary reviewed, factory payloads verified (tests/torch/test_config_factory.py GREEN).
Exit Criteria: Decision record capturing CLI decomposition, keeper vs legacy paths, and deprecation plan for overlapping flags.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Capture current CLI call graph | [x] | Use `python -m scripts.tools.print_import_tree ptycho_torch.train` (or manual inspection) to document functions invoked by `cli_main()` → `main()`. Store notes under `phase_d_cli_wrappers/baseline.md` (include file:line anchors). **COMPLETE:** baseline.md generated with full call graph, file:line references, refactor targets. Artifacts: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T131500Z/phase_d_cli_wrappers_baseline/baseline.md` |
| A2 | Verify test harness baselines | [x] | Run targeted selectors to confirm existing CLI tests encode current expectations:<br>`CUDA_VISIBLE_DEVICES=\"\" pytest tests/torch/test_cli_train_torch.py -vv`<br>`CUDA_VISIBLE_DEVICES=\"\" pytest tests/torch/test_cli_inference_torch.py -vv`<br>Archive logs (`pytest_cli_train_baseline.log`, `pytest_cli_inference_baseline.log`) and note any assumptions (e.g., `create_training_payload` patching). **COMPLETE:** Training: 7/7 PASSED (4 warnings), Inference: 4/4 PASSED, logs archived. Artifacts: `pytest_cli_train_baseline.log`, `pytest_cli_inference_baseline.log` |
| A3 | Decide legacy flag handling | [x] | Document plan for `--device` / `--accelerator` coexistence, MLflow flags, and exit conditions (e.g., remove legacy interface or wrap). Produce `design_notes.md` summarising which behaviour remains, which emits warnings, and which moves to Phase E governance. **COMPLETE:** 8 design decisions (D1-D8) documented with rationale, migration paths, and Phase E backlog items. Artifacts: `design_notes.md` |

### Phase B — Training CLI Thin Wrapper (maps to implementation.md D1)
Goal: Reduce `ptycho_torch/train.py` to argument parsing + delegation while keeping both legacy and new interfaces operational during the transition.
Prereqs: Phase A decision notes approved; `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md` C4.F rows marked complete.
Exit Criteria: Training CLI delegates to factory/workflow helpers, tests updated, deprecation warnings documented, and plan checklists advanced to `[x]`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Author training CLI refactor blueprint | [x] | **COMPLETE 2025-10-20:** Authored `training_refactor.md` (21 KB, comprehensive spec). Captured: (i) helper/module layout (`ptycho_torch/cli/shared.py` with 3 functions), (ii) delegation flow (CLI → helpers → factory → workflow), (iii) RawData ownership decision (Option A: CLI retains loading for Phase D), (iv) accelerator warning strategy (DeprecationWarning + UserWarning), (v) `--disable_mlflow` handling (maps to `enable_progress_bar` via `--quiet` alias). References: baseline.md call graph, design_notes.md D1-D8 decisions, spec §4.8 CONFIG-001 ordering. Artifact: `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T105408Z/phase_d_cli_wrappers_training/training_refactor.md` |
| B2 | Establish RED coverage for legacy/new paths | [x] | **COMPLETE 2025-10-20:** Created `tests/torch/test_cli_shared.py` with 20 unit tests (all RED as expected). Tests cover: `resolve_accelerator()` (5 tests), `build_execution_config_from_args()` (9 tests), `validate_paths()` (6 tests). All tests fail with `ModuleNotFoundError: No module named 'ptycho_torch.cli'` (expected RED behavior). Baseline tests remain GREEN (7/7 PASSED in `test_cli_train_torch.py`). Captured RED log: `pytest_cli_train_thin_red.log` (23 KB, 20 FAILED, 7 PASSED, 4.97s runtime). Artifacts: `tests/torch/test_cli_shared.py`, `pytest_cli_train_thin_red.log` |
| B3 | Implement thin wrapper + GREEN tests | [ ] | Execute B3.* checklist below to land helper module, validation updates, CLI refactor, and GREEN evidence (selectors + docs). |

#### B3 Implementation Checklist

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B3.a | Introduce CLI helper package | [x] | ✅ 2025-10-20 — `ptycho_torch/cli/__init__.py` and `ptycho_torch/cli/shared.py` landed per blueprint; helper functions implement warning semantics and are exercised by `tests/torch/test_cli_shared.py`. Artifact hub: `reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl/`. |
| B3.b | Harden execution-config validation | [x] | ✅ 2025-10-20 — `PyTorchExecutionConfig.__post_init__()` now enforces accelerator whitelist + runtime invariants (num_workers, learning_rate, inference_batch_size, accum_steps, checkpoint_save_top_k, early_stop_patience). Coverage captured in helper + CLI selectors (see artifact hub above). |
| B3.c | Refactor training CLI | [x] | ✅ 2025-10-20 — `ptycho_torch/train.py` delegates to helpers, exposes `--quiet`, marks legacy flags deprecated, and preserves CONFIG-001 ordering. RawData stays in CLI per Option A pending Phase D.C decisions. |
| B3.d | Turn RED tests GREEN | [x] | ✅ 2025-10-20 — Targeted selectors GREEN with logs: `pytest_cli_shared_green.log`, `pytest_cli_train_green.log`, `pytest_workflows_train_cli_green.log` under `reports/2025-10-20T111500Z/phase_d_cli_wrappers_training_impl/`. |
| B3.e | Summarise + update plan ledger | [x] | ✅ 2025-10-20 — Checklist states updated, `summary.md` refreshed (B3 recap), docs/fix_plan Attempt #43 recorded with artifact links. |
| B4 | Update docs + plan status | [x] | ✅ 2025-10-20 — Documentation refresh + hygiene complete. `docs/workflows/pytorch.md` updated with `--quiet`, deprecated flag guidance, and helper flow; `tests/torch/test_cli_shared.py` docstring revised to GREEN status; `train_debug.log` relocated to `reports/2025-10-20T112811Z/phase_d_cli_wrappers_training_docs/`. Implementation plan D1 marked `[x]`; see same hub `summary.md` for deliverables. |

### Phase C — Inference CLI Thin Wrapper (maps to implementation.md D2)
Goal: Mirror training refactor for inference CLI, ensuring bundle loading + reconstruction rely on shared helpers.
Prereqs: Phase B complete with GREEN tests; `load_inference_bundle_torch` contract verified.
Exit Criteria: Inference CLI limited to argument parsing, config delegation, and result reporting; tests + docs updated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Draft inference blueprint | [ ] | Produce `inference_refactor.md` capturing helper structure (`run_inference_from_args()`), handling of `--quiet`, accelerator resolution, and integration with reassembly pathway. Reference spec §4.6 for persistence contract. |
| C2 | RED tests for thin wrapper behaviour | [ ] | Extend `tests/torch/test_cli_inference_torch.py` with mocks asserting helper dispatch + error handling. Capture `pytest_cli_inference_thin_red.log`. |
| C3 | Implement thin wrapper + GREEN tests | [ ] | Refactor `ptycho_torch/inference.py` to reuse helpers (consider new module `ptycho_torch/cli/shared.py`). Ensure RawData loading + device moves occur inside workflow helper rather than CLI. Validate with targeted selector + integration workflow test (`pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`). |
| C4 | Docs + plan updates | [ ] | Document new CLI usage patterns, update `docs/workflows/pytorch.md` §12–§13, note deprecation timeline. Mark implementation plan D2 `[x]` and log fix_plan attempt. |

### Phase D — Smoke Tests, Ledger, and Handoff (maps to implementation.md D3)
Goal: Finalise Phase D with deterministic smoke runs, ledger updates, and Phase E readiness notes.
Prereqs: Phases B & C GREEN, documentation drafts prepared.
Exit Criteria: Artifact bundle capturing smoke evidence, ledger updated, hygiene check recorded, next-phase blockers enumerated.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Capture CLI smoke evidence | [ ] | Run documented commands:<br>`python -m ptycho_torch.train --train_data_file … --output_dir tmp/cli_train_smoke --max_epochs 1 --n_images 16 --accelerator cpu --deterministic --num-workers 0 --learning-rate 5e-4 --disable_mlflow`<br>`python -m ptycho_torch.inference --model_path tmp/cli_train_smoke --test_data tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir tmp/cli_infer_smoke --accelerator cpu --quiet`<br>Store logs + runtime metrics under `reports/<TS>/phase_d_cli_wrappers/`. |
| D2 | Update docs/fix_plan + implementation plan | [ ] | Append Attempt entry to `docs/fix_plan.md`, update `implementation.md` Phase D rows (D1–D3) with state + artifact references, ensure plan tables remain authoritative. |
| D3 | Hygiene + Phase E prep notes | [ ] | Record hygiene commands (`git status`, `find tmp/ -maxdepth 1`), confirm tmp outputs cleaned, and author `handoff_summary.md` listing: remaining execution knobs (checkpoint callbacks, logger backend, scheduler), required governance inputs, test gaps. |

## Reporting Discipline
- Store all Phase D artifacts under `plans/active/ADR-003-BACKEND-API/reports/<ISO8601>/phase_d_cli_wrappers/`.
- Each checklist row should have corresponding artifacts (`baseline.md`, `training_refactor.md`, logs, summaries) linked in docs/fix_plan attempts.
- Maintain TDD flow: RED selectors must be captured before implementation; GREEN logs retained for verification.
