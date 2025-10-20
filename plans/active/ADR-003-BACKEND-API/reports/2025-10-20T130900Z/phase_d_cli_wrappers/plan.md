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
| B1 | Author training CLI refactor blueprint | [ ] | Draft `training_refactor.md` describing new entry-point layout (e.g., `parse_args_new_interface()`, `run_training_from_args()`), legacy shim strategy, and MLflow disable handling. Reference spec §4.8 for CONFIG-001 ordering. |
| B2 | Establish RED coverage for legacy/new paths | [ ] | Expand `tests/torch/test_cli_train_torch.py` with tests that assert (a) new helper functions invoked, (b) `main()` receives expected payloads, (c) legacy path emits deprecation warning. Capture red log `pytest_cli_train_thin_red.log`. |
| B3 | Implement thin wrapper + GREEN tests | [ ] | Refactor `ptycho_torch/train.py` per blueprint: isolate argparse, move heavy lifting into reusable helper (ideally in `ptycho_torch/cli/train_entry.py` or similar), ensure `update_legacy_dict` remains in factory path, remove duplicate RawData loading (delegate to workflows). Run GREEN selector + `pytest tests/torch/test_workflows_components.py -k train_cli -vv` for regression. |
| B4 | Update docs + plan status | [ ] | Refresh `docs/workflows/pytorch.md` CLI example, add deprecation note for `--device` & legacy interface. Mark Phase D1 rows `[x]` in implementation plan and record attempt details. |

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
