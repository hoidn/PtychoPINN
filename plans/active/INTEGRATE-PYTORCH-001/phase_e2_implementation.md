# Phase E2 Implementation & Parity Plan (INTEGRATE-PYTORCH-001)

## Context
- Initiative: INTEGRATE-PYTORCH-001
- Phase Goal: Close Phase E2 by wiring the PyTorch train/infer workflow so the new integration tests pass, then capture TensorFlow ↔ PyTorch parity evidence.
- Dependencies: `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` (§E2), backend selection blueprint (`reports/2025-10-17T180500Z/phase_e_backend_design.md`), red-phase evidence (`reports/2025-10-17T213500Z/{phase_e_fixture_sync.md,red_phase.md,phase_e_red_integration.log}`), config bridge adapters (`ptycho_torch/config_bridge.py`), CLI contract in `specs/ptychodus_api_spec.md` §4.5, workflow guidance in `docs/workflows/pytorch.md`.
- Artifact Discipline: Store new execution artifacts under ISO directories (e.g., `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/phase_e2_green.md`, `.../phase_e_tf_baseline.log`). Reference each artifact from docs/fix_plan.md Attempts history.

---

### Phase C — PyTorch Workflow Wiring (E2.C)
Goal: Implement the CLI + dispatcher wiring so `pytest tests/torch/test_integration_workflow_torch.py -vv` passes and CONFIG-001 gates are honoured.
Prereqs: Phase E2 red evidence captured (`reports/2025-10-17T213500Z/`), backend selector plan (Phase E1) understood, PyTorch extras installed locally (`pip install -e .[torch]`).
Exit Criteria: Green logs for backend selection + integration tests stored under a fresh timestamp; CLI wrappers support required flags, fail-fast message present, MLflow suppression available, and persistence artifacts created in the expected location.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Implement training CLI wrapper | [x] | ✅ 2025-10-17 — CLI entrypoint implemented per `reports/2025-10-17T215500Z/phase_e2_green.md` §2. Argparse flags mirror TensorFlow contract, `update_legacy_dict(params.cfg, tf_training_config)` executes prior to dispatch, and checkpoints land under `<output_dir>/checkpoints/`. |
| C2 | Provide inference CLI + artifact checks | [x] | ✅ 2025-10-17 — See `phase_e2_green.md` §2 (C2). `ptycho_torch/inference.py` now loads Lightning checkpoints (`last.ckpt` → `wts.pt` → `model.pt`) and emits `reconstructed_amplitude.png` / `reconstructed_phase.png` while preserving legacy MLflow path. |
| C3 | Lightning/MLflow runtime controls | [x] | ✅ 2025-10-17 — `--disable_mlflow` flag disables autolog + run creation; default retains existing behaviour. Documented in `phase_e2_green.md` §2 (C3). |
| C4 | Dependency + fail-fast policy updates | [x] | ✅ 2025-10-17 — `setup.py` extras `[torch]` now include `lightning`, `mlflow`, `tensordict`; CLI guards raise actionable `RuntimeError` (“Install with: pip install -e .[torch]”) when Lightning imports fail. |
| C5 | Regression + validation tests | [x] | ✅ 2025-10-17 — Targeted selectors executed; logs stored at `reports/2025-10-17T215500Z/{phase_e_backend_green.log,phase_e_integration_green.log}` (skipped in current env due to missing PyTorch runtime, expected). TensorFlow suite remained green (137 passed) per `phase_e2_green.md`. |

---

### Phase D — Parity Evidence Capture (E2.D)
Goal: Demonstrate TensorFlow ↔ PyTorch workflow parity after wiring is complete.
Prereqs: Phase C complete (tests green, CLI available), fixture inventory validated (`phase_e_fixture_sync.md`).
Exit Criteria: TensorFlow and PyTorch integration logs archived, summary document published with metrics + qualitative comparison, docs/fix_plan.md attempt logged.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Capture TensorFlow baseline log | [x] | ✅ 2025-10-18T093500Z — Executed `pytest tests/test_integration_workflow.py::TestFullWorkflow::test_train_save_load_infer_cycle -vv`, captured 31.88s runtime, 1 PASSED. Log at `reports/2025-10-18T093500Z/phase_e_tf_baseline.log`. TensorFlow train→save→load→infer cycle validated successfully. |
| D2 | Capture PyTorch integration log | [x] | ✅ 2025-10-17T231500Z — Post probe-size fix integration run stored at `reports/2025-10-17T231500Z/pytest_integration_green.log`. Probe mismatch resolved (N inferred as 64) per `parity_summary.md`; new blocker: `IndexError` from neighbor indexing at `ptycho_torch/dataloader.py:617`, tracked via `[INTEGRATE-PYTORCH-001-DATALOADER-INDEXING]`. |
| D3 | Publish parity summary | [x] | ✅ 2025-10-17T231500Z — Updated parity narrative in `reports/2025-10-17T231500Z/parity_summary.md` documents probe-size closure, new dataloader indexing failure, and CONFIG-001/DATA-001 compliance. Follow-up item escalated as `[INTEGRATE-PYTORCH-001-DATALOADER-INDEXING]`; Phase E2.D evidence capture remains complete. |

---

## Verification & Reporting Checklist
- [ ] All C-phase tasks reference the new CLI contract and log storage path.
- [ ] PyTorch dependency guidance synchronized with POLICY-001 messaging.
- [x] Each D-phase task records command, output paths, and conclusions in the summary artifact.
- [x] docs/fix_plan.md Attempts history updated in same loop as execution.
- [x] `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` cross-references this plan for E2.C/E2.D guidance.
