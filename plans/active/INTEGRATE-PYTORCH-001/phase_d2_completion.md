# Phase D2 Completion Plan (INTEGRATE-PYTORCH-001)

## Context
- Initiative: INTEGRATE-PYTORCH-001 — PyTorch backend integration
- Phase Goal: Finish the Phase D2 orchestration work by replacing the remaining Lightning and stitching stubs so `run_cdi_example_torch` can train, stitch, and persist models end-to-end.
- Dependencies:
  - `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md` (D2 scaffold history + design decisions)
  - `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` (integration test expectations)
  - `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/parity_summary.md` (latest failure analysis)
  - `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T230724Z/callchain/summary.md` (FORMAT-001 finding that now keeps dataloader green)
  - Specs: `specs/ptychodus_api_spec.md` §4.5–§4.6 (reconstructor lifecycle contract)
  - Workflow guide: `docs/workflows/pytorch.md` §§5–7 (Lightning + MLflow knobs)
- Artifact discipline: Store new evidence under `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/phase_d2_completion/`. Include `summary.md`, targeted pytest logs, and any Lightning debug traces. Reference artifacts from docs/fix_plan.md attempts.

---

### Phase A — Current-State Baseline
Goal: Capture authoritative baseline of stubbed behaviour and parity gaps before modifying orchestration.
Prereqs: None (first loop checklist).
Exit Criteria: Fresh baseline report summarising current failure, active findings, and configuration preconditions.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Record stub inventory + callsites | [x] | Re-read `ptycho_torch/workflows/components.py` sections for `_train_with_lightning`, `train_cdi_model_torch`, `_reassemble_cdi_image_torch`. Document open TODOs + sentinel behaviours in `reports/<TS>/phase_d2_completion/baseline.md`. **Complete:** See `reports/2025-10-17T233109Z/phase_d2_completion/baseline.md` sections 1.1-1.3. |
| A2 | Reproduce latest integration failure | [x] | Run `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv` (requires `pip install -e .[torch]`). Capture log at `reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log` via `tee`. **Complete:** Log captured at 15KB showing Lightning checkpoint load failure with TypeError for missing 4 config args (model_config, data_config, training_config, inference_config). Training subprocess succeeded; checkpoint created at `<output_dir>/checkpoints/last.ckpt`. |
| A3 | Confirm findings ledger coverage | [x] | Validate POLICY-001 and FORMAT-001 references remain accurate; append gaps (e.g., Lightning init contract) to `baseline.md` so future loops know which policies to honour. **Complete:** See `baseline.md` section 3 confirming POLICY-001 (PyTorch mandatory), FORMAT-001 (auto-transpose), CONFIG-001 (params.cfg gate) compliance. |

---

### Phase B — Lightning Training Implementation (D2.B)
Goal: Replace `_train_with_lightning` stub with full Lightning orchestration that honours configuration, determinism, and persistence hooks.
Prereqs: Phase A baseline ready; ensure torch extras installed (`pip install -e .[torch]`); confirm dataset availability (`datasets/Run1084_recon3_postPC_shrunk_3.npz` + probe NPZ as per parity runs).
Exit Criteria: `_train_with_lightning` delegates to `PtychoPINN_Lightning`, uses deterministic seed/path handling, returns structured results dict consumed by downstream workflows, and passes new regression tests.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Author failing Lightning regression tests | [x] | Follow `reports/2025-10-18T000606Z/phase_d2_completion/phase_b_test_design.md`. Add new class `TestTrainWithLightningRed` covering: (1) Lightning module instantiation with all four config objects, (2) `Trainer.fit` invocation with dataloaders derived from train/test containers, (3) results dict exposing `'models'` map with Lightning handle for persistence. Capture red run via `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv \| tee reports/<TS>/phase_d2_completion/pytest_train_red.log`. **Complete:** See `tests/torch/test_workflows_components.py:713-1059` (TestTrainWithLightningRed class with 3 RED tests), `reports/2025-10-18T000606Z/phase_d2_completion/pytest_train_red.log` (3 failed in 5.03s as expected). |
| B2 | Implement Lightning orchestration | [x] | Execute tasks B2.1–B2.8 in `reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md`: derive configs, keep imports torch-optional, build dataloaders via new helper (e.g., `_build_lightning_dataloaders`) wrapping `TensorDictDataLoader`, instantiate `PtychoPINN_Lightning` with `save_hyperparameters()`, configure `Trainer`, run `trainer.fit`, and return results dict with `'models'` key (Lightning module handle required). Capture green log at `reports/2025-10-18T014317Z/phase_d2_completion/pytest_train_green.log` and update the companion `summary.md` in that directory. **Complete (Ralph Verified Attempt #14):** Implementation at ptycho_torch/workflows/components.py:265-529 (_build_lightning_dataloaders helper + _train_with_lightning orchestrator). Targeted test: 2/3 passing (test_runs_trainer_fit ✅, test_returns_models_dict ✅; test_instantiates_module ❌ fixture limitation documented in summary.md). Full regression: 219 passed, 16 skipped, 1 xfailed, 2 failed (ZERO new failures; identical to Attempt #12 baseline). All blueprint tasks B2.1-B2.7 satisfied and exit criteria validated. Summary + logs at reports/2025-10-18T014317Z/phase_d2_completion/. |
| B3 | Surface determinism + MLflow controls | [x] | Honour `config.debug`, `config.output_dir`, and ensure logging respects POLICY-001. **Complete:** Implementation at ptycho_torch/workflows/components.py:478-490 uses `getattr(config, 'debug', False)` for progress bar control, `getattr(config, 'output_dir', ...)` for checkpoint dir, and `deterministic=True` for reproducibility. MLflow integration deferred (TensorFlow baseline also lacks it; see line 749 TODO). Documentation updated at docs/workflows/pytorch.md sections 6, 8, 9. |
| B4 | Turn Lightning regression suite green | [x] | Fixed `TestTrainWithLightningRed.test_train_with_lightning_instantiates_module` fixture by making monkeypatched stub inherit from `lightning.pytorch.core.LightningModule` with minimal `training_step` + `configure_optimizers`. All 3 TestTrainWithLightningRed tests now passing (3/3 ✅). Evidence: `reports/2025-10-18T171500Z/phase_d2_completion/pytest_train_green.log` (1.5KB, 3 passed in 5.27s), full suite 220 passed/16 skipped/1 xfailed/1 failed (ZERO new failures vs baseline). **Complete:** Attempt #21 (2025-10-19). |

---

### Phase C — Inference & Stitching Implementation (D2.C)
Goal: Implement `_reassemble_cdi_image_torch` to mirror TensorFlow stitching path, including inference, coordinate transforms, and reassembly.
Prereqs: Phase B green; trained model artifacts produced under `config.output_dir`; ensure `load_inference_bundle_torch` persists params snapshot from Phase D3.
Exit Criteria: `_reassemble_cdi_image_torch` returns amplitude, phase, and result dict without raising `NotImplementedError`; supports `flip_x`, `flip_y`, `transpose`, and integrates with `run_cdi_example_torch` when `do_stitching=True`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Design inference data flow | [x] | Design note captured at `reports/2025-10-19T081500Z/phase_d2_completion/inference_design.md` covering container normalization, Lightning prediction loop, reassembly helper reuse, test strategy, and open risks. |
| C2 | Add failing pytest coverage | [x] | **Complete:** Added `TestReassembleCdiImageTorchRed` with 4 test methods covering (1) direct NotImplementedError call, (2) parametrized flip/transpose contract (5 combinations), (3) orchestration delegation from `run_cdi_example_torch(do_stitching=True)`, (4) return value contract. Red log captured at `reports/2025-10-19T081500Z/phase_d2_completion/pytest_stitch_red.log` (11KB, 7/8 tests passing with NotImplementedError, 1 pre-existing _ensure_container issue). Tests validate stitching entry path, flip/transpose parameters, and TensorFlow parity signature. |
| C3 | Implement `_reassemble_cdi_image_torch` | [x] | ✅ Attempt #25 (2025-10-19) — `RawDataTorch.generate_grouped_data` now forwards `dataset_path`; `_build_inference_dataloader` helper added; `_reassemble_cdi_image_torch` performs Lightning inference, applies flip/transpose transforms, reassembles via TF helper for MVP parity, and returns `(recon_amp, recon_phase, results)` while requiring `train_results`. Evidence: `reports/2025-10-19T084016Z/phase_d2_completion/{summary.md,pytest_stitch_green.log}`. |
| C4 | Modernize stitching tests + capture green log | [ ] | Update `TestReassembleCdiImageTorchRed` to GREEN expectations: supply `train_results` fixtures, assert amplitude/phase outputs and results payload, and split any legacy NotImplemented assertions into new regression (if required). After test updates, run `pytest tests/torch/test_workflows_components.py -k ReassembleCdiImageTorch -vv` (expect GREEN) and capture log at `reports/2025-10-19T084016Z/phase_d2_completion/pytest_stitch_green.log`. Refresh summary.md with validation notes. |

---

### Phase D — Parity Verification & Documentation
Goal: Demonstrate Phase D2 completion via integration test, update documentation, and refresh plan + ledger.
Prereqs: Phases B & C green, artifacts stored.
Exit Criteria: Integration test passes through training + stitching, parity summary updated, fix plan attempt recorded.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1 | Run full PyTorch integration workflow | [ ] | Execute `pytest tests/torch/test_integration_workflow_torch.py -vv` and capture log `pytest_integration_green.log`. Confirm workflow reaches inference + stitching without stubs. |
| D2 | Update parity summary & docs | [ ] | Append new section to `plans/active/INTEGRATE-PYTORCH-001/reports/<TS>/phase_e_parity_summary.md` (or new summary) comparing TF vs PyTorch outputs; refresh `docs/workflows/pytorch.md` sections 5–7 if CLI behaviour changed. |
| D3 | Refresh plans & ledger | [ ] | Update this plan checklist states, cross-link artifacts, add Attempt entry to docs/fix_plan.md, and ensure `INTEGRATE-PYTORCH-001/phase_d_workflow.md` references completion evidence. |

---

## Reporting Discipline
- All logs reside under `plans/active/INTEGRATE-PYTORCH-001/reports/<ISO8601>/phase_d2_completion/`.
- Each loop updates docs/fix_plan.md Attempts with timestamped artifact list and checklist IDs touched.
- Maintain pytest selectors consistent with `docs/development/TEST_SUITE_INDEX.md` guidance; note skips if environment lacks GPU.
- Honour TDD: author red tests before implementation in Phases B and C; keep logs showing red → green transition.
