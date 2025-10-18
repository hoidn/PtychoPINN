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
| B2 | Implement Lightning orchestration | [ ] | Execute tasks B2.1–B2.8 in `reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md`: derive configs, keep imports torch-optional, build dataloaders via new helper (e.g., `_build_lightning_dataloaders`) wrapping `TensorDictDataLoader`, instantiate `PtychoPINN_Lightning` with `save_hyperparameters()`, configure `Trainer`, run `trainer.fit`, and return results dict with `'models'` key (Lightning module handle required). Capture green log at `reports/2025-10-18T031500Z/phase_d2_completion/pytest_train_green.log` and update the companion `summary.md` in that directory. |
| B3 | Surface determinism + MLflow controls | [ ] | Honour `config.debug`, `config.output_dir`, CLI `--disable_mlflow` flag, and ensure logging respects POLICY-001. Update `docs/workflows/pytorch.md` if behaviour differs. |
| B4 | Turn training tests green | [ ] | Re-run targeted selectors: `pytest tests/torch/test_workflows_components.py::TestTrainLightningParity* -vv` (new cases) and `pytest tests/torch/test_integration_workflow_torch.py -k train_save -vv`. Archive logs under `reports/<TS>/phase_d2_completion/pytest_train_green.log`. |

---

### Phase C — Inference & Stitching Implementation (D2.C)
Goal: Implement `_reassemble_cdi_image_torch` to mirror TensorFlow stitching path, including inference, coordinate transforms, and reassembly.
Prereqs: Phase B green; trained model artifacts produced under `config.output_dir`; ensure `load_inference_bundle_torch` persists params snapshot from Phase D3.
Exit Criteria: `_reassemble_cdi_image_torch` returns amplitude, phase, and result dict without raising `NotImplementedError`; supports `flip_x`, `flip_y`, `transpose`, and integrates with `run_cdi_example_torch` when `do_stitching=True`.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Design inference data flow | [ ] | Draft short design snippet (`reports/<TS>/phase_d2_completion/inference_design.md`) detailing how to re-use Lightning module for prediction, handle complex conversions, and map to TensorFlow's `reassemble_position`. Include references to `ptycho/image/registration.py` and TF implementation. |
| C2 | Add failing pytest coverage | [ ] | Extend `tests/torch/test_workflows_components.py` to assert stitching path delegates to `_reassemble_cdi_image_torch`, executes inference, and returns `recon_amp`, `recon_phase`. Use fixtures or monkeypatch to avoid GPU-heavy execution. Capture red log `pytest_stitch_red.log`. |
| C3 | Implement `_reassemble_cdi_image_torch` | [ ] | Normalize test data via `_ensure_container`, call Lightning `predict`, convert to numpy, apply flips/transposes, and stitch via PyTorch helper (add parity helper mirroring `ptycho_torch.helper`). Ensure results include `obj_tensor_full`, `coords_nominal`, etc., per spec §4.5. |
| C4 | Validate stitching tests | [ ] | Run new selectors plus integration subset requiring `do_stitching=True`: `pytest tests/torch/test_workflows_components.py::TestRunCdiExampleTorch::test_stitching_path -vv`. Store green log `pytest_stitch_green.log`. |

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
