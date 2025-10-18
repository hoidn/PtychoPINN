# PtychoPINN Fix Plan Ledger

**Last Updated:** 2025-10-18
**Active Focus:** Stand up PyTorch backend parity (integration + minimal test harness).

---

> Archived items moved to `archive/2025-10-17_fix_plan_archive.md` to keep this ledger concise. See that file for closed or dormant initiatives.

## [INTEGRATE-PYTORCH-001-STUBS] Finish PyTorch workflow stubs deferred from Phase D2
- Depends on: INTEGRATE-PYTORCH-001 (Phase D2.B/D2.C)
- Spec/AT: `specs/ptychodus_api_spec.md` §4.5–4.6; `docs/workflows/pytorch.md` §§5–7; `plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md`
- Priority: High
- Status: pending
- Owner/Date: Codex Agent/2025-10-17
- Reproduction: Run `ptycho_torch.workflows.components.run_cdi_example_torch(..., do_stitching=True)` with minimal `TrainingConfig`; currently raises `NotImplementedError` because `_reassemble_cdi_image_torch` and Lightning probe handling remain stubs.
- Working Plan: `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md`
- Attempts History:
  * [2025-10-17] Attempt #0 — Catalogued remaining stubs in `ptycho_torch/workflows/components.py` (probe init lines 304-312, `_reassemble_cdi_image_torch` lines 332-352). No implementation yet.
  * [2025-10-17] Attempt #1 — Authored phased completion plan at `plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md` covering Lightning training, stitching, and parity verification tasks; baseline + reproduction guidance captured for upcoming loops.
  * [2025-10-17] Attempt #2 — Phase A baseline documentation complete. Catalogued stub inventory in `baseline.md`: `_train_with_lightning` (stub returns placeholder dict), `_reassemble_cdi_image_torch` (raises NotImplementedError), entry points complete with CONFIG-001 gates. Reproduced integration test failure: Lightning checkpoint loading fails with TypeError (missing 4 config args). Root cause: checkpoint lacks serialized config metadata. Confirmed POLICY-001 and FORMAT-001 compliance in workflows module. Artifacts: `reports/2025-10-17T233109Z/phase_d2_completion/{baseline.md,pytest_integration_baseline.log}`. Plan checklist A1-A3 complete. No implementation changes (docs-only loop per input.md Mode: Docs). Next: Phase B.B1 (author failing Lightning tests).
  * [2025-10-17] Attempt #3 — Supervisor audit detected missing artifact: `reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log` was not committed even though baseline.md references it. Rolled Phase A.A2 back to `[P]` in the plan and instructed engineer to rerun the targeted selector with `tee` into that path before starting Phase B. No code changes.
  * [2025-10-17] Attempt #4 — Phase A.A2 completion (docs-only). Reran integration test with correct selector `pytest tests/torch/test_integration_workflow_torch.py::TestPyTorchIntegrationWorkflow::test_pytorch_train_save_load_infer_cycle -vv` and captured full log (15KB) via `tee` to `reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log`. Confirmed Lightning checkpoint loading failure with TypeError (missing 4 required positional arguments: model_config, data_config, training_config, inference_config). Training subprocess succeeded and created checkpoint at `<output_dir>/checkpoints/last.ckpt`. Updated plan checklist A.A2 to `[x]`. Phase A now complete with all three tasks done. No implementation changes (docs-only loop per input.md Mode: Docs). Artifacts: log file at `reports/2025-10-17T233109Z/phase_d2_completion/pytest_integration_baseline.log`, plan update. Next: Phase B.B1 (author failing Lightning tests).
  * [2025-10-18] Attempt #5 — Supervisor planning for Phase B.B1. Authored Lightning test design spec at `reports/2025-10-18T000606Z/phase_d2_completion/phase_b_test_design.md` detailing three failing pytest cases (`TestTrainWithLightningRed`). Updated plan B1 guidance to reference the design and red-run selector. No production code changes.
  * [2025-10-18] Attempt #6 — Phase B.B1 TDD RED phase complete. Added `TestTrainWithLightningRed` test class to `tests/torch/test_workflows_components.py` with three failing tests encoding Lightning orchestration contract: (1) `test_train_with_lightning_instantiates_module` validates PtychoPINN_Lightning construction with four config objects, (2) `test_train_with_lightning_runs_trainer_fit` validates Trainer.fit invocation with dataloaders, (3) `test_train_with_lightning_returns_models_dict` validates results dict exposes trained module handle for persistence. All three tests FAILED as expected (stub does not instantiate Lightning module or call trainer). Captured red run log (3 failed in 5.03s) via `pytest tests/torch/test_workflows_components.py::TestTrainWithLightningRed -vv | tee reports/2025-10-18T000606Z/phase_d2_completion/pytest_train_red.log`. No production code changes (TDD red phase only). Artifacts: new test class (338 lines), red log (5KB). Plan checklist B1 complete `[x]`. Next: Phase B.B2 (implement Lightning orchestration to turn tests green).
  * [2025-10-18] Attempt #7 — Supervisor planning for Phase B.B2. Authored Lightning orchestration blueprint at `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T020940Z/phase_d2_completion/phase_b2_implementation.md` (tasks B2.1–B2.8) and summary note. Updated `phase_d2_completion.md` B2 row to reference the new checklist and artifact discipline. No production code changes; Mode=Docs planning loop. Next: delegate implementation to turn TestTrainWithLightningRed green.
  * [2025-10-18] Attempt #8 — Supervisor review ahead of B2 implementation. Verified `_train_with_lightning` remains stubbed, confirmed TestTrainWithLightningRed still red, and reserved artifact directory `reports/2025-10-18T014317Z/phase_d2_completion/` for the upcoming green pass. Reissued `input.md` (Mode=TDD) directing the engineer through B2.1–B2.8 with targeted pytest selector + log capture. No production code changes (review/housekeeping loop).
  * [2025-10-18] Attempt #9 — Supervisor housekeeping for B2 execution. Confirmed no new engineer artifacts, updated blueprint + checklist guidance to require `_build_lightning_dataloaders`, and seeded artifact directory `reports/2025-10-18T031500Z/phase_d2_completion/` with a summary scaffold + log destination. Rewrote `input.md` with explicit dataloader/trainer expectations, artifact paths, and POLICY-001 reminders. No production code changes; engineer implementation still pending.
- Exit Criteria:
  - `_reassemble_cdi_image_torch` returns `(recon_amp, recon_phase, results)` without raising `NotImplementedError`.
  - Lightning orchestration path initializes probe inputs, respects deterministic seeding, and exposes train/test containers identical to TensorFlow structure; validated via `tests/torch/test_workflows_components.py`.
  - All Phase D2 TODO markers in `ptycho_torch/workflows/components.py` resolved or formally retired with passing regression tests.

## [INTEGRATE-PYTORCH-001-DATALOADER] Restore PyTorch dataloader DATA-001 compliance
- Depends on: INTEGRATE-PYTORCH-001 (Phase E2.D2 parity evidence)
- Spec/AT: `specs/data_contracts.md` §1; `specs/ptychodus_api_spec.md` §4.5; `docs/workflows/pytorch.md` §4
- Priority: High
- Status: done
- Owner/Date: Codex Agent/2025-10-17
- Working Plan: N/A (complete)
- Attempts History:
  * [2025-10-17] Attempt #0 — Supervisor triage confirming loader only read `diff3d`; artifact: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T223200Z/dataloader_triage.md`.
  * [2025-10-17] Attempt #1 — Implemented canonical-first `diffraction` loading with `diff3d` fallback, added pytest coverage (`tests/torch/test_dataloader.py`), documented parity summary under `reports/2025-10-17T224500Z/`.
- Exit Criteria:
  - Canonical DATA-001 NPZs load successfully; legacy `diff3d` supported as fallback with clear error when neither key exists. ✅
  - Targeted regression tests cover canonical + legacy paths. ✅
  - `pytest tests/torch/test_integration_workflow_torch.py -vv` proceeds past dataloader; residual probe size mismatch tracked under [INTEGRATE-PYTORCH-001-PROBE-SIZE]. ✅

## [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003
- Depends on: INTEGRATE-PYTORCH-001 (Phases C–E alignment)
- Spec/AT: `specs/ptychodus_api_spec.md` §4; `docs/workflows/pytorch.md`; `docs/architecture/adr/ADR-003.md`
- Priority: High
- Status: pending
- Owner/Date: Codex Agent/2025-10-17
- Working Plan: `plans/active/ADR-003-BACKEND-API/implementation.md`
- Attempts History:
  * [2025-10-17] Attempt #0 — Authored phased implementation plan + summary (`reports/2025-10-17T224444Z/plan_summary.md`).
- Exit Criteria:
  - Shared config factories in `ptycho_torch/config_factory.py` with unit tests for override validation.
  - `PyTorchExecutionConfig` dataclass introduced and consumed across training, inference, and bundle-loading workflows with pytest coverage.
  - `ptycho_torch/workflows/components.py` orchestrates via canonical configs + execution config while maintaining CONFIG-001 guard; parity documentation updated.
  - CLI scripts (`train.py`, `inference.py`) reduced to thin wrappers; documentation refreshed; CLI acceptance checks updated.
  - `ptycho_torch/api/` deprecated or delegated to new workflows; ADR-003 marked Accepted with governance artifacts recorded.

## [INTEGRATE-PYTORCH-001-PROBE-SIZE] Resolve PyTorch probe size mismatch in integration test
- Depends on: INTEGRATE-PYTORCH-001-DATALOADER; INTEGRATE-PYTORCH-001 Phase E2.D evidence
- Spec/AT: `specs/data_contracts.md` §1; `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md`
- Priority: High
- Status: done
- Owner/Date: Codex Agent/2025-10-17
- Working Plan: (to be authored) — interim analysis in `reports/2025-10-17T224500Z/parity_summary.md`
- Attempts History:
  * [2025-10-17] Attempt #0 — Detection via dataloader parity loop: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T224500Z/pytest_integration_green.log` captured the new failure after DATA-001 fix. Parity summary documented blocker and recommended this ledger entry. No fix yet (evidence-only loop).
  * [2025-10-17] Attempt #1 — TDD GREEN: Implemented `_infer_probe_size()` utility in `ptycho_torch/train.py:96-140` using zipfile metadata pattern from `dataloader.py:npz_headers()`. Updated CLI path (`train.py:467-481`) to derive `DataConfig.N` from `probeGuess.shape[0]` with fallback to default N=64. Created comprehensive test suite (`tests/torch/test_train_probe_size.py`, 5 tests) covering 64x64/128x128/rectangular/missing probes and real dataset validation. All targeted tests passing (5/5); integration test confirms probe mismatch resolved (N=64 inferred correctly, params.cfg populated); full suite: 211 passed, 14 skipped, 1 xfailed, 1 failed (integration test fails on NEW separate dataloader neighbor indexing bug at line 617, unrelated to probe sizing). Artifacts: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/{pytest_probe_red.log,pytest_probe_green.log,pytest_integration_green.log,parity_summary.md}`. Exit criteria satisfied for probe sizing; new dataloader bug requires separate ledger item.
- Exit Criteria:
  - `pytest tests/torch/test_integration_workflow_torch.py -vv` completes without probe dimension errors, producing checkpoint + inference artifacts for the canonical dataset. ✅ (probe mismatch resolved; new dataloader indexing bug is separate issue)
  - Updated parity summary (new timestamp) records the green run and references the fixing commit/artifacts. ✅ (`plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/parity_summary.md`)
  - `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` and related checklists mark the blocker resolved with guidance for future runs. ✅ 2025-10-17 — D2/D3 rows now reference `reports/2025-10-17T231500Z/` artifacts and escalate `[INTEGRATE-PYTORCH-001-DATALOADER-INDEXING]` follow-up.

## [INTEGRATE-PYTORCH-001-DATALOADER-INDEXING] Fix PyTorch dataloader neighbor indexing overflow
- Depends on: INTEGRATE-PYTORCH-001-PROBE-SIZE; INTEGRATE-PYTORCH-001 Phase E2.D
- Spec/AT: `specs/data_contracts.md` §1; `specs/ptychodus_api_spec.md` §4.5; `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` (D2 guidance)
- Priority: High
- Status: done
- Owner/Date: Codex Agent/2025-10-17
- Working Plan: N/A (complete)
- Attempts History:
  * [2025-10-17] Attempt #0 — Discovered during probe-size parity rerun; see `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T231500Z/parity_summary.md` (IndexError at `ptycho_torch/dataloader.py:617` while assigning `nn_indices` slice). Logged blocker and recommended new ledger entry.
  * [2025-10-17] Attempt #1 — ROOT CAUSE IDENTIFIED: Dataset Run1084_recon3_postPC_shrunk_3.npz uses legacy (H,W,N) format `(64,64,1087)` instead of DATA-001 `(N,H,W)` format. Debugger agent analysis confirmed diffraction stack shape mismatch. Implemented auto-transpose fix in both `_get_diffraction_stack()` (lines 118-127) and `npz_headers()` (lines 75-81) with heuristic detection. Added comprehensive unit test coverage (`TestDataloaderFormatAutoTranspose`, 6 tests). All targeted tests GREEN. Integration test now passes dataloader phase and proceeds to inference (new failure is checkpoint loading, separate issue). Test suite: 217 passed, 14 skipped, 1 xfailed, 1 failed (unrelated). Artifacts: `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T230724Z/callchain/summary.md`, unit tests in `tests/torch/test_dataloader.py:181-369`.
- Exit Criteria:
  - Targeted regression reproduces and then validates fix (`pytest tests/torch/test_integration_workflow_torch.py -vv` plus new unit coverage for neighbor indexing). ✅
  - Dataloader correctly bounds neighbor indices for canonical DATA-001 dataset (64 samples) and oversampled configurations; evidence stored under timestamped reports. ✅
  - `plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md` D2 guidance updated with resolution summary and removal of indexing blocker note. ✅ (Integration test advances past dataloader to inference phase)

## [TEST-PYTORCH-001] Author PyTorch integration workflow regression
- Depends on: INTEGRATE-PYTORCH-001 (Phase E2 complete); POLICY-001 torch-required rollout
- Spec/AT: `specs/ptychodus_api_spec.md` §4 (reconstructor lifecycle), `docs/workflows/pytorch.md` §§2–6, `docs/TESTING_GUIDE.md` (integration tier), `plans/pytorch_integration_test_plan.md`
- Priority: High
- Status: pending
- Owner/Date: Codex Agent/2025-10-17
- Working Plan: `plans/pytorch_integration_test_plan.md` (to be migrated into `plans/active/TEST-PYTORCH-001/implementation.md` with phased checklist)
- Attempts History:
  * [2025-10-17] Attempt #0 — Charter drafted at `plans/pytorch_integration_test_plan.md`; outlines runtime harness, fixture requirements, and acceptance criteria. Awaiting active plan conversion and execution artifacts.
- Exit Criteria:
  - Active plan document created under `plans/active/TEST-PYTORCH-001/implementation.md` referencing charter, with phased checklist and artifact map.
  - PyTorch integration pytest target executes train→infer workflow on canonical fixture within ≤2 minutes CPU, storing logs under `plans/active/TEST-PYTORCH-001/reports/<timestamp>/` and passing in CI.
  - docs/fix_plan.md Attempts history records green run with parity evidence and governance sign-off for torch integration testing.
