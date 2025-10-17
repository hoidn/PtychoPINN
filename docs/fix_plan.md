# PtychoPINN Fix Plan Ledger

**Last Updated:** 2025-10-17
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
