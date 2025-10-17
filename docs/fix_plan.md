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
- Status: pending
- Owner/Date: Codex Agent/2025-10-17
- Working Plan: (to be authored) — interim analysis in `reports/2025-10-17T224500Z/parity_summary.md`
- Attempts History:
  * [2025-10-17] Attempt #0 — Documented probe/object dimension mismatch after dataloader fix; no implementation yet.
- Exit Criteria:
  - PyTorch CLI infers consistent probe/object dimensions with TensorFlow baseline; targeted test added (CLI + unit).
  - Integration parity log captured with passing run; parity summary updated, plan/ledger states flipped to done.

