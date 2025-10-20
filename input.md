Summary: Finish Phase C4 by refitting the inference CLI around the factory payload, then prove the CLI + integration selectors are green.
Mode: TDD
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T050500Z/phase_c4_cli_integration/{fixture_generation.log,pytest_cli_train_green.log,pytest_cli_inference_green.log,pytest_integration_green.log,plan_updates.md,summary.md}

Do Now:
1. ADR-003-BACKEND-API C4.C6+C4.C7 implementation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — regenerate `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`, then refactor `ptycho_torch/inference.py` to consume `create_inference_payload()` + `load_inference_bundle_torch` without ad-hoc RawData/lightning setup; tests: none.
2. ADR-003-BACKEND-API C4.D1+C4.D2 validation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — run targeted CLI selectors and capture GREEN logs once implementation is in place; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv.
3. ADR-003-BACKEND-API C4.D3+C4.D4 verification @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — execute the PyTorch integration workflow test and record manual observations if needed; tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv.
4. ADR-003-BACKEND-API C4.F plan/ledger sync @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — mark checklist rows, append summary.md + fix_plan Attempt with artefact links; tests: none.

If Blocked: Capture the failing command output to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T050500Z/phase_c4_cli_integration/blocker.log`, mark the relevant checklist ID `[P]` with notes, and log the blocker in docs/fix_plan.md before exiting.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md §C4.C — factory payload must own archive + params to stay CONFIG-001 compliant.
- specs/ptychodus_api_spec.md §4.6 — Ptychodus integration expects `wts.h5.zip`; CLI needs to honor this contract.
- ptycho_torch/workflows/components.py#L880 — `load_inference_bundle_torch` already encapsulates bundle loading + params bridge.
- tests/torch/test_cli_inference_torch.py#L60 — acceptance tests assert execution_config wiring; they now expect the CLI to succeed instead of raising FileNotFoundError.
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T225900Z/phase_b_fixture/fixture_notes.md — documents how to regenerate the minimal PyTorch NPZ fixture referenced by the integration test.

How-To Map:
- Fixture regeneration: `python scripts/tools/make_pytorch_integration_fixture.py --source datasets/Run1084_recon3_postPC_shrunk_3.npz --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --subset-size 64 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T050500Z/phase_c4_cli_integration/fixture_generation.log`; commit both `.npz` and `.json` if they change.
- CLI refactor: use `create_inference_payload()` for CONFIG-001, then call `load_inference_bundle_torch(model_path)` to obtain the Lightning module and params. Thread execution_config + payload configs into a dedicated helper so tests can patch that helper instead of exercising real IO. Keep `RawData` usage behind the helper so the CLI surface isn’t forced to open bogus tmp files during unit tests.
- Targeted tests (store logs with `tee`):
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_train_torch.py::TestExecutionConfigCLI -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T050500Z/phase_c4_cli_integration/pytest_cli_train_green.log`
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_cli_inference_torch.py::TestInferenceCLI -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T050500Z/phase_c4_cli_integration/pytest_cli_inference_green.log`
  - `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv 2>&1 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T050500Z/phase_c4_cli_integration/pytest_integration_green.log`
- Record plan/summary updates to `plan_updates.md` and `summary.md` inside the same artefact directory before wrapping.

Pitfalls To Avoid:
- Do not reintroduce manual checkpoint search (`last.ckpt`/`wts.pt`); rely on `load_inference_bundle_torch` and the spec archive name.
- Keep CONFIG-001 sequencing intact: factory must finish before any torch/npz IO.
- Don’t leave regenerated fixture outputs or logs at repo root; everything belongs under the timestamped report directory.
- Ensure the helper invoked by CLI is patchable so unit tests don’t need real NPZ data.
- Preserve CUDA neutrality (`CUDA_VISIBLE_DEVICES=""`) for all pytest runs to match recorded baselines.
- Update docs/fix_plan and plan checklists in the same loop; no stale statuses.
- Watch for `params.cfg already populated` warnings—use factory-provided state instead of manual updates.
- Keep outputs ASCII and respect existing code style when touching CLI.
- Skip full-suite pytest reruns; stick to the mapped selectors unless new regressions surface.
- Do not delete or skip the `.json` sidecar when regenerating the fixture.

Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md#L80
- ptycho_torch/inference.py:360
- ptycho_torch/workflows/components.py:880
- ptycho_torch/model_manager.py:187
- tests/torch/test_cli_inference_torch.py:40
- tests/fixtures/pytorch_integration/minimal_dataset_v1.json
- specs/ptychodus_api_spec.md:205

Next Up: 1) Close C4.E documentation updates once C4.D evidence is green.
