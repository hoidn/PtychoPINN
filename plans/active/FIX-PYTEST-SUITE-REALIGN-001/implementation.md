# Implementation Plan — FIX-PYTEST-SUITE-REALIGN-001

## Initiative
- ID: FIX-PYTEST-SUITE-REALIGN-001
- Title: Pytest Suite Realignment & CLI Guardrails
- Owner/Date: Ralph / 2025-11-14
- Status: pending
- Priority: High
- Working Plan: this file
- Reports Hub (primary): `plans/active/FIX-PYTEST-SUITE-REALIGN-001/reports/2025-11-14T000000Z/test_cleanup/`

## Context Priming (read before edits)
- [ ] docs/index.md — Canonical map of specs, workflows, and policies; use it to ensure we do not miss required references when touching CLI/workflow code.
- [ ] docs/INITIATIVE_WORKFLOW_GUIDE.md — Defines required plan/test artifacts and Supervisor/Engineer loop expectations for this initiative.
- [ ] docs/TESTING_GUIDE.md — Authoritative instructions for running/recording pytest evidence and updating the test registry.
- [ ] docs/findings.md — Capture CONFIG-001, POLICY-001, DEVICE-MISMATCH-001, EXEC-ACCUM-001; these findings govern CLI config bridging, PyTorch requirements, and Lightning trainer behavior.
- [ ] docs/architecture.md & docs/architecture_torch.md — High-level data/workflow diagrams covering CLI configuration flow, PyTorch workflows, and legacy params bridge.
- [ ] specs/spec-ptycho-workflow.md & specs/ptychodus_api_spec.md — Contract for workflow orchestration, CLI interfaces, and reconstructor lifecycle that the PyTorch tests assert against.
- [ ] plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py — Current Phase G orchestrator whose behavior the failing Phase G tests target.
- [ ] prompts/main.md & prompts/supervisor.md — Loop execution format and dwell/test evidence policies that apply when implementing this plan.

## Problem Statement
Pytest currently fails across multiple suites because the training CLI rejects modern flags, the Phase G orchestrator tests still emulate the old CLI command path, and PyTorch workflow tests rely on stale stubs that no longer match the execution config/plumbing. We need a structured effort to restore CLI compatibility, realign the tests with the current orchestrator + PyTorch behaviors, and prove the fixes with updated regression coverage while honoring CONFIG‑001/POLICY‑001 guardrails.

## Objectives
- Restore CLI compatibility so `run_baseline.py` and dependent scripts accept `--torch_loss_mode` (and future `Literal` fields) without errors.
- Update Phase G orchestrator tests to match the programmatic overlap-generation path and new post-verify automation requirements.
- Realign PyTorch workflow/inference tests with the new execution-config plumbing, Lightning trainer behavior, and tensor-based inference outputs.
- Provide deterministic pytest evidence showing the suites under `tests/study/`, `tests/test_integration_baseline_gs2.py`, `tests/test_workflow_components.py`, `tests/torch/test_cli_inference_torch.py`, and `tests/torch/test_workflows_components.py` all pass locally.

## Deliverables
1. CLI parser update + targeted regression test proving Literal arguments (e.g., `--torch_loss_mode`) work end-to-end.
2. Revised `tests/study/test_phase_g_dense_orchestrator.py` fixtures and utilities that mirror the programmatic Phase D path without requiring real NPZ assets.
3. Updated PyTorch workflow/inference tests (CLI + components) with properly typed stubs/mocks and execution-config assertions.
4. Reports Hub entry containing pytest logs (`pytest tests/test_integration_baseline_gs2.py`, focused selectors for the Phase G suite, PyTorch suites, plus an aggregate `pytest --maxfail=1 tests`) and any auxiliary scripts used for reproductions.

## Phases Overview
- Phase A — CLI Guardrail & Smoke Coverage: fix Literal argument parsing and add baseline regression tests/logs.
- Phase B — Phase G Orchestrator Test Realignment: update the dense pipeline tests to follow the new programmatic overlap path and post-verify automation.
- Phase C — PyTorch Workflow Test Resync: modernize PyTorch CLI/workflow tests, including execution-config plumbing and tensor-based inference stubs, then re-run the full torch suite.

## Exit Criteria
1. `pytest tests/test_integration_baseline_gs2.py` passes without CLI errors and logs stored under the Reports Hub.
2. `pytest tests/study/test_phase_g_dense_orchestrator.py` passes with updated fixtures that no longer depend on the removed overlap CLI commands.
3. `pytest tests/test_workflow_components.py::TestLoadInferenceBundle::test_load_valid_model_directory` and all tests in `tests/torch/test_cli_inference_torch.py` and `tests/torch/test_workflows_components.py` pass on a clean tree.
4. Test registry synchronized: update `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` if selectors changed; archive `pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py` and `pytest --collect-only tests/torch/test_workflows_components.py` outputs under the Reports Hub. Do not close if any tracked selector collects zero tests.

## Phase A — CLI Guardrail & Smoke Coverage
### Checklist
- [ ] A1: Update `ptycho/workflows/components.py::parse_arguments` to detect all `typing.Literal[...]` fields (not just ModelConfig) and register their choices so `--torch_loss_mode` maps cleanly into `TrainingConfig`.
- [ ] A2: Add/refresh CLI regression tests under `tests/test_cli_args.py` or a new targeted module to cover `--torch_loss_mode` and another representative Literal flag (e.g., backend) for `run_baseline.py`.
- [ ] A3: Re-run `pytest tests/test_integration_baseline_gs2.py -vv` and attach stdout/stderr logs to the Reports Hub; verify TensorFlow baseline completes when datasets are present, otherwise mark the test as skipped with clear messaging.

### Pending Tasks (Engineering)
- Draft parser helper for Literal detection and ensure path arguments still coerce correctly.
- Add a pytest fixture mirroring the failing CLI invocation from `tests/test_integration_baseline_gs2.py` to guard against regressions.
- Capture fresh logs for the baseline integration selector.

### Notes & Risks
- When editing `ptycho/workflows/components.py`, avoid touching `ptycho/model.py`, `ptycho/diffsim.py`, or `ptycho/tf_helper.py` (stable modules per CLAUDE.md).
- The integration test requires fly64 data; if absent, ensure the skip path remains intact with a descriptive reason.

## Phase B — Phase G Orchestrator Test Realignment
### Checklist
- [ ] B1: Refactor `tests/study/test_phase_g_dense_orchestrator.py` fixtures to seed minimal `patched_train.npz`/`patched_test.npz` stubs (can be tiny arrays) so the programmatic Phase D call succeeds without real data.
- [ ] B2: Update collect-only assertions to look for the programmatic sentinel + new reporting helper commands instead of hard-coded CLI module names.
- [ ] B3: Adjust execution-mode tests to patch `generate_overlap_views`, `run_command`, and the reporting helper outputs while respecting the new highlight inventory requirements (aggregate_report, metrics digest, verify artifacts).
- [ ] B4: Run `pytest tests/study/test_phase_g_dense_orchestrator.py -vv` and store logs + any helper scripts in the Reports Hub.

### Pending Tasks (Engineering)
- Build utility helpers in the test module for creating temporary NPZs with the keys expected by `generate_overlap_views`.
- Ensure monkeypatched `run_command` stubs create artifacts for aggregate reports, digest, SSIM grid, and verifiers to keep the pipeline moving.
- Capture collector logs demonstrating the new assertions.

### Notes & Risks
- Large temporary NPZs can slow tests; keep generated arrays small (e.g., two positions) and place them under the pytest tmp_path.
- The orchestrator enforces AUTHORITATIVE_CMDS_DOC—tests must set this env var.

## Phase C — PyTorch Workflow Test Resync
### Checklist
- [ ] C1: Update `tests/torch/test_cli_inference_torch.py::test_accelerator_flag_roundtrip` to use real `torch.Tensor` outputs (or a lightweight tensor subclass) so `_run_inference_and_reconstruct` math succeeds; verify device `.to()`/`.eval()` calls remain asserted.
- [ ] C2: Refresh `tests/test_workflow_components.py::TestLoadInferenceBundle::test_load_valid_model_directory` to accept the `DiffractionToObjectAdapter` wrapper (assert on `.inner_model` or via `isinstance`).
- [ ] C3: Modernize `tests/torch/test_workflows_components.py` stubs to accept the `execution_config` kwarg, patch `_build_lightning_dataloaders` to return sentinel loaders, and provide Lightning module stubs exposing `automatic_optimization`, `val_loss_name`, and other fields touched in `_train_with_lightning`.
- [ ] C4: Cover execution-config threading by asserting the `Trainer` spy receives accelerator/deterministic/accumulate_grad_batches values and that gradient-accumulation guardrails raise when `automatic_optimization=False`.
- [ ] C5: Run `pytest tests/torch/test_cli_inference_torch.py tests/torch/test_workflows_components.py tests/test_workflow_components.py::TestLoadInferenceBundle::test_load_valid_model_directory -vv` and archive logs under the Reports Hub.

### Pending Tasks (Engineering)
- Implement helper classes inside the tests to mimic minimal Lightning modules/dataloaders with the attributes the real code inspects.
- Ensure patched functions/fixtures restore the original symbols after each test (pytest monkeypatch handles cleanup).
- Capture the pytest logs plus a short rationale in `reports/.../summary.md`.

### Notes & Risks
- The PyTorch suite imports `torch`/`lightning`; guard tests with `pytest.importorskip` where appropriate to avoid false negatives on machines without GPU support.
- Keep mocks aligned with POLICY-001 (PyTorch >= 2.2) expectations; do not silence real import errors during implementation.

## Deprecation / Policy Banner
- Maintain CONFIG-001: always run `update_legacy_dict(params.cfg, config)` before invoking legacy modules. Document this in test helpers where needed (especially for PyTorch workflows).
- Maintain POLICY-001: do not skip PyTorch requirements silently; tests should surface actionable skip/xfail reasons when torch/lightning are absent.

## Artifacts Index
- Reports root: `plans/active/FIX-PYTEST-SUITE-REALIGN-001/reports/`
- Latest run: `2025-11-14T000000Z/test_cleanup/`

## Open Questions & Follow-ups
- Do we need a shared pytest fixture for building fake Phase G artifacts (aggregate reports, digests) to reuse across tests?
- After the PyTorch test overhaul, should we upstream reusable Lightning stubs into `tests/torch/conftest.py` for future initiatives?
