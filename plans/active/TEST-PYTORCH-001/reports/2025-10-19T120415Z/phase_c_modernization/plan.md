# Phase C — PyTorch Integration Test Modernization (pytest)

## Context
- Initiative: TEST-PYTORCH-001 — PyTorch integration regression guard
- Phase Goal: Replace the legacy `unittest.TestCase` harness with a native pytest workflow that enforces the train→save→load→infer contract while capturing deterministic environment setup and artifact validation.
- Dependencies:
  - `plans/active/TEST-PYTORCH-001/implementation.md` (Phase checklist owner)
  - `plans/pytorch_integration_test_plan.md` (charter scope + acceptance criteria)
  - `specs/ptychodus_api_spec.md` §4.5–§4.6 (reconstructor lifecycle contract)
  - `docs/workflows/pytorch.md` §§5–8 (CLI + deterministic settings)
  - Findings: POLICY-001 (PyTorch mandatory), FORMAT-001 (legacy NPZ transpose guard)
- Baseline Evidence: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/{inventory.md,pytest_integration_current.log,summary.md}`

## TDD Strategy
We will author a new pytest-style regression in `tests/torch/test_integration_workflow_torch.py` using fixtures and helper functions. The RED phase introduces a helper stub that raises `NotImplementedError`, guaranteeing a failing test. The GREEN phase implements the helper by reusing the proven subprocess workflow from the legacy unittest harness. Validation then confirms artifact expectations and updates documentation to reflect GREEN status.

### Phase C1 — RED: Pytest Skeleton & Helper Stub
Goal: Establish pytest-native structure and ensure the new test fails before porting implementation.
Prereqs: Phase A baseline complete; confirm current unittest passes (see baseline log); no code modifications pending in same file.
Exit Criteria: Pytest version of the integration test exists, legacy unittest path disabled, and the new pytest selector fails with a captured RED log referencing the stubbed helper.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1.A | Draft pytest module scaffolding | [x] | ✅ 2025-10-19 — Converted module to pytest style with fixtures (`cuda_cpu_env`, `data_file`), updated docstring references, and skipped legacy unittest class. |
| C1.B | Introduce helper stub | [x] | ✅ 2025-10-19 — Added `_run_pytorch_workflow` helper that raises `NotImplementedError("PyTorch pytest harness not implemented (Phase C1 stub)")` to force RED failure. |
| C1.C | Author failing pytest test | [x] | ✅ 2025-10-19 — Authored `test_run_pytorch_train_save_load_infer` calling the stub and documenting target assertions. Helper raises before assertions execute, ensuring the RED failure. |
| C1.D | Capture RED run | [x] | ✅ 2025-10-19 — Executed `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv` and stored log at `plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/pytest_modernization_red.log`. Legacy unittest class marked skipped to prevent duplicate runtime. |

### Phase C2 — GREEN: Helper Implementation & Deterministic Harness
Goal: Implement the helper to mirror the successful subprocess workflow, ensuring deterministic CPU execution and artifact checks.
Prereqs: Phase C1 RED artifacts committed; helper stub present; legacy unittest either removed or skipped.
Exit Criteria: Pytest test passes, reproducing previous assertions; helper encapsulates subprocess commands with environment controls; logs stored.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C2.A | Implement `_run_pytorch_workflow` | [x] | ✅ 2025-10-19 — Ported subprocess logic from legacy unittest (commit 77f793c). Helper invokes train/infer with CLI args, propagates CUDA_VISIBLE_DEVICES="", surfaces stdout/stderr on failure, and returns SimpleNamespace with artifact paths (lines 65-161). |
| C2.B | Update pytest assertions | [x] | ✅ 2025-10-19 — Assertions execute and pass: checkpoint exists, recon_amp/recon_phase exist, file sizes >1KB. No changes needed to test body (lines 188-196) as original assertions were already correct. |
| C2.C | Remove/skips legacy unittest | [x] | ✅ 2025-10-19 — Legacy `TestPyTorchIntegrationWorkflow` class already skipped in Phase C1 (lines 133-151). Module docstring updated from "Phase C1 (RED)" to "Phase C2 (GREEN)" (lines 1-21). Test docstring updated with GREEN behavior expectations (lines 172-187). |
| C2.D | Capture GREEN run | [x] | ✅ 2025-10-19 — Executed `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv` and captured log via tee to `plans/active/TEST-PYTORCH-001/reports/2025-10-19T122449Z/phase_c_modernization/pytest_modernization_green.log`. Result: 1 PASSED in 35.86s. |

### Phase C3 — Validation & Documentation Alignment
Goal: Confirm artifacts, update auxiliary documentation, and mark plan/fix-plan progress.
Prereqs: GREEN log captured; pytest test stable.
Exit Criteria: Artifact audit recorded, documentation updated, fix plan attempt logged with links.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C3.A | Artifact audit | [ ] | Inspect `training_output_dir` and `inference_output_dir` produced during the GREEN run. Document locations, file sizes, and dtype notes in `plans/active/TEST-PYTORCH-001/reports/<TS>/phase_c_modernization/artifact_audit.md`. |
| C3.B | Update documentation | [ ] | Refresh `plans/pytorch_integration_test_plan.md` to mark open questions resolved (CLI exists, helpers implemented) and update runtime expectations. Update module docstring in `tests/torch/test_integration_workflow_torch.py` to reflect GREEN status. |
| C3.C | Ledger updates | [ ] | Append Attempt entry in `docs/fix_plan.md` summarizing RED→GREEN TDD, log paths, and documentation updates. Update `plans/active/TEST-PYTORCH-001/implementation.md` checklist rows (C1–C3) to `[x]`. |

## Artifact Discipline
- Use timestamp directory `plans/active/TEST-PYTORCH-001/reports/<TS>/phase_c_modernization/` for all logs and notes.
- Recommended filenames:
  - `pytest_modernization_red.log`
  - `pytest_modernization_green.log`
  - `artifact_audit.md`
  - `summary.md` (loop-level narrative)

## Command Reference
- RED run selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv`
- GREEN run selector: same as above (post implementation)
- Full regression (optional, post-change): `pytest tests/torch/test_integration_workflow_torch.py -vv`
- Logs must be captured via `tee` or shell redirection into the artifact directory.

## Risks & Mitigations
- **Runtime Regression (>120s)**: If runtime increases, revisit dataset fixture or adopt Phase B minimization plan.
- **Artifact Drift**: Ensure helper returns the actual output paths so assertions stay coupled to produced artifacts; update audit quickly if CLI paths change.
- **Environment Leakage**: Use context managers to temporarily modify `os.environ` within helper; restore original values to avoid cross-test pollution.

## Next Steps After Phase C
- Proceed to Phase D (documentation + CI integration) as outlined in the main implementation plan once C1–C3 tasks are complete and logged.
