Summary: Begin Phase C by converting the PyTorch integration workflow test to a pytest skeleton with a stubbed helper that fails (RED).
Mode: TDD
Focus: TEST-PYTORCH-001 — Phase C pytest modernization
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/{plan.md,summary.md,pytest_modernization_red.log}

Do Now:
1. TEST-PYTORCH-001 C1.A–C1.D @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md — Port `tests/torch/test_integration_workflow_torch.py` to pytest fixtures with `_run_pytorch_workflow` stub and run the selector (expect failure) capturing `pytest_modernization_red.log` (tests: pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv).
2. TEST-PYTORCH-001 C1.D follow-up @ docs/fix_plan.md — Append Attempt entry documenting the RED run, stub helper, and artifact path (tests: none).

If Blocked: If the selector errors before the stub is hit (e.g., import failure), capture the full traceback to `pytest_modernization_red.log`, revert only the pytest conversion, and note the failure and restoration in docs/fix_plan Attempts before exiting.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/implementation.md:41 — Phase C checklist now references the modernization plan; completing C1 unblocks GREEN work.
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md — Defines the RED helper stub strategy so we stay within TDD.
- tests/torch/test_integration_workflow_torch.py:1 — Legacy unittest harness to convert; docstrings must reflect GREEN status post-migration.
- specs/ptychodus_api_spec.md:180 — Reconstructor lifecycle contract the integration test enforces.
- docs/workflows/pytorch.md:120 — Captures deterministic CPU execution requirements (`CUDA_VISIBLE_DEVICES=""`).

How-To Map:
- Create a working branch backup if needed (`git status` should stay clean apart from intended edits).
- In `tests/torch/test_integration_workflow_torch.py`, replace `unittest.TestCase` with pytest-style fixtures (`pytest`, `tmp_path`, helper stub). Add `_run_pytorch_workflow` raising `NotImplementedError`.
- Mark the old unittest class skipped or remove it once the pytest function exists to avoid duplicate runtime.
- Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/pytest_modernization_red.log`.
- Update docs/fix_plan Attempt history with selector, failure message, and artifact path referencing C1 progress.

Pitfalls To Avoid:
- Do not let both unittest and pytest variants run simultaneously—ensure only the pytest path executes.
- Keep helper stub raising `NotImplementedError` for RED; do not implement the helper yet.
- Maintain artifact hygiene by writing logs only under `phase_c_modernization/`.
- Preserve existing CLI argument set; avoid editing production scripts this loop.
- Leave CUDA disabled for reproducibility by exporting `CUDA_VISIBLE_DEVICES=""` when running pytest.
- Avoid mixing unittest assertions with pytest `assert`; the new function should use native asserts.
- Do not mark Phase C checklist rows complete until RED evidence recorded in plan + fix plan.
- Ensure environment modifications (e.g., `os.environ`) are scoped; do not leak state beyond the test.
- Keep docstrings accurate—flag GREEN status change for next loop but do not rewrite them yet if causing merge noise.

Pointers:
- plans/active/TEST-PYTORCH-001/implementation.md#L40
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md
- tests/torch/test_integration_workflow_torch.py:1
- specs/ptychodus_api_spec.md:180
- docs/workflows/pytorch.md:120

Next Up: C2 (implement helper + GREEN run) after RED evidence is captured and logged.
