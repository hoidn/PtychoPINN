Summary: Wire the PyTorch regression onto the new minimal fixture and clear the remaining smoke-test failures.
Mode: TDD
Focus: [TEST-PYTORCH-001] Author PyTorch integration workflow regression — Phase B3 fixture wiring
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_fixture_pytorch_integration.py -vv, pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/{summary.md,pytest_fixture_green.log,pytest_integration_fixture.log}

Do Now:
1. TEST-PYTORCH-001 B3.A @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Update `tests/torch/test_fixture_pytorch_integration.py` smoke tests to load via `raw_data.diff3d` and import `PtychoDataset` from `ptycho_torch.dataloader`, then point `tests/torch/test_integration_workflow_torch.py` `data_file` fixture and CLI overrides at `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` (preserve CONFIG-001 ordering, keep deterministic CPU flags); tests: none.
2. TEST-PYTORCH-001 B3.B @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md — Run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_fixture_pytorch_integration.py -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/pytest_fixture_green.log` and `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/pytest_integration_fixture.log`; update plan rows B3.A/B3.B to `[x]`, flip implementation.md B2→B3 status, and append docs/fix_plan Attempt summarizing the GREEN evidence; tests: pytest tests/torch/test_fixture_pytorch_integration.py -vv, pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv.

If Blocked: Capture failing command output under `plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/`, leave B3 rows `[P]`, and log the blocker (with traceback + hypothesis) in docs/fix_plan Attempts plus plan.md notes before exiting the loop.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md B3 rows — Only remaining blockers before Phase B can close.
- specs/data_contracts.md §1 — Smoke tests must assert against canonical DATA-001 keys after generator changes.
- docs/workflows/pytorch.md §§4–8 — Confirms RawData expectations and CLI guardrails when swapping datasets.
- docs/findings.md (POLICY-001, FORMAT-001) — PyTorch dependency + legacy `(H,W,N)` transpose policy inform fixture handling.
- plans/active/TEST-PYTORCH-001/implementation.md Phase B — Keeps initiative ledger synchronized with new artifact paths.

How-To Map:
- Adjust smoke assertions to use `raw_data.diff3d` or `raw_data.diffraction` helper accessors; reference `ptycho/raw_data.py` lines 296-332 for available fields.
- Import `PtychoDataset` from `ptycho_torch.dataloader` and instantiate with the generated fixture to ensure loader compatibility.
- Update integration test `data_file` fixture to point at `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz`; align CLI flags (`--n_images`, `--max_epochs`, seeds) with Phase B1 scope so runtime stays <45s.
- After edits, run the two mapped pytest selectors with `CUDA_VISIBLE_DEVICES=""` and capture logs using `tee` into the new artifact hub.
- Record runtime deltas and observations in `plans/active/TEST-PYTORCH-001/reports/2025-10-19T233500Z/phase_b_fixture/summary.md`, then mark B3.A/B3.B `[x]`, update implementation.md, and append docs/fix_plan Attempt with artifact references.

Pitfalls To Avoid:
- Do not delete the legacy canonical dataset; keep references to both fixtures in docs.
- Avoid reintroducing randomness—if sampling parameters are changed, document seeds.
- Keep pytest selectors targeted; do not run full suite unless necessary at loop end.
- Ensure new imports stay local to tests (avoid modifying production modules unless required).
- Do not modify core physics modules (`ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Preserve existing RED log artifacts; store new GREEN logs only under the 2025-10-19T233500Z hub.
- Maintain ASCII-only updates when editing plan/docs.
- Re-run `update_legacy_dict` guard checks if integration test CLI flow diverges; document any deviations.
- Update plan/docs before finishing even if tests stay red.
- Keep fixture NPZ/JSON under version control; do not relocate without updating metadata.

Pointers:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T214052Z/phase_b_fixture/plan.md
- plans/active/TEST-PYTORCH-001/implementation.md
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T225900Z/phase_b_fixture/fixture_notes.md
- specs/data_contracts.md
- docs/workflows/pytorch.md
- docs/findings.md#policy-001
- docs/findings.md#FORMAT-001

Next Up: 1. TEST-PYTORCH-001 B3.C — document runtime delta and update workflows once fixture integration stays green.
