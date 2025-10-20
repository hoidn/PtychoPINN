Summary: Capture the PyTorch integration regression log and run the manual CLI smoke to unblock C4.D.
Mode: Parity
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4.D3+C4.D4
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/{pytest_integration.log,manual_cli_smoke.log,train_debug.log}
Do Now:
1. ADR-003-BACKEND-API C4.D3 targeted integration log @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/pytest_integration.log`; document the returned failure in the log header. tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
2. ADR-003-BACKEND-API C4.D4 manual CLI smoke @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — run `CUDA_VISIBLE_DEVICES="" python -m ptycho_torch.train --train_data_file tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --output_dir /tmp/cli_smoke --n_images 64 --max_epochs 1 --accelerator cpu --deterministic --num-workers 0 --learning-rate 1e-4 | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/manual_cli_smoke.log`; leave the bundled outputs under /tmp. tests: none
3. ADR-003-BACKEND-API C4.F hygiene wrap-up @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — move `train_debug.log` into the new artifact hub (or delete if redundant) and update `summary.md` with the captured failure signatures + manual run outcome. tests: none
If Blocked: Capture the failing stdout/stderr to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/blocker.log`, mark `C4.D3` back to `[P]` with the blocker summary, and log the obstacle in docs/fix_plan.md.
Priorities & Rationale:
- C4.D3 row @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — needs targeted log before we can close the regression gate.
- Supervisor review @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T074135Z/phase_c4_cli_integration_review/summary.md — highlights missing integration log and manual smoke coverage.
- specs/ptychodus_api_spec.md:§4.6 — bundle loading contract still fails; logging the failure keeps parity audit trail.
- docs/workflows/pytorch.md:§12 — documents CLI expectations; manual smoke validates that guidance.
How-To Map:
- Integration log: ensure `tests/fixtures/pytorch_integration/minimal_dataset_v1.npz` exists; regenerate with `python scripts/tools/make_pytorch_integration_fixture.py --source datasets/Run1084_recon3_postPC_shrunk_3.npz --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --subset-size 64` if missing.
- Manual CLI: reuse the same fixture; expect `load_torch_bundle` `NotImplementedError` until Phase D3 ships.
- Hygiene: `mv train_debug.log plans/active/ADR-003-BACKEND-API/reports/2025-10-20T081500Z/phase_c4_cli_integration_debug/train_debug.log` (or `rm` if obsolete) and note the action in `summary.md`.
Pitfalls To Avoid:
- Do not run the entire pytest suite; stick to the targeted selector.
- Keep `CUDA_VISIBLE_DEVICES=""` to avoid GPU-only path regressions.
- Store every new artifact under the 2025-10-20T081500Z hub; nothing at repo root.
- Do not overwrite prior logs (070610Z); create new files for this loop.
- Leave `/tmp/cli_smoke` contents untracked; only capture stdout via `tee`.
- Preserve CONFIG-001 ordering (config bridge before data loading) if you touch scripts.
- Avoid editing TensorFlow modules; PyTorch parity only.
- Do not delete the fixture unless you recreate it immediately.
Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T074135Z/phase_c4_cli_integration_review/summary.md
- specs/ptychodus_api_spec.md
- docs/workflows/pytorch.md
Next Up: C4.E documentation updates once C4.D artifacts are finalized.
