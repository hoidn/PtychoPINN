Summary: Remove the Poisson support violation so Lightning training can run end-to-end on the PyTorch workflow.
Mode: Parity
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase C4 CLI integration (C4.D3)
Branch: feature/torchapi
Mapped tests: CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_poisson_count_contract -vv; CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/{poisson_failure_summary.md,manual_train_cli.log,pytest_poisson_red.log,pytest_poisson_green.log,pytest_integration.log}
Do Now:
1. ADR-003-BACKEND-API C4.D3 Poisson RED test @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — extend `tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining` with `test_lightning_poisson_count_contract` that exercises `_build_lightning_dataloaders()` + `PtychoPINN_Lightning.compute_loss()` on the minimal container (expect the current build to raise the Poisson `ValueError`), then run `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_poisson_count_contract -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/pytest_poisson_red.log`.
2. ADR-003-BACKEND-API C4.D3 Poisson fix @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — align the PyTorch Poisson loss path with the TensorFlow implementation (replicate the `tf.nn.log_poisson_loss` behaviour so amplitudes from `batch[0]['images']` are scaled/squared appropriately before evaluating the log-likelihood) and ensure the regression from step 1 now passes, then rerun the same selector piping to `pytest_poisson_green.log`.
3. ADR-003-BACKEND-API C4.D3 integration validation @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — regenerate the minimal fixture if needed, rerun `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/pytest_integration.log`, and confirm Lightning training completes and produces the bundle.
4. ADR-003-BACKEND-API C4.D3 ledger wrap-up @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md — mark the plan row `[x]` once Poisson parity is restored, update the summary with the scaling decision, and log the attempt in docs/fix_plan.md referencing the 2025-10-20T070610Z hub.
If Blocked: Capture the failing command output to `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/blocker.log`, annotate C4.D3 back to `[P]` with the blocker description, and document the obstacle (scaling discrepancies, fixture generation issues, etc.) before stopping.
Priorities & Rationale:
- C4.D3 plan row @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md#L96 — integration remains blocked until Lightning can evaluate Poisson loss without throwing.
- poisson_failure_summary.md — captures the support violation and points to the command we need to make green.
- specs/ptychodus_api_spec.md:§4.6 — persistence contract assumes Poisson-based physics loss stays consistent between TF and Torch.
- docs/workflows/pytorch.md:§12 — documents execution config + scaling expectations (CONFIG-001 parity) that have to hold once we adjust the loss.
- ptycho/loader.py:355 — reminds us normalization happens pre-Poisson in TF and must be mirrored when we discretise for Torch.
How-To Map:
- Generate the minimal dataset when absent: `python scripts/tools/make_pytorch_integration_fixture.py --source datasets/Run1084_recon3_postPC_shrunk_3.npz --output tests/fixtures/pytorch_integration/minimal_dataset_v1.npz --subset-size 64` (delete afterwards if you do not want it tracked).
- RED selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_workflows_components.py::TestWorkflowsComponentsTraining::test_lightning_poisson_count_contract -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/pytest_poisson_red.log`.
- GREEN selector: repeat with the same command teeing to `pytest_poisson_green.log` after implementing the scaling fix.
- Integration selector: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/pytest_integration.log`.
Pitfalls To Avoid:
- Do not leave generated fixtures committed; keep them for local runs or clean up afterwards.
- Preserve CONFIG-001 ordering (`update_legacy_dict`) when touching loaders to avoid hidden gridsize drift.
- Keep Poisson scaling logic device-agnostic (no hard-coded CPU tensors).
- Ensure new regression test uses native pytest style (no unittest mixins or global state hacks).
- Avoid changing TensorFlow-side code unless the spec explicitly demands it; focus on PyTorch parity.
- Guard against zero/negative rates by adding numerical epsilons when taking logs.
- Capture logs via `tee` into the artifact hub—no stray logs at repo root.
- Maintain complex dtype handling; do not cast probes or diffraction to float64.
- Re-run the targeted selectors before running the expensive integration test to save time.
Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T033100Z/phase_c4_cli_integration/plan.md#L96
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T070610Z/phase_c4_cli_integration_debug/poisson_failure_summary.md
- ptycho_torch/model.py:714
- ptycho_torch/workflows/components.py:332
- ptycho/loader.py:355
- specs/ptychodus_api_spec.md:240
Next Up: (1) C4.F documentation + ledger tidy once Poisson parity is green; (2) Phase D CLI thin wrapper cleanup after integration suite stays green.
