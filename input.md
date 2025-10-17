Summary: Finish Phase E2.D by installing torch extras and capturing the passing PyTorch integration log
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 – Phase E2 Integration Regression & Parity Harness (E2.D2)
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/{pip_install.log,phase_e_torch_run.log,phase_e_parity_summary.md}

Do Now:
1. INTEGRATE-PYTORCH-001 E2.D2 @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — run pip install -e .[torch] | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/pip_install.log (tests: none)
2. INTEGRATE-PYTORCH-001 E2.D2 @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — run pytest tests/torch/test_integration_workflow_torch.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T221500Z/phase_e_torch_run.log (tests: targeted)
3. INTEGRATE-PYTORCH-001 E2.D3 @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — update phase_e_parity_summary.md with new results, flip E2.D2 rows to [x], sync implementation.md + docs/fix_plan.md (tests: none)

If Blocked: Preserve pip/test logs under the artifact directory, note the failure reason + next hypothesis in the summary, revert any premature plan checkbox flips, and document the blocker in docs/fix_plan.md Attempts history.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md: D2 remains ⚠️ until PyTorch run succeeds after installing extras.
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md: E2.D rows depend on capturing the green PyTorch log and documenting parity.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_parity_summary.md: baseline evidence pinpointed the missing mlflow dependency; needs refresh after rerun.
- specs/ptychodus_api_spec.md §4.5: reconstructor contract requires PyTorch CLI parity with TensorFlow workflow.
- docs/workflows/pytorch.md §2: PyTorch extras (mlflow, lightning, tensordict) are mandatory for backend parity.

How-To Map:
- export timestamp=2025-10-17T221500Z; mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp
- pip install -e .[torch] | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/pip_install.log
- pytest tests/torch/test_integration_workflow_torch.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/phase_e_torch_run.log
- Capture runtime + key metrics from the passing test in plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/phase_e_parity_summary.md (reference prior summary; highlight new success + parity checks)
- Update plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md (set D2 → [x]), plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md (E2.D2/E2 row → [x]), and plans/active/INTEGRATE-PYTORCH-001/implementation.md (E2 row → [x])
- Append docs/fix_plan.md Attempt entry with selector results, pip command, artifact paths, and note that Phase E2.D is complete

Pitfalls To Avoid:
- Do not skip pip install; mlflow/lightning must import before rerunning pytest.
- Run commands from repo root so editable install resolves correctly.
- Keep all logs inside the timestamped reports directory; no stray files under repo root.
- Avoid rerunning the TensorFlow baseline unless PyTorch rerun forces shared fixture reset.
- If pytest still fails, do not mark plan rows complete; leave detailed blocker notes instead.
- Don’t delete prior 2025-10-18T093500Z artifacts—they document the red run.
- Ensure parity summary cites CONFIG-001 and POLICY-001 for compliance.
- After pip install, confirm command succeeded (exit code 0) before launching pytest.
- Use native pytest style if follow-up tests are required; do not mix unittest classes.
- Capture PyTorch runtime stats (epochs, checkpoint path) in the summary for parity comparison.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_parity_summary.md
- specs/ptychodus_api_spec.md#L1
- docs/workflows/pytorch.md#L1

Next Up: Phase E3 documentation/spec sync or reopen `[INTEGRATE-PYTORCH-001-STUBS]` once parity evidence is green.
