Summary: Capture TensorFlow vs PyTorch integration logs and publish Phase E2.D parity summary
Mode: Parity
Focus: INTEGRATE-PYTORCH-001 – Phase E2 Integration Regression & Parity Harness (E2.D)
Branch: feature/torchapi
Mapped tests: pytest tests/test_integration_workflow.py -k full_cycle -vv; pytest tests/torch/test_integration_workflow_torch.py -vv
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/{phase_e_tf_baseline.log,phase_e_torch_run.log,phase_e_parity_summary.md}

Do Now:
1. INTEGRATE-PYTORCH-001 E2.D1 @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — run pytest tests/test_integration_workflow.py -k full_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_tf_baseline.log (tests: targeted)
2. INTEGRATE-PYTORCH-001 E2.D2 @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — run pytest tests/torch/test_integration_workflow_torch.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_torch_run.log (tests: targeted)
3. INTEGRATE-PYTORCH-001 E2.D3 @ plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md — draft phase_e_parity_summary.md, update E2.D rows + docs/fix_plan.md Attempt with artifacts (tests: none)

If Blocked: Archive whichever pytest log captures the failure/skip, document root cause + next-step hypothesis inside phase_e_parity_summary.md, flip the affected plan rows back to [P], and log the blocker in docs/fix_plan.md Attempts history referencing the same timestamp.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md Phase D — defines required logs + summary artifacts for E2.D.
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md E2.D rows — parity evidence is the remaining gate before docs/spec sync.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T215500Z/phase_e2_green.md — details current CLI wiring assumptions that the parity run must validate.
- specs/ptychodus_api_spec.md §4.5 — reconstructor contract mandates matching train→save→load→infer lifecycle across backends.

How-To Map:
- timestamp=2025-10-18T093500Z; mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp
- pytest tests/test_integration_workflow.py -k full_cycle -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/phase_e_tf_baseline.log
- pytest tests/torch/test_integration_workflow_torch.py -vv | tee plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/phase_e_torch_run.log
- If torch run skips/fails, keep full output in the log and note skip reason and stack trace in phase_e_parity_summary.md
- Write plans/active/INTEGRATE-PYTORCH-001/reports/$timestamp/phase_e_parity_summary.md summarizing outcomes, metrics, artifacts created, and residual risks; cite CONFIG-001/POLICY-001 compliance
- Update plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md (E2.D1–E2.D3) and plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md (E2.D rows) to [x]; adjust implementation.md E2 row if parity complete
- Log docs/fix_plan.md attempt with selector results + artifact paths before ending the loop

Pitfalls To Avoid:
- Do not run full pytest suite; stick to mapped selectors.
- Keep artifact files inside the timestamped reports directory only.
- Do not edit CLI implementation files this loop—evidence gathering only.
- Ensure pytest commands run from repo root so imports resolve correctly.
- Capture stdout/stderr even for skipped tests; they document environment capability.
- Confirm params.cfg reset after TensorFlow baseline run to avoid cross-test contamination.
- Avoid deleting prior reports; append new evidence alongside existing history.
- Do not modify TensorFlow-only fixtures when adapting parity summary.
- If PyTorch runtime is unavailable, report the precise ImportError/skip reason verbatim.
- Preserve consistent ISO timestamp usage across all referenced artifacts.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md
- plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T215500Z/phase_e2_green.md
- specs/ptychodus_api_spec.md

Next Up: Phase E3 documentation/spec sync once parity evidence is archived; or revisit `[INTEGRATE-PYTORCH-001-STUBS]` to plan Lightning/stitching implementations.
