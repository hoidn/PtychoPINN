Summary: Capture Phase D runtime + environment profile for the PyTorch integration pytest.
Mode: Perf
Focus: TEST-PYTORCH-001 — Author PyTorch integration workflow regression
Branch: feature/torchapi
Mapped tests: pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv
Artifacts: plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/{runtime_profile.md,env_snapshot.txt,pytest_modernization_phase_d.log}

Do Now:
1. TEST-PYTORCH-001 D1.A @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Re-run the targeted selector, capture the full log, and note elapsed time (tests: pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv).
2. TEST-PYTORCH-001 D1.B @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Record environment telemetry (python/torch/lightning versions, CPU, RAM) into env_snapshot.txt (tests: none).
3. TEST-PYTORCH-001 D1.C @ plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Author runtime_profile.md with aggregated timings + guardrails, update summary.md with D1 findings, and flip the plan’s D1 row to [x] with references (tests: none).

If Blocked: If the pytest selector fails or exceeds 120s, keep the failing log under the same artifact hub, log hypotheses + reproduction steps in runtime_profile.md, and set plan D1 state to [P] with blocking notes; skip D1.B/C until the failure is resolved.

Priorities & Rationale:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md — Newly authored Phase D roadmap expects runtime/env evidence before downstream documentation.
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/artifact_audit.md — Provides prior runtime (35.98s) to cross-check against fresh measurements.
- docs/workflows/pytorch.md §§5–8 — Workflow guide needs authoritative runtime + selector info once D1 concludes.
- docs/findings.md#POLICY-001 — PyTorch mandatory dependency; runtime notes must confirm CPU-only path complies.
- tests/torch/test_integration_workflow_torch.py:65-205 — Helper implementation + assertions drive the runtime being measured.

How-To Map:
- Pytest rerun: `CUDA_VISIBLE_DEVICES="" pytest tests/torch/test_integration_workflow_torch.py::test_run_pytorch_train_save_load_infer -vv | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/pytest_modernization_phase_d.log`
- Environment capture (append outputs):
  - `python -V | tee plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/env_snapshot.txt`
  - `python -c "import torch, lightning; print(f'torch {torch.__version__}'); print(f'lightning {lightning.__version__}')" >> plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/env_snapshot.txt`
  - `pip show torch | sed -n '1,6p' >> plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/env_snapshot.txt`
  - `lscpu >> plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/env_snapshot.txt`
  - `grep MemTotal /proc/meminfo >> plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/env_snapshot.txt`
- Runtime profile: Summarize timings from the new log plus 2025-10-19T122449Z & 2025-10-19T130900Z logs, cite file names, and describe acceptable variance (≤90s CPU). Store narrative in runtime_profile.md and note guardrails in summary.md.
- Update plan: set `D1` row in implementation.md to `[x]` once documentation is saved; include bullet with runtime + env snapshot references.

Pitfalls To Avoid:
- Do not leave env_snapshot.txt empty—append all commands in order with labels.
- Keep CUDA disabled (`cuda_cpu_env` fixture expectation); do not enable GPUs for runtime capture.
- Avoid deleting prior Phase C artifacts; reference them instead.
- Don’t exceed artifact naming conventions—store new files in the Phase D hub only.
- When editing implementation.md, change only the D1 row state/comment.
- Ensure pytest log is captured via tee; without it runtime evidence is incomplete.
- Do not modify production code paths while gathering evidence.
- Keep runtime_profile.md factual—cite logs with paths and timestamps.
- Run commands from repo root to preserve relative paths.
- Ensure env snapshot command output is appended (use `>>`) after the first tee.

Pointers:
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/plan.md#phase-d1-runtime--resource-profile
- plans/active/TEST-PYTORCH-001/reports/2025-10-19T130900Z/phase_c_modernization/artifact_audit.md
- tests/torch/test_integration_workflow_torch.py:65-205
- docs/workflows/pytorch.md:120-210
- docs/findings.md#POLICY-001

Next Up: D2 (documentation + ledger updates) once runtime_profile.md exists.
