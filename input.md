Summary: Capture a current TensorFlow ↔ PyTorch parity map for INTEGRATE-PYTORCH-001.
Mode: Docs
Focus: [INTEGRATE-PYTORCH-001] Prepare for PyTorch Backend Integration with Ptychodus
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/{parity_map.md,summary.md}
Do Now: [INTEGRATE-PYTORCH-001] Phase A — produce parity_map.md and companion summary.md covering tasks A1–A3 (no tests yet).
If Blocked: Log missing information or unresolved questions in summary.md, reference the source gap, and update docs/fix_plan Attempts History before ending the loop.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:11 — Phase A defines the parity inventory deliverable we need before code changes.
- specs/ptychodus_api_spec.md:127 — Normative reconstructor contract; every touchpoint inventoried must trace back here.
- ptycho/workflows/components.py:543 — Source of TensorFlow workflow calls we must mirror in PyTorch.
- plans/pytorch_integration_test_plan.md:15 — Upcoming PyTorch integration test relies on the same touchpoint map.
- docs/architecture.md:19 — Data flow diagram ensures grouping/persistence steps stay aligned.
How-To Map:
- mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z
- rg -n "PtychoPINN" ptychodus/src/ptychodus/model/ptychopinn -g"*.py" >> plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/parity_map.md
- python - <<'PY' to append structured sections (TF touchpoints, PyTorch equivalents, gaps, owners) into parity_map.md using insights from specs/ptychodus_api_spec.md and ptycho_torch/* modules
- Summarize findings, risks, and next recommended steps in plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T020000Z/summary.md (include pointers to any missing counterparts)
- Record artifact path and key conclusions in docs/fix_plan.md Attempts History when done
Pitfalls To Avoid:
- Do not edit production code or tests in this loop; documentation only.
- Avoid inventing new commands; cite actual sources (specs, code) for every touchpoint you log.
- Keep artifacts ASCII and under version control; no external note pads.
- Maintain config terminology from spec (e.g., `TrainingConfig`, `PtychoDataContainer`) to avoid drift.
- Capture unresolved questions explicitly; don't leave TODOs unreferenced.
- Respect artifact directory naming; do not overwrite previous reports.
- Stay within evidence scope—no pytest runs unless a new test is authored later under TDD.
- Reference docs/findings.md before assuming behaviour differences.
Pointers:
- specs/ptychodus_api_spec.md:127
- plans/active/INTEGRATE-PYTORCH-001/implementation.md:11
- ptycho/workflows/components.py:543
- plans/pytorch_integration_test_plan.md:15
- docs/architecture.md:19
Next Up (optional): Phase B — design config bridge adapter once parity map is complete; coordinate with TEST-PYTORCH-001 to share fixtures.
