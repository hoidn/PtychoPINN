Summary: Capture EB3 logger options so we can choose a governance path before touching code.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Phase EB3 — Logger backend decision
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/{analysis/current_state.md,analysis/options_matrix.md,decision/proposal.md}
Do Now:
- EB3.A1 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — audit existing logging hooks and legacy expectations; tests: none.
- EB3.A2 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — build logger options matrix with dependency analysis; tests: none.
- EB3.A3 @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — draft decision proposal (pros/cons, acceptance criteria) for supervisor review; tests: none.
If Blocked: Document open questions + blocker evidence in `decision/proposal.md`, leave plan rows `[P]`, and log blockers in docs/fix_plan.md before exiting.
Priorities & Rationale:
- spec/ptychodus_api_spec.md:281 — `logger_backend` remains TBD; need decision context before editing code.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/open_questions.md:60-104 — captures MLflow vs execution config split that Phase EB3 must resolve.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md — authoritative checklist for EB3 Phase A deliverables.
- docs/workflows/pytorch.md:365 — current CLI guide references spec for execution config; logger decision will change this section.
- docs/fix_plan.md:48-65 — latest attempts show EB2 completion and pending EB3 planning; keep ledger synced with new artifacts.
How-To Map:
- For EB3.A1 run `rg -n "mlflow" ptycho_torch` and `rg -n "logger" ptycho_torch`; inspect `ptycho_torch/workflows/components.py` + `ptycho_torch/api/` legacy scripts. Summarize findings in `analysis/current_state.md` (include file:line references and notes on TODOs).
- For EB3.A2 consult Lightning logger docs and existing extras (setup.py). Create `analysis/options_matrix.md` with columns: Option, Dependencies, Pros, Cons, CI impact, User workflow impact. Use bullet list or table format.
- For EB3.A3 write `decision/proposal.md` outlining recommended path (enable logger(s) or formal deprecation), success criteria, required code/test/doc work, and questions needing supervisor sign-off. Link back to A1/A2 artifacts and note any dependency installation implications.
Pitfalls To Avoid:
- Do not edit production code or tests in this loop.
- Keep all notes under the timestamped artifact directory; no stray files at repo root.
- Record exact file paths/line numbers when citing current behaviour.
- Clearly flag assumptions about optional dependencies (MLflow, TensorBoard); avoid promising availability without evidence.
- Maintain ASCII tables/markdown; no rich formatting or non-ASCII characters.
- Leave plan checkboxes `[P]` until evidence completed; supervisor will flip later.
- Ensure proposal distinguishes between canonical config and execution config responsibilities (per open_questions.md Q2).
- Capture blocker list if decision cannot be reached; do not proceed to implementation steps.
- Reference POLICY-001 (PyTorch required) if suggesting new install requirements.
- Include citation to Lightning docs in proposal (URL already in plan.md context).
Pointers:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/open_questions.md:60-104
- specs/ptychodus_api_spec.md:260-320
- docs/workflows/pytorch.md:320-370
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T153300Z/phase_e_execution_knobs/2025-10-23T110500Z/plan.md
Next Up: Phase EB3.B implementation once decision proposal approved.
