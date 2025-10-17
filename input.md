Summary: Capture torch-required handoff notes and owner matrix
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 – Phase F4.3 Handoff coordination
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T210328Z/{handoff_notes.md,owner_matrix.md,verification_commands.md}

Do Now:
1. INTEGRATE-PYTORCH-001 F4.3.A @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md + H1-H2 @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_handoff.md — create the 2025-10-17T210328Z/ report directory, draft `handoff_notes.md` with owner/action table covering TEST-PYTORCH-001, CI/CD, Ptychodus integration, and release versioning (tests: none)
2. INTEGRATE-PYTORCH-001 F4.3.B @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md + H3 @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_handoff.md — append verification cadence section detailing the required pytest selectors and expected outcomes, saving any supplemental lists to `verification_commands.md` (tests: none)
3. INTEGRATE-PYTORCH-001 F4.3.C @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md + H4 @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_handoff.md — update `phase_f_torch_mandatory.md` F4 row, mark relevant checkboxes, and log docs/fix_plan.md Attempt referencing the new artifact path with residual risks noted (tests: none)

If Blocked: Capture the blocker, partial notes, and outstanding questions inside `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T210328Z/handoff_notes.md`, then alert the supervisor before progressing.

Priorities & Rationale:
- `phase_f4_handoff.md:H1-H4` defines the authoritative deliverables for closing Phase F.
- `plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md` F4 row still open; must be updated with the new report for exit criteria.
- Governance record `reports/2025-10-17T184624Z/governance_decision.md §4-7` enumerates CI and communication expectations that the handoff must mirror.
- `plans/pytorch_integration_test_plan.md` provides context for TEST-PYTORCH-001 dependencies that belong in the owner matrix.
- POLICY-001 (`docs/findings.md:8`) anchors the torch-required policy; ensure handoff references align with it.

How-To Map:
- Make the artifact directory before editing: `mkdir -p plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T210328Z`.
- Owner matrix: in `handoff_notes.md`, create a table with columns `Initiative/Owner | Required Action | Blocking Dependencies | References`; source actions from `phase_f4_handoff.md` H1-H2 and governance §7.
- CI updates: document torch installation steps (e.g., add `pip install torch>=2.2` to CI runners) and specify validation command `python -c "import torch; print(torch.__version__)"` prior to pytest; note any environment constraints.
- Verification cadence: list the selectors verbatim (e.g., `pytest --collect-only tests/torch/ -q`, `pytest tests/torch/test_backend_selection.py -k pytorch_unavailable_raises_error -vv`, `pytest tests/torch/test_config_bridge.py -k parity -vv`) and state the expected result (collects cleanly or passes). Save any expanded rationale in `verification_commands.md`.
- Ledger updates: once notes are complete, mark F4.3 rows `[x]` in `phase_f4_doc_sync.md` and `phase_f_torch_mandatory.md`, then append a docs/fix_plan.md Attempt summarizing the artifacts and residual risks per H4 guidance.

Pitfalls To Avoid:
- Do not run pytest; only document the selectors and expectations.
- Keep artifact names and timestamps exact (`2025-10-17T210328Z`) so supervisors can trace evidence.
- Preserve Markdown table formatting in `handoff_notes.md` (pipe alignment, header separator).
- Reference actual file anchors (e.g., `plans/pytorch_integration_test_plan.md:1`) when citing documents.
- Avoid promising code changes—this loop is planning/Docs only.
- Do not adjust Phase F4.3 plan IDs; mark completion only after evidence saved.
- Ensure docs/fix_plan.md Attempt references the new report path to keep the ledger authoritative.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md:39
- plans/active/INTEGRATE-PYTORCH-001/phase_f4_handoff.md:18
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:60
- plans/pytorch_integration_test_plan.md:1
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md:120
- docs/findings.md:8

Next Up: If handoff closes early, prepare the v2.0.0 CHANGELOG entry using governance §5.4 guidance.
