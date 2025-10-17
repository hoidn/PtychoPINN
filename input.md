Summary: Document the policy shift toward a torch-required backend and prep governance artifacts.
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 / Phase F1 Torch Mandatory Transition
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/{directive_conflict.md,governance_decision.md,guidance_updates.md}
Do Now:
- INTEGRATE-PYTORCH-001 Phase F1 — F1.1 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: none): summarize the current torch-optional directive footprint (CLAUDE.md:57, tests/conftest.py, plan notes) and articulate the new "torch-required" intent in `directive_conflict.md` with file:line anchors.
- INTEGRATE-PYTORCH-001 Phase F1 — F1.2 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: none): record the governance/stakeholder decision in `governance_decision.md`, including risks (CI availability, TEST-PYTORCH-001 impact) and mitigation steps for the torch-required shift.
- INTEGRATE-PYTORCH-001 Phase F1 — F1.3 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: none): draft the redline for CLAUDE.md/docs/findings.md updates in `guidance_updates.md`; enumerate exact wording changes and note any follow-up edits required in future loops.
If Blocked: Capture why governance approval or directive updates cannot proceed in `directive_conflict.md`, note blockers in docs/fix_plan.md Attempts History, and halt before touching CLAUDE.md.
Priorities & Rationale:
- `CLAUDE.md:57` still mandates torch-optional parity; we must document the conflict before altering behavior.
- `tests/conftest.py:24-52` encodes torch-optional skip logic that Phase F will remove; inventorying the dependency surface guides later implementation.
- `docs/fix_plan.md` Attempt #63 references torch-optional dispatcher; the new plan requires explicit rationale for breaking that precedent.
- `plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md` defines F1 deliverables and artifact expectations that keep the ledger consistent.
How-To Map:
- Use `rg -n "torch-optional" CLAUDE.md tests/conftest.py docs/fix_plan.md` to capture current directive language; quote relevant lines in `directive_conflict.md`.
- Summarize decision context referencing `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184417Z/phase_f_summary.md` and the fix_plan TODO removal.
- In `governance_decision.md`, include a risks table (CI dependency, developer workflow, TEST-PYTORCH-001) and proposed mitigations; cite `specs/ptychodus_api_spec.md` for contract alignment.
- In `guidance_updates.md`, list proposed edits (e.g., replace the torch-optional directive in CLAUDE.md with new wording, add finding entry) so the next loop can implement them directly.
- Store all artifacts under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/`; append references to docs/fix_plan.md Attempts History when finished.
Pitfalls To Avoid:
- Do not edit production code or run tests during this evidence loop.
- Do not modify CLAUDE.md yet; capture redlines first and wait for explicit supervisor confirmation.
- Keep artifacts torch-neutral (no assumptions about torch availability in example commands).
- Avoid duplicating existing plan content; reference `phase_f_torch_mandatory.md` instead.
- Ensure artifact filenames match those listed above; missing paths break traceability.
- Document open questions explicitly rather than leaving them implicit in prose.
Pointers:
- CLAUDE.md:57-64
- tests/conftest.py:1-60
- docs/fix_plan.md:59-140
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md
Next Up: After F1 artifacts, proceed to Phase F2 inventory (TORCH_AVAILABLE scan, skip audit).
