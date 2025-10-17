Summary: Publish torch-required policy in spec and findings
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 – Phase F4.2 Sync specs & findings
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T204818Z/{spec_sync.md}

Do Now:
1. INTEGRATE-PYTORCH-001 F4.2.A @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md — edit `specs/ptychodus_api_spec.md` per spec_sync_brief (tests: none)
2. INTEGRATE-PYTORCH-001 F4.2.B @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md — append POLICY-001 row to `docs/findings.md` (tests: none)
3. INTEGRATE-PYTORCH-001 F4.2.C @ plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md — update cross-links (CLAUDE.md + spec refs) and log verification in `spec_sync.md` (tests: none)
4. INTEGRATE-PYTORCH-001 Phase F reporting — update `plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md` + docs/fix_plan.md Attempts with new artifacts (tests: none)

If Blocked: Capture the blocker, partial diffs, and open questions in `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T204818Z/spec_sync.md`, then ping supervisor for guidance before editing further.

Priorities & Rationale:
- `specs/ptychodus_api_spec.md:1` needs PyTorch requirement per Phase F governance (`governance_decision.md`) and plan F4.2.A.
- `docs/findings.md:1` must track POLICY-001 so directive changes have an authoritative pointer.
- CLAUDE directive at `CLAUDE.md:57` should reference POLICY-001 to close the documentation loop.
- `plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md` requires spec_sync.md evidence before marking F4.2 complete.
- Ledger alignment keeps `docs/fix_plan.md` and Phase F table authoritative for next loops.

How-To Map:
- Spec edits: add a post-list paragraph in Section 1 (after the two-component list) stating PyTorch >=2.2 is mandatory, link to `docs/findings.md#policy-001` and the governance report; expand Section 2.3 to cover `ptycho_torch.config_bridge.*` adapter contracts; augment Section 4.2 bullets with fail-fast torch import expectations and reference the finding.
- Knowledge base: append `POLICY-001 | 2025-10-17 | policy, PyTorch, dependencies | ... | [Link](plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md) | Active` to the findings table (maintain alphabetical ID order by prefix).
- Cross-ref: update CLAUDE directive block (line 57) to include `<doc-ref type="findings">docs/findings.md#policy-001</doc-ref>`; ensure spec paragraphs added above include same anchor; document verification steps + file:line anchors in `spec_sync.md` under the new timestamp directory.
- Reporting: in `spec_sync.md`, summarize each change (spec, findings, CLAUDE) with anchors and confirmation checklist; mark `phase_f_torch_mandatory.md` F4.2 `[x]` once evidence saved; log Attempt in `docs/fix_plan.md` referencing `2025-10-17T204818Z/spec_sync.md`.

Pitfalls To Avoid:
- Do not move or renumber existing finding IDs; append the new row at the top-level table.
- Avoid speculative statements; align spec text with actual adapter behavior (`ptycho_torch.config_bridge` function names).
- Keep doc-ref tags precise; ensure anchors exist after edits.
- Preserve Markdown table alignment in docs/findings.md (use pipes and spacing consistently).
- Do not modify Phase F4.3 checklist; that work is for a later loop.
- Ensure CLAUDE directive wording remains within current policy bounds (no reintroducing torch-optional language).
- No tests to run; do not invoke pytest.

Pointers:
- specs/ptychodus_api_spec.md:73 (Section 3 intro) for placement context.
- CLAUDE.md:57 (PyTorch directive) for cross-link update.
- docs/findings.md:1 (knowledge base table header) for formatting.
- plans/active/INTEGRATE-PYTORCH-001/phase_f4_doc_sync.md:34 (F4.2 task descriptions).
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T204818Z/spec_sync_brief.md (supervisor prep notes).

Next Up: INTEGRATE-PYTORCH-001 F4.3 (handoff notes) once spec/finding sync is complete.
