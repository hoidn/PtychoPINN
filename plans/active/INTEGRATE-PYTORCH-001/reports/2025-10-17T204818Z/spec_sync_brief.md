# Phase F4.2 Prep Notes — Spec & Findings Synchronization

**Initiative:** INTEGRATE-PYTORCH-001  
**Phase:** F4.2 (Spec & Findings)  
**Author:** galph (supervisor)  
**Timestamp:** 2025-10-17T204818Z

## Objectives for the Next Engineer Loop
- Refresh `specs/ptychodus_api_spec.md` so it explicitly documents the torch-required policy and the configuration adapter contract now in place.  
- Add a new knowledge-base entry that captures the policy change (working ID: `POLICY-001`).  
- Ensure the updated docs cross-link: spec → finding and CLAUDE directive → finding.

## Recommended Spec Changes (`specs/ptychodus_api_spec.md`)
1. **Section 1 Overview**  
   - Insert a short paragraph after the numbered list clarifying that PyTorch `>= 2.2` is now an unconditional runtime dependency for the reconstructor stack.  
   - Mention that TensorFlow remains for legacy modules, but callers must fail fast when PyTorch is absent.  
   - Reference the governance artifact (`plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md`) and the new finding (`docs/findings.md#policy-001`).

2. **Section 2.3 The Compatibility Bridge**  
   - Add a bullet or paragraph describing the PyTorch configuration adapter (`ptycho_torch.config_bridge.to_model_config`, `to_training_config`, `to_inference_config`).  
   - Call out that these adapters MUST produce dataclasses compatible with `update_legacy_dict` and are required for parity with the TensorFlow contract.

3. **Section 4.2 Configuration Handshake**  
   - Extend the existing bullet list to make the fail-fast behavior explicit: if PyTorch cannot be imported, the reconstructor must raise an actionable error (mirrors the guard removal work in Phase F3.2).  
   - Tie the requirement back to the same finding/plan references so downstream readers see the policy linkage.

## Knowledge Base Addition (`docs/findings.md`)
- Append a table row with ID `POLICY-001`, date `2025-10-17`, keywords `policy, PyTorch, dependencies`.  
- Synopsis suggestion: “PyTorch (torch>=2.2) is now a mandatory dependency for PtychoPINN; torch-optional execution paths were removed in Phase F.”  
- Evidence pointer: link to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md`.  
- Status: `Active` (policy in force).

## Cross-Reference Checklist (F4.2.C)
- Update the new CLAUDE directive (lines 57-59) to include a `<doc-ref type="findings">docs/findings.md#policy-001</doc-ref>` tag so the directive points at the knowledge-base entry.  
- Verify the spec paragraph added in Section 1 or 2 also references the finding.  
- Capture verification steps and anchors in `spec_sync.md` (see below).

## Artifact Expectations for Engineer Loop
- `plans/active/INTEGRATE-PYTORCH-001/reports/<new-timestamp>/spec_sync.md` summarizing edits, anchors, and verification status.  
- If CLAUDE.md is touched again, note the new anchor in the same report.  
- Update `phase_f_torch_mandatory.md` (F4.2 row) and `docs/fix_plan.md` attempts history once the edits are complete.

## Open Questions for Execution
- Confirm whether additional docs (README/workflow guide) need doc-ref updates once the finding exists. If so, add them in the same loop to avoid drift.  
- Double-check that the spec text aligns with adapter function names (currently `to_model_config`, `to_training_config`, `to_inference_config`).
