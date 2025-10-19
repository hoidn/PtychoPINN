# Phase E3.B2/B3 Guidance — Dual-Backend Messaging

**Initiative:** INTEGRATE-PYTORCH-001
**Context:** Phase E3 (Documentation, Spec Sync, Handoff)
**Scope:** Tasks B2 (CLAUDE.md + README.md updates) and B3 (documentation cross-link verification)
**Prepared by:** Supervisor (galph)
**Date:** 2025-10-19

## Objectives
- Surface backend selection workflow to onboarding docs so agents and contributors understand how to target the PyTorch backend.
- Reinforce CONFIG-001 requirements for PyTorch runs inside CLAUDE.md so debugging instructions stay correct.
- Update README feature list to advertise dual-backend architecture and direct readers to the authoritative workflow guide (§12) and runtime evidence.
- Verify no stray "NotImplementedError" warnings remain in architecture/workflow docs after Phase E3.B1 edits.

## References
- `specs/ptychodus_api_spec.md:224-235` — §4.8 Backend Selection & Dispatch (normative requirements)
- `docs/workflows/pytorch.md:297-404` — §12 Backend Selection in Ptychodus Integration (new reference section)
- `docs/architecture.md:13` — Backend selector caption (Phase E3.B1 addition)
- `docs/findings.md#policy-001` — PyTorch mandatory policy
- `plans/active/INTEGRATE-PYTORCH-001/phase_e3_docs_plan.md` — Phase checklist and completion criteria

## Task Breakdown

### B2.1 — Update CLAUDE.md (Agent Guidance)
- Add a short paragraph under **4.1 Parameter Initialization** or immediately after the PyTorch requirement directive reminding agents that PyTorch workflows also depend on `update_legacy_dict` before data loading.
- Explicitly reference `docs/workflows/pytorch.md` §3 (Configuration Setup) and spec §4.8 routing guarantees to keep instructions authoritative.
- Mention that backend selection uses `TrainingConfig.backend` / `InferenceConfig.backend` with `'pytorch'` literal when delegating through Ptychodus.
- Highlight artifact evidence location for runtime parity: `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`.

### B2.2 — Update README.md (Onboarding)
- Insert a new subsection (suggested heading: `### Dual-Backend Architecture`) directly under `## Features`.
- Summarize:
  1. Default backend remains TensorFlow for backward compatibility.
  2. PyTorch backend now production-ready via Lightning (`ptycho_torch/workflows/components.py`).
  3. Point to `docs/workflows/pytorch.md` §12 for configuration steps.
  4. Call out runtime evidence (Phase D1 runtime profile) and integration test (TEST-PYTORCH-001).
- Retain ASCII formatting; avoid adding external links beyond existing docs.

### B3 — Cross-Link & Warning Verification
- After edits, run the following command to ensure no stale stub warnings remain:
  ```bash
  rg "NotImplementedError" docs/workflows/pytorch.md docs/architecture.md
  ```
- Capture command output (even if empty) to `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T213900Z/phase_e3_docs_b2/rg_notimplemented.log` using `tee`.
- Record verification notes (pass/fail + file anchors touched) in `summary.md` for this loop.

## Artifact Expectations
- Updated plan checklist rows referencing this guidance document.
- `summary.md` capturing execution notes and verification results.
- Command output log described above.

## Exit Criteria Reminders
- CLAUDE.md includes explicit PyTorch backend selection + CONFIG-001 reminder with references to §4.8 and workflow doc.
- README.md advertises dual-backend architecture and directs readers to workflow spec + runtime evidence.
- `rg` command executes cleanly; output stored in artifact directory; findings summarized.
