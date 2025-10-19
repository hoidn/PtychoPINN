# Phase E3.D — TEST-PYTORCH-001 Handoff Plan

## Context
- Initiative: INTEGRATE-PYTORCH-001 — PyTorch backend integration; equip TEST-PYTORCH-001 with clear operating instructions and regression hooks now that developer-facing docs/specs are aligned.
- Phase Goal: Produce an actionable handoff packet describing backend defaults, required selectors, artifact expectations, and ongoing monitoring so the regression initiative can own PyTorch parity going forward.
- Dependencies:
  - `specs/ptychodus_api_spec.md` §4.8 — normative backend dispatch guarantees referenced in the handoff.
  - `docs/workflows/pytorch.md` §§11–12 — operational runbooks and runtime expectations.
  - `plans/active/TEST-PYTORCH-001/implementation.md` Phase D — regression hardening tasks the handoff must unblock.
  - `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` — authoritative register for Phase E completion state.
  - Findings: POLICY-001 (PyTorch mandatory), FORMAT-001 (NPZ transpose guard) — govern environment messaging.
- Artifact Discipline: Store all outputs for this loop under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T215800Z/phase_e3_docs_handoff/`. At minimum capture `handoff_brief.md`, `summary.md`, and any selectors/log excerpts.

### Phase D1 — Author TEST-PYTORCH-001 Handoff Brief
Goal: Draft a self-contained document that TEST-PYTORCH-001 can use to run, monitor, and extend PyTorch regressions without re-investigating Phase E decisions.
Prereqs: Review runtime evidence (`plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md`) and workflow guidance (§§11–12).
Exit Criteria: `handoff_brief.md` documents configuration defaults, backend flag guidance, required pytest selectors, artifact expectations, and escalation paths.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D1.A | Summarize backend selection contract | [x] | Capture literal values, CONFIG-001 requirement, and error messaging from spec §4.8. Include reminder that failing to call `update_legacy_dict` voids parity guarantees. **COMPLETE 2025-10-19:** handoff_brief.md §1.1–§1.4 documents literals (`'tensorflow'`/`'pytorch'`), CONFIG-001 requirement with code examples, fail-fast RuntimeError messaging per POLICY-001, and dispatcher routing guarantees. |
| D1.B | Enumerate regression selectors & cadence | [x] | List mandatory pytest selectors (integration workflow, backend selection suite, Torch/TensorFlow parity checks) with suggested frequency (per-commit vs nightly). Reference runtime baselines (≤90s CI budget). **COMPLETE 2025-10-19:** handoff_brief.md §2.1–§2.3 enumerates 4 selector groups (integration workflow 35.92s baseline, backend selection <5s, parity suite ~20-25 tests, model manager weekly) with cadence recommendations (per-PR/nightly/weekly) and runtime guardrails table (≤90s/60s/36s±5s/<20s thresholds). |
| D1.C | Document artifact expectations & ownership | [x] | Describe required artifacts (checkpoints, recon PNGs, logs) and where TEST-PYTORCH-001 should archive them. Provide contact/ownership matrix for handing off issues back to INTEGRATE-PYTORCH-001 if regressions appear. **COMPLETE 2025-10-19:** handoff_brief.md §3.1–§3.4 documents required artifacts (Lightning .ckpt with hyper_parameters, PNGs >1KB, debug logs), checkpoint validation command (torch.load + assert), archival policy (transient vs persistent), 5-row ownership matrix (test harness/backend impl/config bridge/dispatcher/checkpoint persistence), and 5-step escalation workflow with command examples. |

### Phase D2 — Update Plan & Ledger References
Goal: Ensure initiative plans and ledger entries reflect the handoff deliverables once D1 is complete.
Prereqs: D1 handoff brief finalized.
Exit Criteria: `phase_e_integration.md` and `docs/fix_plan.md` reference the handoff artifacts; plan checklist rows marked `[x]` with guidance preserved.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D2.A | Mark phase_e_integration.md entries complete | [ ] | Update E3 rows to `[x]`, summarizing documentation/spec alignment and linking to `handoff_brief.md`. Ensure instructions remain for future auditors. |
| D2.B | Append ledger attempt summary | [ ] | Record a new Attempt in `docs/fix_plan.md` detailing the handoff, artifact paths, and next-step recommendations. |

### Phase D3 — Define Follow-Up Checks & Alerts
Goal: Provide TEST-PYTORCH-001 with explicit monitoring hooks and schedule for parity verification to avoid regressions slipping through CI.
Prereqs: D1 brief complete (so schedule references correct selectors).
Exit Criteria: Handoff brief (or companion note) lists follow-up checks with owners and triggers.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| D3.A | Specify monitoring cadence | [ ] | Recommend cadence for running integration workflow tests (e.g., per PR vs nightly) and note environment nuances (CPU only, CUDA opt-in). |
| D3.B | Capture escalation triggers | [ ] | Document what failures require escalation back to INTEGRATE-PYTORCH-001 (e.g., runtime >90s, checkpoint load failures, POLICY-001 violations) and how to log them. |

## Reference Documents Consulted
- `specs/ptychodus_api_spec.md` §4.8 — backend dispatch contract wording.
- `docs/workflows/pytorch.md` §§11–12 — regression runtime expectations and backend selection walkthrough.
- `plans/active/TEST-PYTORCH-001/implementation.md` Phase D — clarifies regression hardening scope to coordinate with handoff.
- `plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md` — ensures checklist parity and dependencies.
- `plans/active/TEST-PYTORCH-001/reports/2025-10-19T193425Z/phase_d_hardening/runtime_profile.md` — runtime guardrails for selectors.
