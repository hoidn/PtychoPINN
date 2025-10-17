# INTEGRATE-PYTORCH-000 — Rebaseline PyTorch Integration Plans

## Context
- Initiative: INTEGRATE-PYTORCH-000
- Phase Goal: Refresh the planning corpus (integration plan, parity notes, fix plan ledger) so it accurately reflects the rebased `ptycho_torch/` tree landing in commit bfc22e7.
- Dependencies: specs/ptychodus_api_spec.md (contract to reconcile), plans/ptychodus_pytorch_integration_plan.md (legacy plan needing overhaul), plans/active/INTEGRATE-PYTORCH-001/implementation.md (downstream execution plan), docs/workflows/pytorch.md (workflow reference), docs/findings.md (guardrails).
- Artifact storage: create evidence folders beneath `plans/active/INTEGRATE-PYTORCH-000/reports/<ISO8601Z>/` for inventories, notes, and diffs. Reference each artifact from docs/fix_plan.md Attempts History.

---

### Phase A — Source Audit & Gap Identification
Goal: Capture an authoritative snapshot of the latest `ptycho_torch/` modules and surface planning-impacting deltas vs. the legacy plan.
Prereqs: None.
Exit Criteria: `reports/<timestamp>/module_inventory.md` documenting tree structure + high-impact modules, plus `reports/<timestamp>/delta_log.md` summarizing divergences against the previous plan.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| A1 | Generate module inventory for `ptycho_torch/` | [ ] | Run `python scripts/tools/list_repo_tree.py ptycho_torch > reports/<ts>/module_inventory.md` (or fallback `find ptycho_torch -maxdepth 2`). Highlight new `api/`, `datagen/`, `reassembly_*` packages introduced in bfc22e7. |
| A2 | Compare inventory against legacy plan scope | [ ] | Annotate `module_inventory.md` with "covered / missing" tags referencing sections in `plans/ptychodus_pytorch_integration_plan.md`; capture summary bullets in `delta_log.md`. |
| A3 | Flag spec touchpoints requiring rework | [ ] | While reviewing, note any modules that alter assumptions in `specs/ptychodus_api_spec.md` (e.g., new dataclass schema in `config_params.py`). Record them in `delta_log.md` with file:line citations. |

---

### Phase B — Plan Document Refactor
Goal: Update `plans/ptychodus_pytorch_integration_plan.md` so it aligns with the audited module set and hands off cleanly to INTEGRATE-PYTORCH-001.
Prereqs: Phase A inventory + delta log.
Exit Criteria: Revised integration plan checked into repo, annotated with fresh module references and phase mapping to current initiatives.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| B1 | Draft revised outline incorporating new subsystems | [ ] | Use `delta_log.md` to insert/update sections for `api/`, `datagen/`, Lightning persistence, etc. Capture draft in `reports/<ts>/plan_redline.md` before editing the canonical plan. |
| B2 | Update canonical integration plan | [ ] | Edit `plans/ptychodus_pytorch_integration_plan.md` with phased structure mirroring current initiatives (Phase 0→E). Cross-link new sections to module inventory + spec hits. |
| B3 | Peer-ready review notes | [ ] | Summarize key revisions and open questions in `reports/<ts>/summary.md` to brief downstream initiatives (INTEGRATE-PYTORCH-001, TEST-PYTORCH-001). |

---

### Phase C — Governance & Handoff Sync
Goal: Ensure the refreshed plan propagates to execution artifacts and the fix ledger, preventing drift.
Prereqs: Phase B complete.
Exit Criteria: docs/fix_plan.md updated with latest attempts, pointer to refreshed plan, and action state communicated to INTEGRATE-PYTORCH-001 owner.

| ID | Task Description | State | How/Why & Guidance |
| --- | --- | --- | --- |
| C1 | Update fix plan entry & attempts history | [ ] | Note completion of Phase B with artifact paths; keep status accurate. |
| C2 | Notify INTEGRATE-PYTORCH-001 plan owners | [ ] | Append note to `plans/active/INTEGRATE-PYTORCH-001/implementation.md` (if needed) or galph_memory referencing new plan sections to consume. |
| C3 | Confirm input.md alignment | [ ] | Ensure supervisor directives for subsequent loops reference the refreshed plan before moving focus back to execution tasks. |

---

### Risk Log
- Massive notebook additions may bloat inventories; keep reports lean by focusing on `.py` and critical assets.
- Lightning/MLflow defaults in `train.py` could shift recommended TDD strategy—flag during Phase A if plan needs new mitigation guidance.
- Keep coordination tight with TEST-PYTORCH-001 to avoid conflicting fixture expectations during doc refresh.

### Verification & Artifacts
- Preferred commands documented in `reports/<ts>/module_inventory.md`.
- All reports saved under `plans/active/INTEGRATE-PYTORCH-000/reports/` with ISO timestamps (e.g., `.../2025-10-17T030000Z/{module_inventory.md,delta_log.md}`).
- Treat notebooks as informational; if referenced, include path + purpose but avoid copying large content.

### Completion Definition
- Phase A inventory + delta analysis complete and shared.
- Integration plan doc updated and merged.
- Fix ledger + execution plans synchronized with refreshed documentation.
- Initiative marked ready to hand off to INTEGRATE-PYTORCH-001 (Phase B tasks) and TEST-PYTORCH-001 (fixture alignment).
