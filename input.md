Summary: Capture Phase A inventories for ADR-003 backend API standardization
Mode: Docs
Focus: ADR-003-BACKEND-API — Phase A Architecture Carve-Out
Branch: feature/torchapi
Mapped tests: none — documentation focus
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-17T224600Z/{cli_inventory.md,execution_knobs.md,phase_a_summary.md}

Do Now:
1. ADR-003-BACKEND-API A1 @ plans/active/ADR-003-BACKEND-API/implementation.md — Document CLI → config mappings for `ptycho_torch/train.py` & `ptycho_torch/inference.py` in cli_inventory.md (tests: none)
2. ADR-003-BACKEND-API A2 @ plans/active/ADR-003-BACKEND-API/implementation.md — Catalogue PyTorch execution knobs (Lightning, MLflow, device/strategy) in execution_knobs.md (tests: none)
3. ADR-003-BACKEND-API A3 @ plans/active/ADR-003-BACKEND-API/implementation.md — Summarize dependencies/overlaps with INTEGRATE-PYTORCH-001 in phase_a_summary.md and update plan checkboxes + docs/fix_plan attempts (tests: none)

If Blocked: Capture partial findings in the same artifacts, note blockers in docs/fix_plan Attempts history, leave checklist entries unmet, and alert supervisor next loop.

Priorities & Rationale:
- docs/fix_plan.md#L37 — New `[ADR-003-BACKEND-API]` item requires Phase A inventory before implementation begins.
- plans/active/ADR-003-BACKEND-API/implementation.md — Phase A tasks unblock factory/workflow refactors by defining required inputs.
- specs/ptychodus_api_spec.md §4 — Canonical dataclass contract must remain authoritative; inventories prevent field omissions.
- docs/workflows/pytorch.md §§2–5 — Documents existing CLI usage patterns that the new API must preserve.
- ptycho_torch/train.py & ptycho_torch/inference.py — Current procedural entry points to be refactored; need thorough parameter capture first.

How-To Map:
- export timestamp=2025-10-17T224600Z; mkdir -p plans/active/ADR-003-BACKEND-API/reports/$timestamp
- Use source references (`ptycho_torch/train.py`, `ptycho_torch/inference.py`, `ptycho_torch/train_utils.py`, `config_params.py`) to fill cli_inventory.md tables (flag name → PyTorch field → canonical field/override)
- For execution_knobs.md, document Lightning Trainer keyword usage, MLflow toggles, device/strategy selection logic, and any implicit defaults noted in code
- In phase_a_summary.md, highlight cross-plan dependencies (INTEGRATE-PYTORCH-001 Phases C/D), unresolved questions, and mark plan checklist A1–A3 states; reflect same in implementation.md and docs/fix_plan.md attempts
- No tests to run; ensure artefacts remain within the timestamped reports directory

Pitfalls To Avoid:
- Do not modify production code during this documentation loop
- Keep inventories explicit (include source file & line anchors for each mapping)
- Avoid duplicating data already captured; link back to prior reports when reusing context
- Maintain consistent terminology with ADR-003 (canonical config vs execution config)
- Store only markdown artefacts in the reports directory; no stray logs
- Update plan checkboxes only after verifying artefacts exist
- Record any open questions in phase_a_summary.md for supervisor review
- Respect existing findings (CONFIG-001, POLICY-001) when noting dependencies
- Coordinate with INTEGRATE-PYTORCH-001 plan to prevent conflicting directives
- Leave docs/fix_plan exit criteria untouched until implementation phases

Pointers:
- docs/fix_plan.md#L37
- plans/active/ADR-003-BACKEND-API/implementation.md
- ptycho_torch/train.py
- ptycho_torch/inference.py
- ptycho_torch/train_utils.py
- specs/ptychodus_api_spec.md#L1
- docs/workflows/pytorch.md#L1

Next Up: Phase B factory design once Phase A inventory artefacts are reviewed.
