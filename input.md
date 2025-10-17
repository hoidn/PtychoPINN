Summary: Capture the full torch-optional surface area and author the Phase F2 migration blueprint.
Mode: Docs
Focus: INTEGRATE-PYTORCH-001 / Phase F2 Impact Inventory & Migration Blueprint
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T192500Z/{torch_optional_inventory.md,test_skip_audit.md,migration_plan.md}
Do Now:
- INTEGRATE-PYTORCH-001 Phase F2 — F2.1 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: none): scan the repo for torch-optional guards and record a file:line inventory in `torch_optional_inventory.md`.
- INTEGRATE-PYTORCH-001 Phase F2 — F2.2 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: none): analyze pytest skip/whitelist behavior and summarize required test changes in `test_skip_audit.md`.
- INTEGRATE-PYTORCH-001 Phase F2 — F2.3 @ plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md (tests: none): draft `migration_plan.md` sequencing implementation, gating checks, and risk mitigations informed by F2.1/F2.2.
If Blocked: Document blockers and partial findings in `migration_plan.md`, flag the unresolved items in docs/fix_plan.md Attempts History, and halt before altering any production code.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:24-33 — Phase F2 checklist defines deliverables and acceptable evidence.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/directive_conflict.md — Governance summary clarifies why every guard must be inventoried before removal.
- tests/conftest.py:24-46 — Current TORCH_OPTIONAL whitelist drives skip behavior; we need file-level impacts before rewriting it.
- docs/fix_plan.md:138-140 — Attempt #65 closure plus Attempt #66 prep commit this loop to the ledger; inventory must bridge to upcoming F3 execution.
How-To Map:
- Guard inventory: `rg -n "TORCH_AVAILABLE" -g"*.py" -g"*.pyi"` and `rg -n "torch_available" tests` to capture both constants and fixtures; include modules like `ptycho_torch/config_bridge.py`, `ptycho_torch/data_container_bridge.py`, `ptycho_torch/memmap_bridge.py`, `ptycho_torch/workflows`, and CLI adapters. Cross-check with `rg -n "import torch" ptycho_torch tests/scripts` to find conditional imports.
- Skip audit: read `tests/conftest.py` and note whitelist semantics; list affected test modules (torch/ subpackages, backend selection, workflows) and distinguish skip vs. xfail vs. unconditional passes. Capture expected behavior changes once torch becomes mandatory.
- Migration blueprint: outline Phase F3 gating (dependency promotion, guard removal, pytest updates) and Phase F4 follow-through (docs/spec sync). Reference F1 governance risks and note required CI/environment prerequisites. Store each deliverable in the artifact directory listed above.
- Keep each artifact scoped: inventory = tabular matrix with package/module/guard type; skip audit = narrative plus checklist; migration plan = phased steps with decision gates and owners.
Pitfalls To Avoid:
- Do not modify code or tests—evidence only.
- Avoid double-counting guards; group related lines but keep explicit anchors.
- Note dynamic imports separately; do not assume `TORCH_AVAILABLE` is the only guard pattern.
- Keep skip audit torch-neutral; no assumptions about torch being installed during execution.
- Record open questions explicitly; do not leave TBDs without owner or follow-up phase.
- Ensure artifact filenames exactly match the header; no extra suffixes.
- Maintain parity with CLAUDE.md directives until Phase F3 implementation is authorized.
Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md:24-33
- tests/conftest.py:24-46
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T184624Z/governance_decision.md
- docs/fix_plan.md:138-141
Next Up: Phase F3 dependency + guard removal (F3.1–F3.3) once inventory and migration plan are accepted.
