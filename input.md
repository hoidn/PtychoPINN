Summary: Capture a fresh `ptycho_torch/` module inventory so we can refresh the integration plan for the rebased PyTorch stack.
Mode: Docs
Focus: INTEGRATE-PYTORCH-000 — Pre-refresh Planning for PyTorch Backend Integration
Branch: feature/torchapi
Mapped tests: none — evidence-only
Artifacts: plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/{module_inventory.md,delta_log.md}
Do Now: INTEGRATE-PYTORCH-000 — Phase A.A1 module inventory; run `find ptycho_torch -maxdepth 2 -type f \( -name '*.py' -o -name '*.ipynb' \) | sort` and capture the curated output in `module_inventory.md` alongside high-impact notes.
If Blocked: Record partial findings in `delta_log.md` under the same artifact directory, note the blocker in docs/fix_plan.md Attempts History, and stop.
Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-000/implementation.md:15 — Phase A exit criteria demand an authoritative module inventory before revising plans.
- plans/ptychodus_pytorch_integration_plan.md:1 — Legacy plan must reference the new `api/` and `datagen/` packages introduced post-rebase.
- specs/ptychodus_api_spec.md:20 — Reconciled contracts depend on understanding which PyTorch modules satisfy each requirement.
- ptycho_torch/config_params.py:1 — New dataclass definitions may shift assumptions captured in existing plans.
How-To Map:
- `mkdir -p plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z` before writing artifacts.
- Run `find ptycho_torch -maxdepth 2 -type f \( -name '*.py' -o -name '*.ipynb' \) | sort > plans/active/INTEGRATE-PYTORCH-000/reports/2025-10-17T025000Z/module_inventory.md`.
- Review the listing and append bullets describing notable additions (e.g., `api/`, `datagen/`, `reassembly_*`) directly in `module_inventory.md`.
- Start `delta_log.md` in the same folder summarizing which plan sections are impacted; leave clear TODO markers for Phase A.A2 follow-up.
Pitfalls To Avoid:
- Do not edit production code or notebooks; this loop is documentation only.
- Keep inventory output readable—no huge raw dumps without context.
- Avoid assuming nonexistent helper scripts; rely on `find` and manual annotation.
- Stay within the specified artifact directory; no files elsewhere.
- Note any uncertainties or missing coverage in `delta_log.md` so we can address them next loop.
Pointers:
- plans/active/INTEGRATE-PYTORCH-000/implementation.md:9
- plans/ptychodus_pytorch_integration_plan.md:1
- specs/ptychodus_api_spec.md:20
- docs/workflows/pytorch.md:1
Next Up: 1) Phase A.A2 delta tagging of plan coverage; 2) Phase B.B1 plan redline draft once inventory is reviewed.
