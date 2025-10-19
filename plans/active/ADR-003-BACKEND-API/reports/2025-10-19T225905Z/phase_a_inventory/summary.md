# Phase A Planning Summary — ADR-003 Backend API (2025-10-19T225905Z)

## Loop Overview
- **Supervisor:** galph
- **Focus Issue:** `[ADR-003-BACKEND-API]` Phase A — Architecture Carve-Out
- **Action Type:** Planning (Mode: Docs)
- **Objective:** Provide engineering-ready guidance to execute Phase A tasks (CLI inventory, backend knob catalog, cross-plan overlap audit).

## Key Decisions
- Established dedicated artifact hub `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/` with required deliverables (`cli_inventory.md`, `execution_knobs.md`, `overlap_notes.md`, `summary.md`).
- Authored structured Phase A execution plan dividing work into A1/A2/A3 subsections with explicit task IDs (A1.a–A3.c), commands (`rg` selectors), and exit criteria tied to specs and existing PyTorch workflow documentation.
- Highlighted critical source files (`ptycho_torch/train.py`, `ptycho_torch/inference.py`, `ptycho_torch/workflows/components.py`, `config_params.py`, `config_bridge.py`) and upstream plans (`phase_e2_implementation.md`, TEST-PYTORCH-001 implementation) that must be referenced during evidence capture.
- Flagged missing ADR draft (`docs/architecture/adr/ADR-003.md`) as an item to confirm during the overlap audit so governance gaps are tracked early.

## Next Steps for Engineering Loop
1. Execute A1 tasks to capture CLI flag inventory, including parity check against TensorFlow scripts, and populate `cli_inventory.md`.
2. Perform A2 backend knob cataloging, leveraging config singletons and workflow components, outputting `execution_knobs.md`.
3. Complete A3 overlap audit with `overlap_notes.md`, confirming ownership boundaries and noting missing ADR documentation.
4. Update `plans/active/ADR-003-BACKEND-API/implementation.md` Phase A rows and append a new `docs/fix_plan.md` attempt with artifact links.

## Open Questions / Risks
- Need confirmation whether the ADR-003 architecture document exists elsewhere or must be authored during Phase B; record outcome in `overlap_notes.md`.
- Ensure CLI parity gaps (if any) are translated into follow-up tasks before Phase B begins (e.g., missing TensorFlow flags such as `--sequential_sampling`).

## Artifacts Created
- `plan.md` — Phase A execution blueprint with task tables and verification checklist.
- `summary.md` — (this file) capturing planning highlights and next-step expectations.

All artifacts stored under `plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/`.
