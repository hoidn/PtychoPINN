Summary: Capture Phase A inventories for ADR-003 before refactoring starts
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase A inventory
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/{plan.md,summary.md,cli_inventory.md,execution_knobs.md,overlap_notes.md,logs/a1_cli_flags.txt}

Do Now:
1. ADR-003-BACKEND-API A1.a–A1.c @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/plan.md — build `cli_inventory.md` (include parity gaps vs TF CLI) and capture raw flag scan to `logs/a1_cli_flags.txt`; tests: none.
2. ADR-003-BACKEND-API A2.a–A2.c @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/plan.md — author `execution_knobs.md` cataloguing PyTorch-only runtime knobs with proposed `PyTorchExecutionConfig` placement; tests: none.
3. ADR-003-BACKEND-API A3.a–A3.c @ plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/plan.md — summarize ownership overlaps in `overlap_notes.md` and note missing ADR draft; tests: none.
4. ADR-003-BACKEND-API Phase A wrap-up — update implementation plan Phase A rows and append docs/fix_plan Attempt citing new artifacts; tests: none.

If Blocked: Capture the conflicting detail (e.g., ambiguous flag usage) in the relevant artifact, leave plan rows `[P]`, and record the blocker + evidence path in docs/fix_plan.md Attempts. Ping supervisor in summary.md.

Priorities & Rationale:
- ptycho_torch/train.py:366 — New CLI surface must be inventoried to design factories.
- ptycho_torch/inference.py:320 — Inference CLI needs parity mapping before refactor.
- ptycho_torch/config_params.py:1 — Lists current PyTorch-only knobs lacking canonical home.
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md:12 — Documents existing Lightning wiring to avoid duplication.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/plan.md — Defines the detailed checklist for this loop.

How-To Map:
- Export authoritative commands doc: `export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before recording command guidance.
- Run `rg --no-heading --line-number "add_argument" ptycho_torch/train.py ptycho_torch/inference.py | tee plans/active/ADR-003-BACKEND-API/reports/2025-10-19T225905Z/phase_a_inventory/logs/a1_cli_flags.txt`.
- Compare TensorFlow CLI parity by reviewing `scripts/training/train.py` and `scripts/inference/inference.py`; note deltas in `cli_inventory.md`.
- For execution knobs, inspect `ptycho_torch/workflows/components.py`, `config_params.py`, and `config_bridge.py`; tabulate findings with proposed `PyTorchExecutionConfig` placement in `execution_knobs.md`.
- Cross-reference upstream plans (`phase_e2_implementation.md`, `phase_e_closeout/closure_summary.md`, `plans/active/TEST-PYTORCH-001/implementation.md`) when compiling `overlap_notes.md`.
- Once artifacts are populated, update `plans/active/ADR-003-BACKEND-API/implementation.md` Phase A rows with `[x]`/`[P]` and cite the new files, then append a docs/fix_plan attempt summarizing outcomes.

Pitfalls To Avoid:
- Do not modify production code or tests during this documentation loop.
- Keep artifact filenames exactly as specified; do not move prior timestamp directories.
- Capture raw command outputs (rg, etc.) into the `logs/` subfolder for traceability.
- Cite precise file:line anchors for every mapping; avoid vague references.
- Distinguish PyTorch-only knobs from ones already handled by TensorFlow dataclasses.
- Flag ambiguities instead of guessing target configuration destinations.
- Preserve ASCII formatting and existing `<doc-ref>` tags when editing docs.
- Avoid duplicating content already covered in `phase_e2_implementation.md`; reference instead.
- If `docs/architecture/adr/ADR-003.md` is missing, log it in `overlap_notes.md` rather than creating a placeholder.
- Ensure docs/fix_plan attempt includes artifact paths and checklist IDs touched.

Pointers:
- ptycho_torch/train.py:366
- ptycho_torch/inference.py:320
- ptycho_torch/config_params.py:1
- ptycho_torch/config_bridge.py:1
- plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md:1

Next Up: 1. Draft `PyTorchExecutionConfig` schema (Phase B1) once inventories are complete.
