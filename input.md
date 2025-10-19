Summary: Document PyTorch parity win and sync Phase D2 docs ledger
Mode: Docs
Focus: INTEGRATE-PYTORCH-001-STUBS D2 — Update parity summary & docs
Branch: feature/torchapi
Mapped tests: none — docs
Artifacts: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T201500Z/phase_d2_completion/{parity_update.md,summary.md}

Do Now:
1. INTEGRATE-PYTORCH-001-STUBS D2 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Author a new parity update (`parity_update.md`) under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T201500Z/phase_d2_completion/` summarising the now-green PyTorch integration run (Use Attempt #40 logs). Capture deltas vs the 2025-10-18 parity summary and note that MLflow guard is satisfied. (tests: none)
2. INTEGRATE-PYTORCH-001-STUBS D2 @ docs/workflows/pytorch.md — Refresh §§5–7 to reflect that `_reassemble_cdi_image_torch` now stitches successfully (no longer a stub), including notes on artifact paths and dtype safeguards added in D1d. (tests: none)
3. INTEGRATE-PYTORCH-001-STUBS D3 @ plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md — Mark D2/D3 rows `[x]`, add links to the 2025-10-19T111855Z + 2025-10-19T201500Z evidence, update `phase_d_workflow.md` with the new integration log reference, and record docs/fix_plan Attempt #41 documenting the documentation refresh. (tests: none)

If Blocked: Draft a `parity_blocked.md` narrative under `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T201500Z/phase_d2_completion/`, outlining why parity docs could not be updated and cite any missing evidence; do not alter plan checklists until blockers clear.

Priorities & Rationale:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md D2/D3 rows remain open despite Attempt #40 success; documentation must catch up.
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_parity_summary.md still reports PyTorch failure; needs follow-up narrative pointing to 2025-10-19 green evidence.
- docs/workflows/pytorch.md §7 currently warns `_reassemble_cdi_image_torch` is unimplemented, conflicting with the new center-crop fix.
- specs/ptychodus_api_spec.md §4.6 parity contract relies on published parity evidence; ledgers must reference the latest logs.

How-To Map:
- Parity doc: Use the 2025-10-18 parity summary as a template; emphasise the new 2025-10-19T111855Z logs (`pytest_integration_shape_green.log`, `summary.md`) and compare to prior MLflow import failure. Include bullet on decoder fix + dtype enforcement to show closure of D1d/D1e blockers.
- Workflow guide: Update sections describing `_reassemble_cdi_image_torch` (currently “raises NotImplementedError”) to outline the implemented flow (Lightning predict → decoder crop → amplitude/phase return). Mention dtype safeguards from D1d and cite `plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/summary.md`.
- Plan & ledger: In `phase_d2_completion.md`, flip D2/D3 to `[x]` with concise notes referencing the new parity update. In `phase_d_workflow.md` Phase D2 section, append a bullet pointing to the 2025-10-19T111855Z passing log. Log docs/fix_plan Attempt #41 noting doc updates + artifact paths; attach the new timestamp directory. Keep `train_debug.log` under `2025-10-19T111855Z/phase_d2_completion/`.

Pitfalls To Avoid:
- Do not revert earlier attempt history or remove historical parity summaries—add an addendum.
- Keep all new artifacts in the 2025-10-19T201500Z directory; no files at repo root.
- Avoid altering production code while editing docs.
- Ensure doc text stays consistent with specs (no promises beyond delivered behavior).
- Be explicit about selectors and timestamps when citing logs.
- Update `docs/fix_plan.md` once per Do Now completion—no partial ledger edits.
- Do not rerun the full test suite; this is a documentation loop.
- Maintain ASCII formatting; no Markdown tables without headers.
- Reference actual filenames/paths; avoid “latest log” phrasing.
- Keep summary.md concise (<1 page) and link out to parity_update.md for detail.

Pointers:
- plans/active/INTEGRATE-PYTORCH-001/phase_d2_completion.md:70
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T111855Z/phase_d2_completion/summary.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T093500Z/phase_e_parity_summary.md
- docs/workflows/pytorch.md:§5-§7
- docs/fix_plan.md:120
- plans/active/INTEGRATE-PYTORCH-001/phase_d_workflow.md:28

Next Up: 1) Convert TEST-PYTORCH-001 charter into phased plan once parity docs land; 2) Begin Phase D3 parity narrative for Ptychodus UI integration.
