Summary: Align PyTorch workflow docs and findings with the execution-config spec redline.
Mode: Docs
Focus: [ADR-003-BACKEND-API] Standardize PyTorch backend API per ADR-003 — Phase E.A3 (docs alignment)
Branch: feature/torchapi
Mapped tests: none — docs-only
Artifacts: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T151734Z/phase_e_governance_workflow_docs/{doc_update_summary.md,findings_update.md}

Do Now:
1. ADR-003-BACKEND-API E.A3 (workflow guide) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:E.A3 — update `docs/workflows/pytorch.md` Section 12 training table to match the CLI defaults (accelerator `auto`, expanded choices) and add an inline pointer to spec §4.9; capture edits + rationale in `doc_update_summary.md`; tests: none.
2. ADR-003-BACKEND-API E.A3 (knowledge base) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:E.A3 — add finding `CONFIG-002` to `docs/findings.md` documenting the PyTorchExecutionConfig contract (auto accelerator default, params.cfg isolation, CLI helper enforcement) with evidence link to `spec_redline.md`; document the addition in `findings_update.md`; tests: none.
3. ADR-003-BACKEND-API E.A3 (ledger sync) @ plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:E.A3 — mark the plan row `[x]`, append a docs/fix_plan Attempt referencing the artifact directory, and sanity-check Markdown tables render correctly; tests: none.

If Blocked: Log the issue in `plans/active/ADR-003-BACKEND-API/reports/2025-10-20T151734Z/phase_e_governance_workflow_docs/blocker.md`, leave E.A3 `[P]`, and note the blocker in docs/fix_plan Attempts History before stopping.

Priorities & Rationale:
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T133500Z/phase_e_governance/plan.md:E.A3 — plan exit criteria call for doc + findings alignment ahead of Phase E.B work.
- specs/ptychodus_api_spec.md:210 — spec redline sets accelerator default `auto` and formalizes execution-config validation that docs must mirror.
- ptycho_torch/train.py:401 — authoritative parser default (`default='auto'`) for `--accelerator`.
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md — change log to cite in documentation and findings updates.
- docs/workflows/pytorch.md:317 — current training flag table still lists `'cpu'`; must be corrected for parity.

How-To Map:
- Update `docs/workflows/pytorch.md`: adjust the training execution flags table (lines ~317-320) so `--accelerator` default is `'auto'`, list choices `auto/cpu/gpu/cuda/tpu/mps`, and add a sentence linking readers to spec §4.9 for the complete PyTorchExecutionConfig field catalog; clarify that the dataclass default is `'cpu'` but CLI overrides to `'auto'`. Keep runtime metrics and helper narrative unchanged. Document before/after bullets plus file:line references in `doc_update_summary.md` under the artifact directory.
- Add knowledge-base entry: append row `CONFIG-002` to `docs/findings.md` with date `2025-10-20`, keywords (`execution-config`, `cli`, `params.cfg`), synopsis summarizing the PyTorchExecutionConfig guarantees (auto accelerator default, params.cfg isolation, CLI helper validation), and evidence link to the spec redline summary file. Capture the exact table row you added inside `findings_update.md` along with any formatting notes.
- After edits, verify Markdown tables align (pipe counts) and run `grep -n "--accelerator" docs/workflows/pytorch.md` to confirm only `'auto'` is listed as the default. Then update the plan row to `[x]` with artifact references and append a new Attempt entry to `docs/fix_plan.md` citing the doc update summary + findings update. No tests to run.

Pitfalls To Avoid:
- Do not alter runtime metrics or historical benchmarks in Section 11.
- Keep helper narratives consistent; avoid introducing new CLI scenarios beyond documented behavior.
- Maintain Markdown table formatting (same column count, pipe alignment) and avoid trailing spaces.
- Ensure the findings entry follows the existing table structure (pipes, status column) and stays sorted by ID.
- Do not downplay POLICY-001/CONFIG-001 statements elsewhere in the docs.
- Avoid promising future execution knobs beyond what Phase E.B plans cover; note backlog instead.
- Leave existing artifact paths and citations intact when adding new references.
- No code changes or tests this loop.

Pointers:
- docs/workflows/pytorch.md:317
- specs/ptychodus_api_spec.md:210
- ptycho_torch/train.py:401
- plans/active/ADR-003-BACKEND-API/reports/2025-10-20T150020Z/phase_e_governance_spec_redline/spec_redline.md
- docs/findings.md:1

Next Up: 1. Phase E.B1 — expose checkpoint & early-stop controls via CLI; 2. Phase E.B2 — scheduler and accumulation knob wiring.
