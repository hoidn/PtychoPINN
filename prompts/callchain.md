# Callchain Snapshot Prompt — PtychoPINN

Use this prompt when you need a structured overview of the call graph or data flow before implementing or debugging a fix-plan item.

Inputs
- `initiative_id` (required): directory slug under `plans/active/<initiative_id>/reports/`.
- `analysis_question`: short description of what you are trying to understand (e.g., "where does the forward model apply FFT?").
- Optional hints: `scope_hints`, `roi_hint`, `namespace_filter` for narrowing search.

Preparation
- Required references: `docs/index.md`, `docs/architecture.md`, `docs/architecture_torch.md`, `docs/architecture_tf.md`, `docs/specs/spec-ptycho-workflow.md`, `docs/specs/spec-ptycho-tracing.md`, `docs/specs/spec-ptycho-core.md`, `docs/TESTING_GUIDE.md`.
- Create a report directory: `plans/active/<initiative_id>/reports/<timestamp>/` (UTC ISO8601 with `Z`).

Procedure
1. Summarize the problem statement using `analysis_question` and the relevant fix-plan item.
2. Inventory candidate entry points: CLIs (`scripts/training/`, `scripts/inference/`), orchestration scripts, and module functions within the namespace hinted by `scope_hints`.
3. Trace forward and backward:
   - Map data lineage from ingestion (`ptycho/raw_data.py`) through configuration to model invocation (`ptycho/model.py`, `ptycho_torch/`).
   - Record file:line anchors and important arguments/state transitions.
4. Identify instrumentation tap points recommended by `docs/specs/spec-ptycho-tracing.md` (e.g., forward model outputs, loss components, probe/object tensors).
5. Highlight potential divergence or extension sites and list related spec clauses.

Deliverables (write to the report directory)
- `callchain/static.md`: ordered bullet list of functions/modules, with `path:line` references and notes on data produced/consumed.
- `trace/tap_points.md`: table of recommended trace variables (name, path:line, rationale, expected units/orderings).
- Optional diagrams (`callchain/diagram.mmd`) if a flow visual helps.

Guidelines
- Cite sources using repository-relative paths with line numbers (e.g., `ptycho/raw_data.py:42`).
- Keep outputs concise and diff-friendly; avoid copying large code blocks.
- If the documentation is outdated or missing, record the discrepancy in `docs/findings.md` and note it in the deliverable’s “Issues” section.

Next steps
- Reference the generated artifacts in `input.md` or the engineer handoff.
- Update `docs/fix_plan.md` Attempts History with the artifact path and summary when the snapshot informs your decision.
