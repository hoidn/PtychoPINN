Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `revision_design_seed` and `revision_context` artifacts before acting.

Create a revision-study design document from the seed design. Treat the seed design as immutable provenance: do not edit it in place.

This is a design-only step. Do not implement the study, edit manuscript/checklist/source files, or run expensive experiments. Only write the design document to the output path.

The design must decide the right implementation shape:
- use an existing script or workflow when that is sufficient
- propose a small one-off script only when the work is genuinely isolated
- include an architectural design section when the study needs reusable machinery, shared analysis utilities, data-contract changes, or changes to existing workflow APIs

The design should be specific enough to support implementation planning. Include:
- reviewer issue and manuscript scope
- proposed implementation shape and rationale
- source data, scripts, figures, tables, and manuscript files likely to be touched
- dependency, data-contract, and provenance decisions needed before execution
- pivot criteria for narrowing claims or switching to text-only response if the study cannot be made clean
- required final assets, including manuscript, changelog, checklist, metrics, figure, table, and manifest updates where relevant
- verification commands and inspection checks

For the output contract's `approved_design_path`, read the path recorded in that pointer file and write exactly one revision-study design document there.
