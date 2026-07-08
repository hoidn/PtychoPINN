Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `revision_design_seed`, `revision_context`, `approved_design`, and `design_review_report` artifacts before acting.

Revise the revision-study design to address the unresolved in-scope findings. Treat the seed design as immutable provenance and do not edit it in place.

This is a design-only step. Do not implement the study, edit manuscript/checklist/source files, or run expensive experiments. Only write the revised design document to the output path.

Preserve useful accepted content from the current design. Tighten the implementation-shape rationale, architecture if needed, scientific policy, dependency/provenance decisions, pivot criteria, required assets, and verification where the review found gaps.

For the output contract's `approved_design_path`, read the path recorded in that pointer file and write exactly one revised revision-study design document there.
