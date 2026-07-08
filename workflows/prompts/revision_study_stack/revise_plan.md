use receiving-code-review to address the feedback

Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `approved_design`, `revision_context`, `plan`, and `plan_review_report` artifacts before acting.

Revise the implementation plan in place to address every `NEW`, `STILL_OPEN`, or `SPLIT` finding whose `scope_classification` is `blocking_prerequisite` or `required_in_scope`.
Stay aligned with the approved design and do not expand the study beyond the revision scope.

Preserve task order where it reflects necessary provenance or dependency gates. Keep tranche order coherent and verification explicit. Tighten any missing file paths, verification commands, pivot criteria, manuscript update steps, and checklist/changelog requirements.

For the output contract's `plan_path`, read the path recorded in that pointer file and write the revised plan document there.
