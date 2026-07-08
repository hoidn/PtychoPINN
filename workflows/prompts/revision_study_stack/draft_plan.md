Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `approved_design` and `revision_context` artifacts before acting.

Draft an execution plan for the revision study. The plan should be concrete enough for an implementation agent working in the current checkout.

The plan should:
- break the work into coherent tranches
- put prerequisites before dependent work
- include verification for each tranche
- call out compatibility boundaries, explicit non-goals, and any manuscript-claim limits
- include ordered implementation tasks with exact file paths where known
- data/provenance discovery tasks before any metric or claim update that depends on them
- bounded decision gates and pivot criteria from the approved design
- expected generated artifacts, including metrics, manifests, figures, tables, manuscript edits, changelog, and checklist updates
- runnable verification commands and manual PDF/manuscript inspection steps
- explicit stop conditions when evidence would make the reviewer response scientifically unsafe

Do not require editing the original revision design seed unless the approved design explicitly says to update it.

For the output contract's `plan_path`, read the path recorded in that pointer file and write the plan document there.
