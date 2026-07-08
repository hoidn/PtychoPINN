Use the injected redesign brief as the authoritative scope and path contract.

Task:
- Design one coherent `lines_256` redesign candidate that is broad enough to justify the redesign factory, but still narrow enough to score as a single candidate.

Required output:
- Write the design note to the `design_doc_path` named in the redesign brief.

The design note must include:
- the redesign hypothesis
- why the direct proposal factory is a poor fit for this idea
- the exact model or workflow surfaces likely to change
- expected risks and likely failure modes
- the minimal targeted checks the implementation phase should run

Constraints:
- Do not write `candidate_metadata.json` in this step.
- Do not write `proposal_result.json` in this step.
- Do not move queue items or edit session/accepted-state files.
- Keep the design candidate-local; do not redesign the outer controller loop.
