Read the `Consumed Artifacts` section first and treat it as the authoritative input list.
Read the consumed `design`, `roadmap`, and `tranche_context` artifacts before acting.

Draft an execution plan for the selected tranche from the approved design, roadmap, and tranche context.

The roadmap is binding planning context, not background reading. The plan must preserve the roadmap's phase order, gates, artifact expectations, and fallback decisions unless the design explicitly supersedes them.

The tranche context is the scope boundary for this plan. Include only the selected tranche's executable work plus the prerequisite checks and carry-forward context needed to do it safely. Do not silently expand the plan to later roadmap phases.

If the repo has a local implementation-plan template or planning guide under `docs/templates/`, use it for document structure unless it conflicts with the consumed design, consumed roadmap, or output contract. Omit irrelevant optional sections rather than padding the plan.

Before writing the task checklist, decide whether the work needs an Implementation Architecture section.

Include an Implementation Architecture section when the implementation has more than one meaningful responsibility, when future work will depend on the result, when behavior crosses a boundary between modules, tools, artifacts, or processes, or when a poor file or component split would make the work harder to verify, review, reuse, or change safely.

This section is plan-level architecture: translate the approved design and roadmap into implementable boundaries without changing their decisions. Include:
- proposed implementation units and the responsibility of each
- stable interfaces, data structures, commands, artifacts, or data-flow boundaries each unit owns
- what each unit must not own, especially behavior that belongs elsewhere
- dependency direction between units
- compatibility, migration, and backward-compatibility boundaries that must remain pinned
- focused tests for each unit or boundary
- sequencing constraints that prevent broad shared files or broad test files from absorbing unrelated responsibilities

If the work is small enough for a single implementation unit, state that briefly and justify why a single unit and focused test locus are sufficient.

If the design or roadmap lacks a material architectural decision needed to plan safely, call out the missing decision explicitly instead of inventing conflicting architecture.

The plan should:
- break the work into coherent tranches
- put prerequisites before dependent work
- preserve the roadmap's phase order unless a documented gate requires a pivot
- include verification for each tranche
- organize implementation tranches around the Implementation Architecture boundaries when that section is present
- call out migrations, compatibility boundaries, and explicit non-goals
- avoid vague shared-work tasks such as "implement the validator" or "update the helper"; split nontrivial shared work by owned interface, data flow, validation, IO, command or reporting surface, and tests as appropriate
- include discoverability or documentation update steps when the work changes behavioral specs, public or internal APIs, architectural conventions, development processes, test conventions, data or oracle contracts, creates important docs, or changes other durable project knowledge; when qualifying docs are created or materially changed, include a task for updating the relevant documentation index such as `docs/index.md` when present; avoid documentation churn for purely local implementation details
- when the design or roadmap identifies generated artifacts, helper scripts, validators, curated data, benchmark scorecards, or paper-facing evidence maps, plan the concrete file targets, generation commands, validation checks, and tests needed to make the work executable

For the output contract's `plan_path`, read the path recorded in that file and write the plan document to that current-checkout-relative path. Leave the `plan_path` file containing only the path.
