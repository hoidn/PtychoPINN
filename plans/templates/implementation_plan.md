# Implementation Plan Template

> Copy to `docs/plans/YYYY-MM-DD-<initiative>.md` and replace every placeholder.
> A folder such as `docs/plans/<initiative>/` is optional and should be used
> only when genuinely multi-document work needs a shared durable home.

Use this template proportionately. Delete sections and prompts that do not apply,
but keep enough detail that another engineer can execute the plan without
inventing requirements, architecture, commands, or expected results.

An implementation plan sequences accepted work. It may record a small local
approach, but it must not silently decide unresolved behavior, architecture,
public or scientific contracts, migration policy, or material alternatives. If
those decisions are still open, stop and use the canonical design template at
`plans/templates/design_template.md` first; link the accepted design here
afterward.

## Plan summary

- **Title:** <initiative title>
- **Status:** draft | approved | in_progress | blocked | complete
- **Goal:** <one observable outcome>
- **Architecture / approach:** <how the change fits the existing system; link an
  accepted design when one owns the decision>
- **Authority:** <governing specs, accepted design, user acceptance criteria,
  and repository policy, ordered by authority>
- **Scope:** <included behavior and files or subsystems>
- **Out of scope:** <nearby work deliberately excluded>
- **Assumptions to verify:** <only execution assumptions; unresolved design
  choices belong in a design>
- **Invariants:** <behavior, interfaces, compatibility, safety, or claim
  boundaries that must remain true>

### Scientific, data, and provenance constraints (only when relevant)

- **Scientific claim / metric boundary:** <what the evidence may and may not
  support>
- **Data identity and split:** <dataset/version/checksum, split, normalization>
- **Reproducibility / provenance:** <seed, environment, config, checkpoint, or
  run lineage that the acceptance claim requires>

## Execution tasks

Create one task per bounded, reviewable behavior or migration slice. Name every
file exactly; do not use placeholders such as “relevant files” in a completed
plan. Order tasks by dependency.

### Task 1: <behavioral outcome>

**Files**

- Create: `<exact/path>`
- Modify: `<exact/path>`
- Test: `<exact/test/path>`

**Contract and constraints**

- <observable behavior this task adds or changes>
- <authority clause or invariant this task must preserve>

**Steps**

1. <small implementation step>
2. <small implementation step>

**Behavioral evidence**

For a feature or bug fix, record a RED/GREEN cycle that can distinguish the
missing behavior from an environment or collection failure. For documentation,
data-only, mechanical, or already-covered changes, use the smallest direct
check instead of manufacturing a RED phase.

- **RED:** `<exact command>`
  - Expected: <test collects and fails for the missing or incorrect behavior;
    name the assertion or signal>
- **GREEN:** `<exact command>`
  - Expected: <exact pass count or observable result, with no unexpected skips
    or errors>
- **Task check (when RED/GREEN does not apply):** `<exact command>`
  - Expected: <specific success signal>

### Task 2: <next behavioral outcome>

Repeat the Task 1 structure with exact paths, ordered steps, and claim-matched
evidence. Add tasks only when they create independently reviewable progress.

## Review and commit checkpoints

Place checkpoints after coherent slices, not automatically after every file.
Each checkpoint must be bounded to named tasks and files.

### Checkpoint 1: <tasks or slice>

- Review: `<exact git diff command limited to the named files>`
- Confirm: <contract, edge cases, adjacent-system compatibility, and absence of
  unrelated changes>
- Re-run: `<exact narrow verification command>`
- Expected: <review condition and command result>
- Commit: <optional exact commit command and message, only when committing is
  authorized; otherwise state “no commit”>

## Verification ladder

List fresh commands in cheapest-to-broadest order. Every completed plan must
replace generic examples with commands that exist in this repository and state
the expected result. A broader tier is required only when the acceptance claim
or affected contract needs it.

1. **Static / collection:** `<exact lint, type, parse, link, or
   pytest --collect-only command>`
   - Expected: <specific result; new or renamed test modules collect expected
     tests>
2. **Focused behavior:** `<exact test selector or executable probe>`
   - Expected: <pass count or observable output; identify acceptable skips>
3. **Affected integration:** `<exact integration or smoke command>`
   - Expected: <end-to-end behavior and artifact/schema checks>
4. **Broad regression (when justified):** `<exact suite command>`
   - Expected: <pass/fail/skip contract and why this breadth is needed>
5. **Scientific or performance validation (when claimed):** `<exact command>`
   - Expected: <metric, tolerance, comparison, runtime, or provenance fields>

## Documentation impact

- Update: `<exact documentation/index/spec path>` — <what must change and why>
- No documentation change: <explain why user-facing, contract, and routing docs
  remain accurate>

Do not use a documentation update to redefine a governing contract implicitly;
route contract changes through the owning spec or accepted design.

## Completion criteria

- [ ] <observable acceptance criterion tied to the goal>
- [ ] <affected invariants and compatibility boundaries are preserved>
- [ ] <required verification ladder tiers have fresh successful evidence>
- [ ] <documentation and discoverability are accurate, or no-impact rationale
  is recorded>
- [ ] <diff contains only authorized, task-scoped changes>

## Stop conditions

Stop execution and report the condition instead of improvising when:

- a required authority or accepted design is missing or contradictory;
- an assumption fails in a way that changes scope, architecture, behavior, or
  scientific claims;
- a required dependency, dataset, environment capability, or credential is
  unavailable and no authorized fallback preserves the contract;
- verification exposes an unrelated failure that cannot be safely classified;
- the next action would modify files, external systems, or claims outside the
  authorized scope.

Record the failing command or observation, affected task, work completed, and
the concrete condition for resuming.
