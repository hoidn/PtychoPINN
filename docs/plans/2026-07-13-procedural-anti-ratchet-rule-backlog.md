# Procedural Anti-Ratchet Rule Backlog

**Date:** 2026-07-13  
**Status:** Pending after the Stage 1 authority-and-verification patch  
**Scope:** Project and reusable-agent process rules; no PtychoPINN product behavior changes

## Problem

A controller-selected tactic can acquire a name, appear in a report or progress ledger, be repeated by a later task, and then be treated as a governing requirement without any authoritative adoption. The concrete failure was:

```text
three files authorized for Slice A
-> controller runs all three modules
-> report calls them "owned modules"
-> later slice inherits a whole-module gate
```

The opposite failure is also possible: an over-broad anti-process rule can let an agent shrink a claim until a narrow passing check appears adequate, dismiss a real regression as "supplemental," or omit necessary consumer work because it was not pre-enumerated.

Stage 1 addresses the authority and evidence core in `AGENTS.md`, `verification-before-completion`, and `executing-plans`. The remaining changes below are intentionally staged so each can be behaviorally evaluated before adoption.

## Governing Invariants

1. Current authority creates requirements; execution evidence does not.
2. Authorized paths grant mutation permission, not test ownership or deliverables.
3. A controller-added tactic is local, bounded, and expiring unless higher authority explicitly adopts it.
4. Current authority defines the complete acceptance unit before evidence is selected.
5. Evidence must be sufficient for the claim and capable of falsifying it.
6. Supplemental observations can prove a current contract violation, but cannot create a new requirement by their mere existence or failure.
7. Necessary emergent work remains admissible through a direct causal link to a current requirement, affected contract, safety invariant, or deliverable.
8. Durable recovery state preserves facts and approved decisions, not free-text requirements or historical process conventions.

## Stage 2: Skill Selection And Expiry

### Target

Revise `using-superpowers` after pressure tests prove the replacement avoids both indiscriminate skill stacking and rationalized omission of mandatory preventive skills.

### Changes

- Replace the probabilistic “1% chance” invocation rule with deterministic triggers for mandatory skills and named-risk selection for optional skills.
- Invoke a skill before its first governed action when one of these is true:
  - current authority explicitly requires it;
  - an observable condition satisfies the skill's declared mandatory trigger; or
  - omitting it would materially increase a named current-contract or safety risk that the skill specifically controls.
- Do not require a second subjective risk showing after a deterministic trigger matches.
- Select the narrowest complete workflow set for the distinct active risks. Do not stack skills merely because descriptions overlap.
- Re-evaluate selection when task phase, failure mode, delegation boundary, scope, irreversible operation, safety-relevant discovery, or completion claim changes.
- State that a selected skill controls method and safety stops only. It cannot add product scope, acceptance criteria, durable artifacts, unrelated workflows, or future-task obligations.
- Make skill-local tactics expire when the matched risk or phase ends.
- Remove the current wording that treats nearly any conceivable applicability as mandatory.

### Behavioral acceptance scenarios

- A reproducible bug still activates systematic debugging, TDD for the behavior change, and verification before completion.
- A prose-only review does not activate implementation workflows.
- An unexpected test failure during planned work triggers re-evaluation and debugging rather than blind continuation.
- A task split into small subtasks does not evade a mandatory trigger.
- An explicitly required skill is not omitted because an agent calls the task “small.”
- Distinct risks may compose skills; broad topical overlap alone does not.

## Stage 3: Review And Recovery State

### Targets

- `subagent-driven-development`
- `requesting-code-review`
- reusable durable-progress guidance and any controller ledger schema that currently stores semantic task authority

### Changes

- Replace unconditional per-task review and final-review rituals with review-admission predicates:
  - current user, specification, or approved plan requires review;
  - the change alters a public or cross-task API, schema, configuration, CLI, wire/event/IPC payload, or persisted format with consumers outside the task;
  - the change affects authorization, security, privacy, safety, provenance, data integrity, destructive or irreversible effects, concurrency, nondeterminism, migration, compatibility, or another invariant not fully falsifiable by a deterministic focused check;
  - independently implemented slices share state, a call chain, format, acceptance invariant, or producer/consumer boundary; or
  - implementer self-review leaves an unresolved concern or untraced consumer.
- For observably coupled slices, review the integrated delta once. Do not create review merely because a task ended, a diff exists, or an earlier task used review.
- Reviewer scope comes from current task requirements, applicable governing constraints, and actual changed or affected producer/consumer boundaries—not “owned modules,” files present in a diff, commands previously run, or ledger wording.
- Replace semantic progress ledgers with a minimal durable-recovery contract only where interruption, multi-agent handoff, compaction, or external side effects make recovery necessary.
- Recovery state may record immutable candidate identity, current authoritative plan/spec revisions, explicit task IDs, checkpoint state, claim-bound evidence digests, factual receipts for in-flight effects, and references to already approved durable decisions.
- Recovery state must not store free-text requirements, newly inferred gates, test ownership, controller procedure, severity policy, or historical verification conventions.
- On resume: verify candidate and evidence identities, reread current authority, rederive current gates, and use the ledger only to avoid repeating completed external effects or losing factual work.
- Explicitly expire or freeze task-local controller procedure when the task or risk closes.

### Conditional recommendation

A machine-enforced authority/expiry schema is appropriate only for persistent orchestrator workflows whose state is already generated and validated automatically. Do not create a new registry, waiver database, or per-command provenance artifact for ordinary agent work.

## Stage 4: Todo, Cleanup, And Delivery Admission

### Targets

Harness-level todo rules, cleanup rules, delivery-completeness language, and any reusable plan-execution wording that currently turns optional work into task state.

### Changes

- Todos may track current user deliverables, applicable governing requirements, explicitly approved active-plan tasks, and the smallest concrete prerequisites or focused risk-control steps directly necessary to satisfy them.
- Every newly introduced blocking item must name its current authority or direct causal link, concrete trigger or evidence, affected invariant or deliverable, and closure condition.
- User-enumerated deliverables remain tracked exactly as requested.
- Controller preferences, historical practices, bookkeeping, and diagnostics are not authority. A diagnostic may justify only the linked repair and recheck when it reproducibly demonstrates violation of current authority.
- Post-implementation cleanup must inspect direct producers, consumers, callsites, contract-bearing tests, documentation, metadata, indexes, or manifests affected by changed behavior, interfaces, or data.
- Update only artifacts made incompatible, materially stale or misleading, incomplete relative to an existing completeness claim, or unable to substantiate current acceptance.
- Remove temporary execution or diagnostic scaffolding.
- Re-run final required proof after cleanup when cleanup changes the candidate in a way relevant to the claim.
- Do not create broad changelog, documentation, test-expansion, or general cleanup work merely because implementation occurred.
- Replace vague “all affected artifacts” language with direct causal and contract-bearing scope.

### Behavioral acceptance scenarios

- A consumer migration discovered during implementation is admitted through direct causal necessity even if absent from the initial file list.
- An optional diagnostic cannot become a durable blocking todo merely because it might be useful.
- A real out-of-scope defect is reported as nonblocking and tracked only through explicit adoption.
- Cleanup that changes executable behavior invalidates earlier proof and triggers only the relevant post-cleanup check.
- A literally true but materially misleading document is updated even though implementation did not make a sentence syntactically false.

## Stage 5: Procedural Ratchet Detection

### Target

Add a focused recovery section to `consistency-quality-pass` after Stages 1–4 stabilize.

### Changes

- When procedural ratcheting is suspected, locate the current normative source and safety invariants.
- Trace each claimed gate to that source.
- Treat todos, reports, reviews, ledgers, summaries, and prior executions as evidence unless current authority explicitly adopts their exact requirement.
- Remove stale duplicated process wording rather than harmonizing every dependent artifact around it.
- Preserve factual observations that establish a current contract violation.
- Expire controller-added tactics when their matched task, phase, or risk ends.
- Search the focused concept footprint for controller-coined normative labels such as `owned tests`, `owned modules`, `standard gate`, `established convention`, `required because previously run`, and `complete suite for this task`; replace each with the exact governing criterion or classify it as a task-local diagnostic when no governing source exists.
- Do not create a process registry, waiver ledger, or per-action provenance artifact.

This is defense in depth, not the primary prevention mechanism. It should activate only after identified authority drift, not force a repository-wide procedural sweep for every task.

## Rejected Rules

The following formulations must not be adopted:

- “Run the smallest check capable of falsifying the claim.” Capability to falsify is necessary but not sufficient to establish a broad claim.
- “Supplemental checks are always nonblocking.” A supplemental observation can prove a real current contract or safety violation.
- “Invoke skills only when the controller identifies material risk.” This permits rationalized omission of deterministic preventive workflows.
- “Stack skills only when one output is a prerequisite for the next.” Distinct risks may require complementary debugging, TDD, safety, and verification controls.
- “Do not use durable progress state.” Long-running, interruptible, multi-agent, or externally effectful workflows need factual recovery state; the fix is to remove semantic authority from it.
- “Only update artifacts made incorrect.” Artifacts can be materially stale, misleading, incomplete, or unable to support a current claim while remaining literally true.
- “Never stop for a supplemental check.” A check's provenance does not determine whether its observation proves a current violation.
- “All independent tasks require per-task review” or “review often.” Review needs an authority or material-boundary predicate and a stop condition.
- “Every newly found issue becomes a todo.” Admission requires current authority or a direct causal link to a current requirement, invariant, or deliverable.
- A new universal process registry, waiver workflow, or provenance ceremony. Persistent procedure belongs in existing governing sources.

## Cross-Stage Simulation Matrix

Before adopting each stage, compare baseline and candidate behavior on at least these scenarios:

1. Exact focused selector required; authorized files do not imply complete modules.
2. Supplemental full-module run passes; procedure expires at task close.
3. Supplemental run finds an unrelated superseded historical failure; disclose without repinning.
4. Supplemental run finds a real current no-drift regression; block through the governing invariant.
5. Named selector collects zero tests; evidence remains missing and the selector is repaired or authority resolved.
6. Public producer/consumer boundary changes; focused integration evidence and review are admitted.
7. Required selector is flaky; rerun-until-green does not erase contradictory evidence.
8. Reproducible bug fix activates debugging, TDD, and verification.
9. Necessary emergent consumer work is admitted through direct causal necessity.
10. Historical review cadence does not become a requirement for a later task with no admitted review boundary.

For each simulation, record the authoritative acceptance unit, selected procedure, evidence collected, blocking classification, expiry decision, and whether any derived artifact incorrectly created a future obligation.

## Completion Criteria

The backlog is complete only when:

- each adopted stage has baseline and candidate behavior traces;
- the adopted wording passes the cross-stage simulation matrix without suppressing genuine defects or expanding process from historical artifacts;
- replaced contradictory wording is removed rather than left beside the new rule;
- repository and reusable-skill surfaces agree on authority, evidence, expiry, review admission, recovery state, todo admission, and cleanup scope; and
- any persistent orchestrator schema change is tested through actual routing/state behavior rather than literal prompt-text assertions.
