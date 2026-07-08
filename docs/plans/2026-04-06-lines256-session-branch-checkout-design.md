# Lines256 Session Branch Checkout Design

## Goal

Replace the current detached-`HEAD` run-checkout behavior in the `lines_256`
controller workflow with a session-local branch model that is easier for humans
to inspect and resume, while keeping exact commit SHAs as the authoritative
scientific provenance.

## Problem

The current `lines_256` controller runs in a dedicated checkout, which is
correct, but source-candidate validation is still framed around `HEAD` matching
an exact `candidate_commit`. In practice this pushes the runtime toward
detached-commit operation.

That has several costs:

- humans inspecting the run checkout see detached `HEAD` instead of an obvious
  session state
- resume/debug flows are harder to reason about
- user-authored out-of-band changes in the dedicated checkout are less legible
- the controller’s ergonomics are worse than necessary even though commit SHAs,
  not branch names, are the real provenance contract

The problem is not that the workflow uses a dedicated checkout. The problem is
that the checkout has no stable operator-facing branch identity.

## Constraints

- Keep the dedicated run checkout model.
- Keep exact commit SHAs as the authority for:
  - accepted-state lineage
  - scored candidate identity
  - ledgers and resume checks
- Preserve the existing protected-local-paths contract for allowed user edits in
  the run checkout.
- Do not require a branch per candidate.
- It is enough for the accepted/session lineage to remain branch-attached.
  Discarded source candidates do not need persistent inspectable refs unless
  there is a clean optional mechanism.

## Approaches Considered

### 1. Session branch only

Keep one local branch per session, for example:

- `lines256/session/<session_id>`

The branch is just the ergonomic runtime handle. The controller still records
resolved commit SHAs and uses those SHAs for scientific provenance.

Pros:

- simple operator model
- minimal branch clutter
- compatible with current scoped-cleanup logic
- preserves current workflow structure

Cons:

- discarded source-candidate commits are inspectable only while still reachable
  from reflog or other refs

### 2. Session branch plus hidden candidate refs

Same as approach 1, but optionally write hidden refs such as:

- `refs/lines256/sessions/<session_id>/iterations/<n>`

before resetting away discarded source candidates.

Pros:

- keeps discarded candidates inspectable without cluttering normal branch lists

Cons:

- extra retention and cleanup policy
- more moving pieces than the workflow currently needs

### 3. Per-iteration branches

Create a new branch for each source candidate.

Pros:

- maximally inspectable

Cons:

- too much branch clutter
- worse resume/cleanup ergonomics
- pushes deterministic orchestration complexity into Git naming/state

## Recommendation

Use **approach 1** now:

- one session-local branch per run checkout
- exact commit SHAs remain authoritative
- optional hidden candidate refs can be added later if real forensic retention
  becomes necessary

This gives the workflow normal Git ergonomics without weakening provenance.

## Proposed Design

### Session start

When a new session is created:

1. ensure the dedicated run checkout exists
2. create or reset a local branch named:
   - `lines256/session/<session_id>`
3. check out that branch at the baseline accepted ref
4. persist `session_branch` in `session.json`

The persisted branch name is an operational handle, not the scientific identity.

### Source proposals

For `source` candidates:

- the provider edits and commits on the checked-out session branch
- after proposal resolution, the controller records:
  - `resolved_candidate_commit = git rev-parse HEAD`
- that resolved SHA becomes the authoritative candidate identity
- provider-supplied `candidate_commit` remains advisory and may still be logged
  for mismatch diagnostics

This keeps the workflow aligned with the existing “resolved commit beats provider
claim” direction already emerging in the controller.

### Keep / discard behavior

On `KEEP`:

- leave the session branch at the accepted commit
- update `accepted_ref` to that resolved commit SHA

On `DISCARD`, `TIMEOUT`, `CRASH`, or recoverable invalid execution:

- run the existing candidate-scoped cleanup
- move the session branch back to `accepted_ref`
- preserve protected local tracked dirt per the existing
  `protected_local_paths.json` contract

This means the branch always names the current accepted/session state, not the
last attempted candidate.

### Resume behavior

Resume should validate both:

- current branch name matches persisted `session_branch`
- `HEAD` matches the expected commit for the current phase

If the checkout is detached but otherwise recoverable, the controller may
reattach it to `session_branch` before continuing rather than failing with a
detached-state surprise.

### Protected local changes

The current protected-local-path model should stay intact.

The controller must continue to distinguish:

- accepted/session branch state
- candidate-scoped paths
- protected user edits in the dedicated checkout

The shift from detached `HEAD` to a session branch should not weaken the rule
that candidate cleanup must not clobber protected local tracked changes.

## Optional extension: hidden retention refs

If discarded source candidates later need durable inspectability, add an
optional controller-owned ref-write before reset:

- `refs/lines256/sessions/<session_id>/iterations/<n>`

This should be off by default. It is not required for the core redesign.

## Workflow-authoring boundary

This is controller-owned deterministic state, not prompt-owned behavior.

- prompts may still propose and commit source candidates
- the workflow/controller must own branch creation, branch validation, cleanup,
  resume attachment, and provenance recording

The prompt should not reason about branch names or Git retention policy.

## Non-goals

- Do not replace the dedicated run checkout with the human development checkout.
- Do not make branch names the scientific provenance surface.
- Do not introduce per-candidate branches.
- Do not require persistent refs for every discarded candidate in phase 1.

## Expected benefits

- easier human inspection of live run checkouts
- clearer resume semantics
- less confusion around “what state is this checkout in?”
- no loss of scientific reproducibility, because SHAs remain authoritative

## Follow-on implementation work

An implementation plan should cover:

1. session metadata changes (`session_branch`)
2. start-time branch creation/checkout
3. resume-time branch validation/reattachment
4. discard/cleanup reset semantics on a named session branch
5. tests that prove:
   - source candidates can be proposed and scored on a session branch
   - discard resets branch tip to accepted ref
   - protected local changes survive branch-based cleanup
   - detached-but-recoverable sessions can be reattached cleanly
