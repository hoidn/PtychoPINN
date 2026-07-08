# Workflow-Contract Plan Template

Use this template when a plan is intended to run inside an orchestrated workflow.

## 1) Plan Metadata

- Plan title:
- Plan owner:
- Date:
- Linked backlog item:
- Linked workflow file:

## 2) Workflow Contract Surface (Required, Strictly `< 3`)

List only the key plan-level artifacts that must cross step/stage boundaries at workflow level.

Hard limit:
- Total contract artifacts must be 1 or 2.
- If more are needed, split the plan or explicitly pause and redesign with workflow author.

### Contract Artifact 1

- Artifact name:
- Direction: input | output
- Producer (workflow step/stage):
- Consumer (workflow step/stage):
- Artifact type: `relpath` | scalar (`enum`/`integer`/`string`) | JSON bundle field
- Canonical resolution rule:
  - Must resolve from the current producer artifact.
  - Must not use timestamp-pinned or hardcoded historical output paths.
- Validation/gate rule:

### Contract Artifact 2 (Optional)

- Artifact name:
- Direction: input | output
- Producer (workflow step/stage):
- Consumer (workflow step/stage):
- Artifact type: `relpath` | scalar (`enum`/`integer`/`string`) | JSON bundle field
- Canonical resolution rule:
  - Must resolve from the current producer artifact.
  - Must not use timestamp-pinned or hardcoded historical output paths.
- Validation/gate rule:

## 3) Non-Contract I/O (Implementation-Local)

- List any additional inputs/outputs used by scripts or prompts.
- These are local details and must not be wired as workflow-level producer/consumer artifacts.

## 4) Plan <-> Workflow Coordination (Required)

Plan author + workflow author must both confirm:

- [ ] Contract artifact count is `< 3` (max two artifacts).
- [ ] Each contract artifact has a named producer and consumer.
- [ ] Workflow DSL `artifacts`/`publishes`/`consumes` matches this plan exactly.
- [ ] No hardcoded stale run-root paths are used for inter-stage artifact resolution.
- [ ] Consumer steps fail closed when required producer artifact is missing or stale.

- Workflow author sign-off (name/date):
- Plan author sign-off (name/date):

## 5) Acceptance Criteria (Contract-Level)

- [ ] All contract artifacts are produced and consumed through workflow semantics.
- [ ] Evidence shows producer->consumer lineage for each contract artifact.
- [ ] No fallback to historical timestamped paths for contract artifacts.

