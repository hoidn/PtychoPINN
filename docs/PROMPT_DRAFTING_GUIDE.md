# Prompt Drafting Guide

This guide defines how to write prompts for orchestrated agent runs in PtychoPINN.
It is based on failure modes observed in backlog-loop execution and review/fix cycles.
For workflow/step/prompt/plan boundaries, read `docs/workflows/orchestration_start_here.md` first.
This guide is prompt-authoring guidance for provider steps, not workflow routing or DSL design guidance.

## Primary Rules

1. Keep abstraction boundaries clean.
- Prompt text should describe the current invocation only.
- Do not encode workflow control logic in prompt language unless that prompt is explicitly the control-plane step.

2. Use contract-driven IO only.
- Inputs come from injected `Consumed Artifacts`.
- Outputs are written exactly as defined by the injected output contract.
- Avoid hardcoded filesystem handshake paths in prompt text.

3. KISS by default.
- Prefer short, plain-text instructions over schema-heavy output formats.
- Add structure only when a downstream parser or gate requires it.

4. Be scope-flexible.
- Prompts should handle one file or many files without branching into separate prompt variants unless behavior truly differs.
- Scope rules should be deterministic and explicit.

5. Fail closed on ambiguity.
- If required source artifacts are missing or ambiguous, instruct the agent to stop and report blockers.
- Do not allow "best guess" path discovery from historical directories.

## Recommended Prompt Shape

Use this order:

1. Input contract
- "Use the injected `Consumed Artifacts` as the authoritative input list."

2. Task
- One paragraph describing the objective in plain language.

3. Scope resolution
- How to derive scope from consumed artifacts (single/multiple plan paths, backlog items that reference plans, dedupe behavior).

4. Allowed fix classes
- Enumerate the specific defect classes to fix.
- Keep this list short and causally tied to known failures.

5. Actions
- 2-5 ordered action bullets.

6. Outputs
- Describe output content, not fixed destination paths.
- Explicitly say outputs must follow the injected output contract.

7. Constraints
- No unrelated edits.
- No fabricated evidence.
- No queue movement unless prompt purpose is queue movement.

## When Structured Output Is Justified

Use JSON/schema output only if at least one is true:
- A machine parser consumes the response directly.
- A gate checks specific keys/values.
- You need stable fields across retries for automation.

If none apply, use concise plain text.

## Anti-Patterns To Avoid

- Hardcoded `state/...` output paths in prompt body.
- Timestamped or machine-local path literals as required inputs.
- Mixing invocation-local instructions with workflow routing semantics.
- Requiring verbose templates when a short report is sufficient.
- Implicit scope ("figure out which plan") without deterministic resolution rules.

## Minimal Example Skeleton

```md
Use the injected `Consumed Artifacts` as the authoritative input list and read all listed artifacts first.

Task:
<one-paragraph objective>

Scope resolution:
- <deterministic rules for deriving in-scope files>
- If scope cannot be resolved, report blocker and return non-approval decision.

Fix only:
1. <defect class A>
2. <defect class B>

Actions:
1. Review in-scope files against listed defect classes.
2. Edit only in-scope files to fix confirmed issues.
3. Record unresolved blockers explicitly.

Outputs:
- <concise report contents>
- <decision semantics>
- Write outputs exactly as specified by the injected output contract.

Constraints:
- <no unrelated edits>
- <no fabricated evidence>
```

## Quick Review Checklist

Before adopting a new prompt, verify:
- No hardcoded IO paths in prompt body.
- Inputs/outputs are contract-driven.
- Scope rules cover single and multi-item cases.
- Instructions are short and unambiguous.
- Output format complexity is justified.
