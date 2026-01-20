# Router Routing Contract (Phase A)

## Deterministic Routing

Inputs (authoritative):
- `sync/state.json`: `iteration`, `expected_actor`, `status`
- Config parameters: `review_every_n`, `prompt_map`, `allowlist`

Definitions:
- `expected_actor` values map to prompts: `galph -> supervisor.md`, `ralph -> main.md`.
- Reviewer prompt is actor-agnostic by default (may be actor-specific if configured).

Algorithm:
1) `base_prompt = prompt_map[expected_actor]`.
2) If `review_every_n > 0` and `iteration % review_every_n == 0`, set `candidate = prompt_map['reviewer']`.
3) Else set `candidate = base_prompt`.
4) Validate `candidate` is in `allowlist` and the prompt file exists.
5) Validate `candidate` is allowed for `expected_actor` (reviewer prompt allowed for both actors by default).
6) Deterministic routing result is `candidate`.

## Router Prompt Override

- Optional router prompt runs after deterministic selection.
- Router prompt output must be a single, non-empty line naming a prompt path or name.
- If the output is valid and allowed, it overrides the deterministic result.
- If the output is invalid (empty, not in allowlist, file missing, or actor-disallowed), crash with a descriptive error and do not dispatch.

## State File Extension

- Add `last_prompt` (string) to `sync/state.json`.
- Router writes only the final selected prompt to `last_prompt`.
- No additional router metadata is persisted in state.json.

## Failure Behavior

- Missing review prompt on a review iteration -> crash with a descriptive error.
- Invalid router prompt output -> crash with a descriptive error.
- Router never mutates `expected_actor` or `status`.

## Notes

- `galph` and `ralph` are role labels only; they map to `supervisor.md` and `main.md`.
- YAML/config provides parameters only; routing logic lives in code.
