# Seed Plan

## Directive
Check `inbox/` and act accordingly.

## Scope
- Monitor `inbox/` for maintainer requests.
- Respond with a concrete action or status update.

## Checklist
- [x] S1: Check `inbox/` for new requests and read them.
- [x] S2: If a request exists, document the response plan and update `fix_plan.md`.
- [x] S3: If no requests exist, record the check in `fix_plan.md`.
- [ ] S4: Close out the D0 parity logger deliverable by surfacing stage-level stats for every dataset in the Markdown log and syncing the new CLI/test selector through the testing docs.

## Notes
- Keep responses in the maintainer inbox format (one file per request).
- Avoid modifying core modules without an explicit plan.
- Disambiguation: use the `<maintainers>` block in `CLAUDE.md` and the `<maintainer_communication>` section in
  `prompts/supervisor.md` when communicating; label requests as either (a) local agent actions in this repo or
  (b) dose_experiments maintainer actions, and state the target role explicitly.
