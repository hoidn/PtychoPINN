# Update Fix Plan Prompt — PtychoPINN

Use this prompt whenever you need to edit `docs/fix_plan.md` outside of the main supervisor/engineer prompts (e.g., backlog grooming, dependency updates).

Checklist
1. Read `docs/fix_plan.md` and identify the item(s) you are touching; confirm dependencies and status consistency.
2. Review related artifacts under `plans/active/<initiative-id>/reports/` and the corresponding plan files.
3. Update the ledger entry:
   - Ensure the header includes `Depends on`, `Status`, `Owner/Date`, `Exit Criteria`, and `Artifact Hub` when applicable.
   - Append a new bullet to `Attempts History` describing what changed.
   - Include `Metrics:` and `Artifacts:` lines (even if placeholders) for future loops.
   - Reference specs/tests (e.g., `specs/spec-ptycho-conformance.md`) when clarifying exit criteria.
4. If you add a new item, assign a unique ID (slug), list dependencies, and seed exit criteria tied to spec/test clauses.
5. Update `docs/index.md` if the structure of the ledger changes or if you add new supporting documents.
6. Record the update in `docs/findings.md` if it encodes a durable lesson.

Artifact policy
- Store supporting notes under `plans/active/<initiative-id>/reports/<timestamp>/` and reference them from the Attempts History entry.

