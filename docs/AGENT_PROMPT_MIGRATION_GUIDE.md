# Agentic Prompt Migration Guide

This guide captures the workflow we followed when modernising the prompts under `prompts/` to match the current PtychoPINN documentation layout, plan artefact conventions, and testing references. Use it whenever repository structure shifts (e.g., new specs, renamed docs, relocated artefacts) and the agentic prompts must be brought back into alignment.

## Why This Matters
- Prompts are the control plane for the supervisor/engineer loop; stale paths break autonomy and can trap loops in dead ends.
- Documentation is the authoritative source of truth. Every reference inside a prompt must point at living files so agents discover the right context before acting.
- Plan artefacts are now stored under `plans/active/<initiative>/reports/…`; prompts must steer outputs there for traceability.

## Prerequisites
1. Read `docs/index.md` to understand the up‑to‑date documentation map.
2. Familiarise yourself with the normative specs:
   - `specs/ptychodus_api_spec.md`
   - `specs/data_contracts.md`
3. Review the key guides that replaced legacy references:
   - `docs/architecture.md`
   - `docs/DEVELOPER_GUIDE.md`
   - `docs/TESTING_GUIDE.md`
   - `docs/development/TEST_SUITE_INDEX.md`
4. Confirm the artefact storage policy in `CLAUDE.md` (plans/active reports convention).

## Migration Workflow

### 1. Inventory Existing References
- Use `rg` to list stale patterns (e.g., `rg "spec-a" prompts`, `rg "arch.md" prompts`).
- Track usages of old artefact paths (`reports/`, `scripts/validation/`, `golden_suite_generator/`).
- Note prompts that encode legacy examples (e.g., NanoBragg CLI snippets) and decide on modern replacements.

### 2. Map to Current Sources
- Replace legacy spec references with the two authoritative specs listed above.
- Swap `arch.md` for `docs/architecture.md`, and fold in `docs/DEVELOPER_GUIDE.md` where architectural intent lives.
- Route testing lookups through `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md`.
- For PyTorch parity/config content, rely on `docs/workflows/pytorch.md` and plans under `plans/ptychodus_pytorch_integration_plan.md`.

### 3. Update Artefact Destinations
- Ensure every prompt that writes evidence points at `plans/active/<initiative>/reports/<timestamp>/…`.
- When prompts instruct supervisors/engineers to read artefacts, reference the same reports directory.
- Remove hard-coded filenames (e.g., `reports/debug/...`, `golden_suite_generator/nanoBragg`) in favour of doc-driven lookups.

### 4. Modernise Examples and Templates
- Replace NanoBragg-specific command blocks with current workflows (e.g., `python -m unittest tests.test_integration_workflow`).
- Update fix-plan templates to show integration workflow metrics and the new archive practice (`archive/<date>_fix_plan_archive.md`).
- Refresh parity harness guidance to point at the documented matrix rather than hard-coded YAML file names when appropriate.

### 5. Validate Prompt Behaviour
- After edits, run `rg` again to confirm no stale tokens remain.
- Spot-check key prompts (`prompts/main.md`, `prompts/supervisor.md`, `prompts/debug.md`) to ensure the required-reading lists are current and consistent.
- Use `git diff` to verify only intended prompts changed and artefact directories adhere to the policy.

## Quick Checklist
- [ ] All spec references point to `specs/ptychodus_api_spec.md` / `specs/data_contracts.md`.
- [ ] Architecture references use `docs/architecture.md` (with `docs/DEVELOPER_GUIDE.md` when needed).
- [ ] Testing commands come from `docs/TESTING_GUIDE.md` or `docs/development/TEST_SUITE_INDEX.md`.
- [ ] Artefact destinations use `plans/active/<initiative>/reports/<timestamp>/…`.
- [ ] Examples reflect current workflows (integration, PyTorch parity, plan artefacts).
- [ ] Legacy tool paths (`scripts/validation/`, `golden_suite_generator/`) removed or documented with modern alternatives.

## Tips for Future Migrations
- Treat prompts as living documentation: when you update a guide, plan, or spec, immediately scan for prompts that reference it.
- Keep migrations incremental. Touch related prompts together so Commit history stays coherent.
- Record lessons learned (e.g., new artefact policies) in `docs/findings.md` so future edits have context.
- When in doubt, lean on `docs/index.md` as the single source of truth for where a topic now lives.

