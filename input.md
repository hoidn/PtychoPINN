Summary: Add the missing orchestration.yaml, fix the reviewer prompts to reference real docs/specs, and wire supervisor --no-git gating plus tests so offline runs stop touching git.
Focus: DOC-HYGIENE-20260120 — Phase A/B/C config + prompt + supervisor guard
Branch: paper
Mapped tests:
  - pytest scripts/orchestration/tests/test_router.py::test_router_config_loads -v
  - pytest scripts/orchestration/tests/test_supervisor.py::TestNoGit::test_sync_iteration_skips_git_ops -v
Artifacts: plans/active/DOC-HYGIENE-20260120/reports/2026-01-20T235033Z/

Do Now:
- Implement: orchestration.yaml — check in the root-level config capturing prompts_dir, router.review_every_n=3, logs_dir, state_file, doc/tracked-output whitelists, agent defaults, and spec_bootstrap dirs so reviewers have a real source of truth.
- Implement: prompts/arch_writer.md — replace references to docs/architecture/data-pipeline.md and docs/spec-shards with docs/architecture.md + specs/spec-ptycho-workflow.md#pipeline-normative (plus specs/spec-ptycho-interfaces.md#data-formats-normative) and remind writers to start at docs/index.md.
- Implement: prompts/spec_reviewer.md — update required reading + spec_bootstrap sections to reference docs/index.md, specs/ (shards live under specs/), and the new orchestration.yaml spec_bootstrap entries instead of docs/spec-shards/*.md.
- Implement: scripts/orchestration/supervisor.py::main — honor --no-git in both legacy and sync modes (skip branch guard, _pull_with_error, add/commit/push/auto-commit helpers, submodule scrub) while still running prompt execution/logging.
- Implement: scripts/orchestration/tests/test_supervisor.py::TestNoGit (new module) — stub git_bus helpers to prove the supervisor never calls them when --no-git is set; keep an async/sync test variant. Update docs/TESTING_GUIDE.md §2 and docs/development/TEST_SUITE_INDEX.md with the new selector + instructions on storing pytest logs under the artifacts hub.

How-To Map:
1. Create orchestration.yaml at repo root with YAML keys from scripts/orchestration/config.OrchConfig (router block, agent block, spec_bootstrap.templates_dir/specs.dir/implementation dirs). Drop it alongside README; no secrets, check in.
2. Rerun config sanity tests: `pytest scripts/orchestration/tests/test_router.py::test_router_config_loads -v` and `pytest scripts/orchestration/tests/test_agent_dispatch.py -v`; stash the logs to plans/active/DOC-HYGIENE-20260120/reports/2026-01-20T235033Z/.
3. Update prompts/arch_writer.md + prompts/spec_reviewer.md per plan, keeping XML tags intact and ensuring docs/index.md + specs/spec-ptycho-*.md anchors match the links.
4. Modify scripts/orchestration/supervisor.py so args.no_git short-circuits branch guard, _pull_with_error, submodule scrub, add/commit/push, and auto-commit helpers. When sync-via-git is requested with --no-git, still update state.json locally but treat git bus calls as no-ops.
5. Author tests in scripts/orchestration/tests/test_supervisor.py that monkeypatch git_bus/autocommit helpers (raise if invoked) and run supervisor.main() in sync mode with --no-git to assert no git calls occur while prompts execute. Capture tee_run invocations via stub.
6. Run `pytest --collect-only scripts/orchestration/tests/test_supervisor.py -q`, `pytest scripts/orchestration/tests/test_supervisor.py -v`, and `pytest scripts/orchestration/tests/test_orchestrator.py::test_combined_autocommit_no_git -v`; dump logs to the artifacts hub.
7. Update docs/TESTING_GUIDE.md §2 + docs/development/TEST_SUITE_INDEX.md with the new supervisor selector + log archiving instructions; mention orchestration.yaml’s new location in docs/index.md if needed.

Pitfalls To Avoid:
1. Do not run real git commands inside tests; always monkeypatch git bus helpers.
2. Keep YAML 2-space indented and ASCII only; no tabs or secrets.
3. Preserve prompt XML tags and escaping; only change doc references.
4. Supervisor --no-git must skip both branch guard and auto-commit helpers—forgetting one still triggers git.
5. No edits to stable physics modules (CLAUDE.md §6); keep changes in orchestration/prompt surfaces.
6. Follow PYTHON-ENV-001 — invoke python scripts/tests via `python` or pytest; no repo-specific wrappers.
7. Store pytest logs under the artifacts hub; avoid cluttering repo root.
8. Update docs/test registries once new selectors land; do not skip the documentation gate.
9. Ensure new tests assert they were called (e.g., tee_run stub) so they fail if supervisor exits early.
10. Keep router cadence defaults (review_every_n=3, allowlist prompts) consistent between YAML and README.

If Blocked:
- Capture failing command + log snippet under the artifacts hub, set docs/fix_plan.md Attempts entry to `blocked`, and DM Galph with the error plus reproduction. If git helpers still fire despite --no-git, log the stack trace and short-circuit by raising RuntimeError in the stub to keep the failure obvious.

Findings Applied (Mandatory):
- TEST-CLI-001 — orchestration CLI changes require explicit pytest selectors + log artifacts.
- PYTHON-ENV-001 — run python/pytest via PATH `python`; no ad-hoc interpreter wrappers.

Pointers:
- plans/active/DOC-HYGIENE-20260120/implementation.md:1 — phased plan + checklist IDs.
- scripts/orchestration/README.md:1 — router cadence + orchestration config contract.
- scripts/orchestration/supervisor.py:1 — CLI entrypoint to update for --no-git.
- prompts/arch_writer.md:1 — architecture prompt references to fix.
- prompts/spec_reviewer.md:1 — spec reviewer prompt references/spec_bootstrap block.

Next Up (optional): Validate combined orchestrator docs once supervisor fix lands (rerun `pytest scripts/orchestration/tests/test_orchestrator.py -v` and refresh docs/index.md router references).

Doc Sync Plan:
- After adding `scripts/orchestration/tests/test_supervisor.py`, run `pytest --collect-only scripts/orchestration/tests/test_supervisor.py -q` and archive the log to the artifacts hub before implementation is declared done.
- Once the new tests pass, update `docs/TESTING_GUIDE.md` §2 and `docs/development/TEST_SUITE_INDEX.md` with the supervisor selector + expected log filenames.

Mapped Tests Guardrail:
- `scripts/orchestration/tests/test_supervisor.py` must collect >0; run the collect-only command before claiming success. Do not finish the loop until the new selector exists and is documented.

Normative Reference:
- See scripts/orchestration/README.md §Router + §Sync via Git for the authoritative behavior this change must preserve.
