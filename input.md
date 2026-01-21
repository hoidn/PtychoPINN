Summary: Align the spec-bootstrap tooling with the canonical specs/ directory and prove the updated config/tests remain green.
Focus: DOC-HYGIENE-20260120 — Phase B3 spec bootstrap alignment
Branch: paper
Mapped tests:
  - pytest scripts/orchestration/tests/test_router.py::test_spec_bootstrap_defaults -v
  - pytest scripts/orchestration/tests/test_router.py::test_router_config_loads -v
Artifacts: plans/active/DOC-HYGIENE-20260120/reports/2026-01-21T000742Z/

Do Now:
- Implement: scripts/orchestration/config.py::SpecBootstrapConfig — point specs_dir defaults to Path("specs"), have discover_shards() look under templates_dir / "specs" first with a fallback to templates_dir / "docs/spec-shards", and keep load_config() constructing the SpecBootstrapConfig even when orchestration.yaml omits the specs.dir block so repos without docs/spec-shards/ still get a valid config.
- Implement: scripts/orchestration/init_project.sh::init_directories — replace the docs/spec-shards scaffolding with specs/ (including copy_dir_files falling back to templates/docs/spec-shards if templates/specs is missing), mirror the same behavior inside scripts/orchestration/init_spec_bootstrap.sh (SPECS_DIR constant + template discovery), and refresh scripts/orchestration/README.md plus prompts/arch_reviewer.md text so reviewers are told to read specs/* instead of the dead docs/spec-shards/* path.
- Implement: scripts/orchestration/tests/test_router.py::test_spec_bootstrap_defaults — add a pytest that writes a minimal orchestration.yaml with a spec_bootstrap block lacking specs.dir, asserts cfg.spec_bootstrap.specs_dir == Path("specs"), and exercises the legacy template fallback by seeding fake shard files under both specs/ and docs/spec-shards/ to prove the new search order before finalizing logs.

How-To Map:
1. Set ARTIFACTS=plans/active/DOC-HYGIENE-20260120/reports/2026-01-21T000742Z and mkdir -p "$ARTIFACTS/cli" so pytest logs land under the new hub.
2. Update scripts/orchestration/config.py + tests first so the new defaults exist before touching the shell scripts; keep discover_shards() tolerant of legacy template layouts.
3. Patch scripts/orchestration/init_project.sh (init_directories + copy_dir_files call) and scripts/orchestration/init_spec_bootstrap.sh so they mkdir specs/ and copy shard templates there while falling back to templates/docs/spec-shards when templates/specs is absent. Keep the helper echo/logging structure untouched.
4. Rewrite the spec_bootstrap snippet inside scripts/orchestration/README.md plus the required-reading block in prompts/arch_reviewer.md to cite specs/ + docs/index.md, then rerun `rg -n "docs/spec-shards" scripts/orchestration prompts -n` to confirm only historical references remain.
5. Run pytest --collect-only scripts/orchestration/tests/test_router.py::test_spec_bootstrap_defaults -q | tee "$ARTIFACTS/cli/pytest_spec_bootstrap_defaults_collect.log" to prove the selector exists before edits continue.
6. Run pytest scripts/orchestration/tests/test_router.py::test_spec_bootstrap_defaults -v | tee "$ARTIFACTS/cli/pytest_spec_bootstrap_defaults.log".
7. Run pytest scripts/orchestration/tests/test_router.py::test_router_config_loads -v | tee "$ARTIFACTS/cli/pytest_router_config_loads.log" to guard the existing YAML parsing path.
8. Stage updated docs/TESTING_GUIDE.md or TEST_SUITE_INDEX.md only if you needed to add new selectors (current selectors already cover test_router, so no update expected this loop).

Pitfalls To Avoid:
- Keep SpecBootstrapConfig optional; do not assume spec_bootstrap exists when orchestration.yaml omits the section.
- Preserve backwards compatibility: discover_shards() and init scripts must fall back to templates/docs/spec-shards when templates/specs is missing.
- Do not delete historical references in docs/fix_plan or plan files; only clean living docs/prompts.
- Maintain ASCII YAML/Markdown formatting and two-space indentation in orchestration.yaml snippets.
- When editing shell scripts, avoid bashisms that break on macOS dash; stick with existing helper functions.
- Tests must not touch the real templates directory—use tmp_path fixtures only.
- Capture pytest logs under the artifacts hub or TEST-CLI-001 will be violated.
- Follow PYTHON-ENV-001 and run pytest via PATH `pytest`; no repo-specific wrappers.

If Blocked:
- Capture the failing command + stderr into $ARTIFACTS/blocked.log, note the reproduction in docs/fix_plan.md Attempts History, and ping Galph with whether the blocker is in config.py vs. shell scripts so we can rescope B3 accordingly.

Findings Applied (Mandatory):
- TEST-CLI-001 — Archive pytest logs for the new spec-bootstrap guard selectors under the artifacts hub.
- PYTHON-ENV-001 — Invoke pytest via the PATH python environment; no repo-specific interpreters.

Pointers:
- plans/active/DOC-HYGIENE-20260120/implementation.md:76 — B3 checklist + expected tests.
- scripts/orchestration/config.py:93 — SpecBootstrapConfig defaults and discover_shards helper that now need specs/ support.
- scripts/orchestration/README.md:15 — Spec_bootstrap sample YAML that still cites docs/spec-shards.
- scripts/orchestration/init_project.sh:224 — init_directories currently creating docs/spec-shards.
- scripts/orchestration/init_spec_bootstrap.sh:35 — SPECS_DIR/template discovery that must switch to specs/.
- prompts/arch_reviewer.md:24 — Required reading block that still points to docs/spec-shards/.

Next Up (optional): After B3 closes, unblock ORCH-AGENT-DISPATCH-001 by refreshing input.md for that initiative.

Doc Sync Plan: Not needed — test_router is already listed in docs/TESTING_GUIDE.md and TEST_SUITE_INDEX.md; no new selectors introduced.

Mapped Tests Guardrail: Verify the new test_spec_bootstrap_defaults collects (>0) before editing other files; if pytest --collect-only ever returns 0, stop and ensure the test exists before proceeding.

Normative Reference: scripts/orchestration/README.md §Spec Bootstrap defines the root config contract this change must follow; keep the implementation in lockstep.
