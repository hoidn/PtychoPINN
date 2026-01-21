### Turn Summary
Created orchestration.yaml config file at repo root with router, agent, and spec_bootstrap sections to centralize orchestration configuration.
Updated prompts/arch_writer.md and prompts/spec_reviewer.md to reference correct doc paths (docs/index.md, specs/*.md) instead of stale docs/spec-shards/ references.
Implemented --no-git flag in supervisor.py to skip all git operations (branch guard, pull, submodule scrub, add, commit, push, autocommit) while still running prompt execution and logging locally.
Created test_supervisor.py::TestNoGit with 4 tests validating --no-git behavior; all tests pass (4 passed in 0.05s).
Next: commit changes and push to remote.
Artifacts: plans/active/DOC-HYGIENE-20260120/reports/2026-01-20T235033Z/ (pytest_supervisor_no_git.log, pytest_router_config.log)
