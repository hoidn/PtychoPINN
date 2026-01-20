# User Input — 2026-01-20

## Summary
- Root-level orchestration config is still missing even though docs/fix_plan.md treats it as present, so reviewers cannot read `router.review_every_n` / `logs_dir` from the documented source of truth.
- `scripts/orchestration/supervisor.py` exposes a `--no-git` flag but never checks it, so “no git” runs still pull/push, auto-commit, and will fail in environments without git access.
- Prompt templates still reference non-existent documentation roots (`docs/architecture/data-pipeline.md`, `docs/spec-shards/*.md`), undermining the doc-hygiene fixes that were declared complete.

## Evidence
- docs/fix_plan.md:520-535 claims the repository now ships `orchestration.yaml`, but no such file exists anywhere under the repo root (see `rg --files -g 'orchestration.yaml'` returning nothing).
- scripts/orchestration/supervisor.py:34-140 adds `--no-git`, yet the file never reads `args.no_git`, so git pulls/pushes/auto-commits always run.
- prompts/arch_writer.md:38-55 and prompts/spec_reviewer.md:20-34 still reference `docs/architecture/data-pipeline.md` and `docs/spec-shards/*.md`, neither of which exists (specs live under `specs/` and architecture docs are flat files such as `docs/architecture.md`).

## Requested Next Steps
1. Add the promised root-level `orchestration.yaml` (router.review_every_n/state_file/logs_dir) and commit it so reviewer tooling has an authoritative config source.
2. Wire the supervisor’s `--no-git` flag through the pull/push/auto-commit code paths (and guard against git usage when it is set) so local-only/spec bootstrap runs stop failing.
3. Update `prompts/arch_writer.md` and `prompts/spec_reviewer.md` to reference valid docs (`docs/architecture.md`, `../specs/*.md`), keeping docs/index.md as the canonical map.
