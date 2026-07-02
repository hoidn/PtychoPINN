# CI Test Gate (main)

`main` only accepts changes through pull requests, and a PR only merges when
the `pytest-cpu` check is green **against the current tip of main** (the
ruleset's strict/up-to-date flag re-tests stale branches). Direct pushes and
force pushes to main are rejected server-side by the `main-test-gate` ruleset
(`ci/main-ruleset.json`).

The ruleset is enforced on the public repo (`hoidn/PtychoPINN`), where
rulesets are free. On the internal mirror the same workflow runs
advisory-only: a red check is visible on every push and PR but cannot block.

## What the gate runs

`bash ci/run_ci_tests.sh` — `tests/torch`, CPU-only (`CUDA_VISIBLE_DEVICES=""`),
`-m "not slow"`, minus the exclusion baseline:

- `ci/known_failures.txt` — node IDs `--deselect`ed: tests needing CUDA,
  untracked data (only git-tracked files exist in CI), or failing on main
  before the gate existed.
- `ci/collect_ignores.txt` — files `--ignore`d for module-level import errors.

Reproduce locally with the same command. A green local conda run does NOT
guarantee green CI (GPU and local data can mask failures); the CI environment
is authoritative.

## Ratchet policy

The exclusion baseline may only shrink. Removing an entry (after fixing the
test or its environment need) is always welcome. Adding an entry requires a
PR whose diff includes a comment line stating why, and is reserved for tests
that *cannot* pass in CI (new CUDA/data requirement) — never for silencing a
regression.

## When your PR is red

1. Read the failing node IDs in the `pytest-cpu` log.
2. If your change broke them: fix your change.
3. If the test newly requires CUDA or untracked data: move that requirement
   behind a skip-marker or add a baseline entry with justification (see
   ratchet policy) — expect that to be challenged in review.

## Known limitation

Two PRs individually green can still break main together (semantic conflict);
the strict up-to-date flag forces the second to re-test after the first
merges, which closes this for serial merges. If merge volume ever makes that
racy, enable a merge queue on the ruleset.
