# CI Test Gate

This document is the authoritative reference for continuous integration on
PtychoPINN: what the gate is, how it runs, how to work with it day to day, and
how to maintain it. It is the policy of record for any change to the CI
configuration or the exclusion baseline.

---

## 1. What the gate guarantees

`main` only accepts changes through pull requests, and a PR merges only when the
required **`pytest-cpu`** check is green **against the current tip of `main`**.
Direct pushes and force pushes to `main` are rejected server-side.

Concretely, the gate blocks:

- **direct pushes to `main`** (including from admins — there are no bypass
  actors);
- **force pushes / history rewrites** on `main` (`non_fast_forward`);
- **branch deletion** of `main` (`deletion`);
- **merging any PR whose `pytest-cpu` check is failing, pending, or stale**
  relative to the current `main`.

The gate does **not** enforce human review (`required_approving_review_count`
is `0`). It is a *test* gate, not a *review* gate. Raise the approval count in
`ci/main-ruleset.json` later if review enforcement is wanted.

### Why "against the current tip of main"

The ruleset sets `strict_required_status_checks_policy: true`. This means a PR
branch must be up to date with `main` and re-tested before it can merge. A PR
that passed against an older `main` cannot land untested against a newer one.
This is the mechanism that catches **regressions introduced by interleaving
merges**, not just failures within a single branch.

---

## 2. Where enforcement lives (repo topology)

There are two remotes, and they are **not symmetric**:

| Remote | URL | Visibility | Role |
| --- | --- | --- | --- |
| `origin` | `git@github.com:hoidn/PtychoPINN.git` | **public** | enforcement home; ruleset is active here |
| `internal` | `git@github.com:hoidn/PtychoPINN-internal.git` | private | mirror; workflow runs advisory-only |

Enforcement is on the **public** repo because GitHub repository rulesets are
free on public repositories but require a paid plan on private ones
(Pro/Team/Enterprise — and Enterprise attaches to organizations, not personal
namespaces). The `main-test-gate` ruleset is therefore applied to
`hoidn/PtychoPINN`.

The **same** workflow file lives on both mains. On the internal mirror it runs
but cannot block — a red ✗ is visible on pushes and PRs there, it just is not
enforced. This is strictly better than no signal.

**Operating rule:** keep the two `main` branches identical. Land changes on the
public repo (through the gate), then fast-forward the internal `main` to the
same commit:

```bash
git fetch origin main
git push internal origin/main:refs/heads/main
```

> **Push policy:** `origin` is public. Do not push to it without explicit
> owner approval. Feature branches, probe branches, and `main` syncs to the
> public repo are all owner-approved actions, not routine automation.

---

## 3. What the gate runs

The check `pytest-cpu` runs a single command:

```bash
bash ci/run_ci_tests.sh
```

which expands to (see `ci/run_ci_tests.sh`):

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest tests/torch -m "not slow" -q -rf --color=yes \
  --deselect <each node ID in ci/known_failures.txt> \
  --ignore   <each file in ci/collect_ignores.txt>
```

Scope decisions baked into that command:

- **`tests/torch` only.** The active implementation is `ptycho_torch/`; its
  tests are the v1 gate scope. The TensorFlow-side suites (`tests/` top level,
  `tests/study`, `tests/scripts`, `tests/io`) are heavier on data/env coupling
  and are **out of scope for v1** (see §8, Non-goals).
- **`CUDA_VISIBLE_DEVICES=""`.** The runner is CPU-only. GitHub-hosted runners
  have no GPU; any test that hard-requires CUDA must be excluded (§6).
- **`-m "not slow"`.** Tests marked `slow` are skipped. (The `oom` marker is
  also registered and its tests never run without `--run-oom`.)
- **Only git-tracked files exist in CI.** The runner checks out the repo and
  nothing else — no local datasets, no generated artifacts. A test that reads
  an untracked data file will fail in CI even if it passes locally.
- **Extra args pass through.** `run_ci_tests.sh "$@"` forwards trailing
  arguments to pytest, so `bash ci/run_ci_tests.sh --collect-only` or
  `... -k some_expr` work for local debugging.

### The environment (`requirements-ci.txt` + workflow)

The CI environment is CPU-only and pinned deliberately. Two ordering/version
facts are load-bearing:

1. **Torch is installed from the CPU wheel index _before_ `requirements-ci.txt`**
   (`pip install torch --index-url https://download.pytorch.org/whl/cpu`). If
   torch were resolved from the default index it would pull the CUDA build.
2. **`tensorflow-cpu`** replaces the GPU TensorFlow flavor.

Two pins in `requirements-ci.txt` exist for specific reasons — do not remove
them without addressing the underlying cause:

- **`pytest<8`** — `tests/conftest.py` implements `pytest_ignore_collect` with
  the pre-pytest-8 `(path, config)` hook signature. pytest 8 removed that
  signature; under pytest 8 the plugin fails to register. (See §7 for the
  related conftest bug that was fixed.)
- **`tf-keras`** — `tensorflow-probability` on TensorFlow ≥ 2.16 imports Keras 2
  through the `tf-keras` shim; without it, collection fails on import.

The workflow (`.github/workflows/tests.yml`) also:

- triggers on `pull_request` and on `push` to `main` (so `main` always carries
  a fresh status even for changes that somehow bypass a PR);
- uses `concurrency` with `cancel-in-progress: true`, so a new push to a branch
  cancels that branch's older in-flight run;
- runs with a least-privilege token (`permissions: contents: read`);
- caps at `timeout-minutes: 45` (a normal run is ~4 minutes end to end,
  install-dominated; the test phase itself is ~30–45 s).

---

## 4. The exclusion baseline

Two files record what the gate deliberately does **not** run. They are the
gate's memory of the pre-existing state of `main` at the moment the gate was
introduced.

### `ci/known_failures.txt` — deselected node IDs

Individual pytest node IDs passed to `--deselect`. As of the baseline freeze
(run `28567152991`, commit `cc18506e`) this holds **47** node IDs, grouped by
cause with whole-line `#` comments. The cause buckets are:

| Cause | Meaning |
| --- | --- |
| needs CUDA | test hard-requires a GPU |
| behavioral, pre-existing | fails on `main` identically in a local GPU env; not a CI artifact |
| `InferenceConfig` lacks `log_patch_stats` / `patch_stats_limit` | test references config fields not present on `main` |
| config dataclass lacks a field the test passes (e.g. `torch_loss_mode`) | test constructs a config with a kwarg `main` doesn't define |
| generator-adapter API absent from `model.py` | test imports symbols not on `main` |
| `helper.derive_intensity_scale_from_amplitudes` absent | missing symbol |
| `helper.normalize_probe_like_tf` absent | missing symbol |
| `model._build_optimizer` absent | missing symbol |

These are **not** CI defects. They are real gaps between the test suite and the
implementation on `main` — several of them are already fixed on the
`varpro-ablation` / `fno-stable` line of work and will be removable once that
work lands (§6, removing entries).

### `ci/collect_ignores.txt` — ignored files

Whole file paths passed to `--ignore`. As of the baseline this holds **3**
files. These are separate from `known_failures.txt` because they fail at
**collection / import time**, not at test-execution time: the module raises
`ImportError` when pytest tries to import it, so `--deselect` (which needs the
node to be collectable) cannot suppress them. `--ignore` skips the file before
import.

The three entries import symbols absent from `main`
(`adaptive_gradient_clip_`, `compute_grad_norm`, `normalize_probe_like_tf`).

### File format rules

Both files are parsed by a `while read` loop in `run_ci_tests.sh`:

- **one entry per line**;
- blank lines and whole-line `#` comments are skipped;
- **no inline comments** — a trailing `# reason` on the same line as a node ID
  becomes part of the `--deselect` argument and silently corrupts it (the
  entry then matches nothing, and the test it was meant to exclude runs
  unexpectedly). Put reasons on their own comment line above the entry.
- Keep a trailing newline. A missing final newline drops the last line in the
  classic `while read` EOF gotcha.

---

## 5. The ratchet policy

**The exclusion baseline may only shrink.**

- **Removing an entry** — after fixing the test or supplying its environment
  need — is always welcome and needs no special justification beyond the PR
  that does it. Each removal must, of course, leave the gate green.
- **Adding an entry** requires a PR whose diff includes a comment line stating
  *why*, and is reserved for tests that genuinely **cannot** pass in CI: a new
  CUDA requirement, or a new dependency on untracked data. It is **never** a
  legitimate tool for silencing a regression in code that should pass. Expect
  an addition to be challenged in review, and expect the reviewer to ask
  "why can't this test pass on a CPU runner with only tracked files?"

The ratchet is what keeps the gate meaningful over time: without it, the
baseline becomes a dumping ground and the green check stops meaning "the suite
passes."

---

## 6. Working with the gate day to day

### Reproduce the CI run locally

```bash
bash ci/run_ci_tests.sh
```

Run it from the repo root. This is the *exact* command CI runs. Useful
variations (extra args pass through to pytest):

```bash
bash ci/run_ci_tests.sh -k test_name        # run a subset
bash ci/run_ci_tests.sh --collect-only      # see what would run
bash ci/run_ci_tests.sh -x                  # stop at first failure
```

> A green **local conda run does not guarantee green CI.** Your local
> environment has a GPU and local datasets that mask failures the CPU /
> tracked-files-only runner will hit. **The CI environment is authoritative** —
> when local and CI disagree, CI is right about what merges.

### When your PR is red

1. Open the `pytest-cpu` check's log and read the failing node IDs.
2. **If your change broke them:** fix your change. This is the gate doing its
   job.
3. **If the failure is environmental** — the test newly requires CUDA or an
   untracked data file — either move that requirement behind a skip marker
   (`@pytest.mark.slow`, a CUDA-availability skip, etc.) or add a baseline
   entry with justification per the ratchet policy. Adding a baseline entry to
   make a *code regression* go away is not acceptable.
4. **If your PR went stale** (someone merged to `main` after your last green
   run), push your branch again or rebase — the strict policy requires a fresh
   run against the new tip.

### Removing a baseline entry (the happy path)

When a fix lands that makes an excluded test pass on a CPU runner:

1. Delete its line from `ci/known_failures.txt` (or `ci/collect_ignores.txt`).
2. If a whole cause bucket empties, delete the now-orphaned `#` comment too.
3. Run `bash ci/run_ci_tests.sh` locally to confirm the newly included test
   passes and nothing else broke.
4. Open a PR. The removal is self-justifying.

---

## 7. History: the conftest fix

The baseline mechanism depends on `--ignore` working, and it initially did not.
`tests/conftest.py`'s `pytest_ignore_collect` hook returned `False` from its
"no opinion" branches. Because `pytest_ignore_collect` is a `firstresult` hook,
a non-`None` return **short-circuits the hook chain** — including pytest's own
built-in handler that implements `--ignore` / `--ignore-glob`. The result was
that `--ignore` was silently disabled for the entire suite, repo-wide, for
everyone (not just CI).

The fix (commit `cc18506e`) changes those branches to `return None`, which lets
the built-in handler run. This was ratified as an amendment to the CI plan
because it touches a test file rather than a top-level CI file, but it is a
genuine repo-wide bug fix, not a CI-only workaround. Verified to change
collection behavior for **zero** existing invocations (no `test_*_backup.py`
files exist; `.ipynb` files never matched the `python_files` pattern).

---

## 8. Known limitations and non-goals

### Semantic-conflict window

Two PRs that are each green in isolation can still break `main` together (a
semantic conflict neither branch's tests catch). The strict up-to-date flag
forces the *second* PR to re-test after the first merges, which closes this for
**serial** merges. If merge volume ever makes that racy, enable a **merge
queue** on the ruleset.

### v1 non-goals (documented so they are not scope-crept in silently)

- **TF-side suites** (`tests/` top level, `tests/study`, `tests/scripts`,
  `tests/io`): heavier data/env coupling. Add later by widening
  `run_ci_tests.sh` scope and re-running the baseline-discovery loop.
- **GPU coverage:** requires a self-hosted runner. The local machine remains
  the GPU test environment.
- **Merge queue:** unnecessary for a single-maintainer repo; the strict
  up-to-date flag suffices for serial merges.
- **Fixing the pre-existing baseline failures:** each fix is separate follow-up
  work and deletes its own baseline line (the ratchet).

---

## 9. Maintenance reference

### Files

| File | Purpose |
| --- | --- |
| `.github/workflows/tests.yml` | the `pytest-cpu` job (triggers, env, install, run) |
| `ci/run_ci_tests.sh` | the runner: assembles the pytest command from the baseline files |
| `ci/known_failures.txt` | deselected node IDs (execution-time failures) |
| `ci/collect_ignores.txt` | ignored files (collection-time import errors) |
| `requirements-ci.txt` | CPU-only dependency set with the two load-bearing pins |
| `ci/main-ruleset.json` | the `main-test-gate` ruleset definition (provenance record) |
| `docs/ci.md` | this document — policy of record |

### Re-applying or updating the ruleset

The ruleset was applied with:

```bash
gh api -X POST repos/hoidn/PtychoPINN/rulesets \
  -H "Accept: application/vnd.github+json" \
  --input ci/main-ruleset.json
```

To inspect or update the live ruleset:

```bash
gh api repos/hoidn/PtychoPINN/rulesets                       # list (find the id)
gh api repos/hoidn/PtychoPINN/rulesets/<id>                  # show one
gh api -X PUT repos/hoidn/PtychoPINN/rulesets/<id> \
  -H "Accept: application/vnd.github+json" --input ci/main-ruleset.json
```

Keep `ci/main-ruleset.json` in sync with the live ruleset so the committed file
stays an accurate provenance record.

### Verifying the gate still bites

Two probes confirm enforcement (run against a throwaway branch, then clean up):

```bash
# 1. Direct push must be rejected
git commit --allow-empty -m "ruleset probe (must be rejected)"
git push origin HEAD:main            # expect GH013 rules-violation rejection
git reset --hard origin/main

# 2. A red PR must be unmergeable
#    push a branch with a deliberately failing test, open a PR, wait for the
#    run to fail, then:
gh pr view <n> --json mergeStateStatus --jq .mergeStateStatus   # expect BLOCKED
gh pr close <n> --delete-branch
```
