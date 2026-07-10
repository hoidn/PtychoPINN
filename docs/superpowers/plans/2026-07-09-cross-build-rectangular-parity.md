# Cross-Build Rectangular Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace non-portable bit-exact frozen-float assertions with a bounded cross-build parity contract that still rejects material physics drift.

**Architecture:** Keep the change in the cross-branch verification test and its documentation. A private assertion helper owns the established `rtol=1e-5`, `atol=1e-6` contract for forward tensors and loss scalars; fixture-backed boundary tests prove the helper accepts backend roundoff and rejects material perturbations.

**Tech Stack:** Python 3.11, PyTorch, pytest, GitHub Actions CPU gate.

---

### Task 0: Commit Approved Planning Artifacts

- [ ] **Step 1:** Commit the approved spec amendments and this implementation
  plan before implementation starts:

```bash
git add \
  docs/superpowers/specs/2026-07-09-cross-build-rectangular-parity-design.md \
  docs/superpowers/plans/2026-07-09-cross-build-rectangular-parity.md
git commit -m "docs: plan portable rectangular parity checks"
```

### Task 1: Portable Frozen-Fixture Contract

**Files:**
- Modify: `tests/torch/test_cross_branch_rectangular_parity.py`
- Modify: `docs/development/TEST_SUITE_INDEX.md`
- Modify: `docs/findings.md`
- Modify: `docs/plans/2026-07-07-rebase-fno-stable-onto-main.md`

- [ ] **Step 1: Write the failing tolerance-boundary regression**

Add `test_cross_build_tolerance_accepts_roundoff_and_rejects_material_drift`.
Load `c1_bigF`, calculate the real forward tensor, Poisson loss, and MAE loss,
then call a not-yet-defined `_assert_frozen_close` helper.

For both a representative forward value and each scalar loss, construct:

```python
bound = FROZEN_ATOL + FROZEN_RTOL * expected.abs()
within = expected + 0.5 * bound
outside = expected + 100.0 * bound
```

Assert the helper accepts `within` and raises `AssertionError` for `outside`.
Keep a direct assertion that the real unperturbed forward/loss results satisfy
the helper so the test remains fixture-backed.

- [ ] **Step 2: Run the regression and verify RED**

Run:

```bash
python -m pytest \
  tests/torch/test_cross_branch_rectangular_parity.py::test_cross_build_tolerance_accepts_roundoff_and_rejects_material_drift \
  -vv
```

Expected: collection or execution fails because `_assert_frozen_close` and the
named tolerance constants do not exist.

- [ ] **Step 3: Implement the shared cross-build assertion**

Add:

```python
FROZEN_RTOL = 1e-5
FROZEN_ATOL = 1e-6


def _assert_frozen_close(actual, expected):
    torch.testing.assert_close(
        actual, expected, rtol=FROZEN_RTOL, atol=FROZEN_ATOL)
```

Replace the two `torch.equal` forward assertions and all zero-tolerance frozen
loss assertions in this module with `_assert_frozen_close`. Both forward checks,
including matched-padding replay, compare against cross-build fixtures and use
the same portable contract. Do not alter mode forcing, stored weighting,
padding monkeypatching, shape checks, or finite residual checks.

- [ ] **Step 4: Correct stale contract wording**

Update the module docstring, test names/messages where needed,
`docs/findings.md`, and the existing CI-flake incident note. State that
cross-build fixture replay, including the matched-padding comparison, is
numerically equivalent at `1e-5/1e-6`; reserve bit-exact claims for same-build
or structural comparisons that do not use frozen floating-point outputs.

Regenerate `docs/development/TEST_SUITE_INDEX.md` with:

```bash
python scripts/tools/generate_test_index.py \
  > docs/development/TEST_SUITE_INDEX.md
```

Inspect the diff and retain only generator-consistent changes attributable to
the updated test module.

- [ ] **Step 5: Run targeted GREEN verification**

Run in both environments:

```bash
python -m pytest tests/torch/test_cross_branch_rectangular_parity.py -q
CUDA_VISIBLE_DEVICES='' /tmp/ptycho-torch213/bin/python -m pytest \
  tests/torch/test_cross_branch_rectangular_parity.py -q
```

Expected: all tests pass in both the repository and CI-matched Torch 2.13 CPU
environments.

- [ ] **Step 6: Run adjacent and exact CI verification**

```bash
python -m pytest \
  tests/torch/test_cross_branch_rectangular_parity.py \
  tests/torch/test_rectangular_scaled_forward.py -q
bash ci/run_ci_tests.sh
PATH=/tmp/ptycho-torch213/bin:$PATH CUDA_VISIBLE_DEVICES='' \
  bash ci/run_ci_tests.sh
git diff --check
```

Expected: zero failures.

- [ ] **Step 7: Commit**

```bash
git add tests/torch/test_cross_branch_rectangular_parity.py \
  docs/development/TEST_SUITE_INDEX.md \
  docs/findings.md \
  docs/plans/2026-07-07-rebase-fno-stable-onto-main.md
git commit -m "test(torch): make frozen parity checks cross-build portable"
```

### Task 2: Main Integration And CI

- [ ] **Step 1:** Review Task 1 for spec compliance and code quality.
- [ ] **Step 2:** Create a disposable clone from `origin/main` because repository
  policy forbids worktrees and the shared `fno-stable` checkout is dirty.
- [ ] **Step 3:** Create a feature branch in the clone and cherry-pick the
  approved planning-artifact commit and implementation commit.
- [ ] **Step 4:** Run `bash ci/run_ci_tests.sh` and the targeted Torch 2.13 CPU
  parity module on the feature branch.
- [ ] **Step 5:** Push the feature branch, open a PR, and monitor the required
  `pytest-cpu` check. Do not merge while it is failing, pending, or stale.
- [ ] **Step 6:** Merge the green PR, verify the resulting `main` push run, then
  add the internal remote, fetch the merged public tip, verify that fetched SHA
  equals the PR merge commit, and fast-forward the internal mirror:

```bash
git remote add internal git@github.com:hoidn/PtychoPINN-internal.git
git fetch origin main
test "$(git rev-parse origin/main)" = "<merged-main-sha>"
git push internal origin/main:refs/heads/main
```
- [ ] **Step 7:** Report the PR, CI run URLs, conclusions, and final public and
  internal `main` hashes.
