# Hybrid ResNet Regression Bisect Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Identify the first commit on `fno2-phase8-optimizers` (after the LCA with `fno-stable`) that causes `tests/torch/test_grid_lines_hybrid_resnet_integration.py` to fail.

**Architecture:** Manual bisect by repeatedly merging a target commit from `fno2-phase8-optimizers` into a throwaway branch based on `fno-stable`, running the integration test, and narrowing the commit range.

**Tech Stack:** git, pytest (via `python -m pytest`), bash.

---

### Task 1: Establish Good/Bad Baselines

**Files:**
- Modify: none
- Test: `tests/torch/test_grid_lines_hybrid_resnet_integration.py`

**Step 1: Create log dir**

Run:
```bash
mkdir -p .artifacts/bisect
```

**Step 2: Run test on known-good branch (`fno-stable`)**

Run:
```bash
git checkout fno-stable
python -m pytest -v -m integration tests/torch/test_grid_lines_hybrid_resnet_integration.py \
  | tee .artifacts/bisect/baseline_fno-stable.log
```
Expected: PASS. If FAIL, record failure and stop (no known good).

**Step 3: Run test on known-bad branch (`fno2-phase8-optimizers`)**

Run:
```bash
git checkout fno2-phase8-optimizers
python -m pytest -v -m integration tests/torch/test_grid_lines_hybrid_resnet_integration.py \
  | tee .artifacts/bisect/baseline_fno2-phase8-optimizers.log
```
Expected: FAIL. If PASS, there’s no regression to bisect.

**Step 4: Return to `fno-stable` for bisect start**

Run:
```bash
git checkout fno-stable
```

---

### Task 2: Build the Commit Range for Bisect

**Files:**
- Create: `.artifacts/bisect/commit_list.txt`
- Modify: none

**Step 1: Compute latest common ancestor (LCA)**

Run:
```bash
LCA=$(git merge-base fno-stable fno2-phase8-optimizers)
echo "$LCA" | tee .artifacts/bisect/lca.txt
```

**Step 2: List commits from LCA..fno2-phase8-optimizers (oldest→newest)**

Run:
```bash
git rev-list --reverse "${LCA}..fno2-phase8-optimizers" \
  | tee .artifacts/bisect/commit_list.txt
```

**Step 3: Record the total count**

Run:
```bash
wc -l .artifacts/bisect/commit_list.txt \
  | tee .artifacts/bisect/commit_count.txt
```

---

### Task 3: Manual Bisect Loop (Repeat Until Single Commit)

**Files:**
- Modify: `.artifacts/bisect/bisect_log.csv`

**Step 1: Initialize a bisect log**

Run:
```bash
echo "sha,status,notes" > .artifacts/bisect/bisect_log.csv
```

**Step 2: Pick mid commit from current range**

Example (replace START/END line numbers):
```bash
MID=$(( (START + END) / 2 ))
SHA=$(sed -n "${MID}p" .artifacts/bisect/commit_list.txt)
```

**Step 3: Create throwaway branch from `fno-stable` and merge target commit**

Run:
```bash
git checkout fno-stable
git branch -D bisect-temp 2>/dev/null || true
git checkout -b bisect-temp

git merge --no-ff "$SHA"
```

If merge conflicts, abort and record:
```bash
git merge --abort
printf "%s,conflict,merge failed\n" "$SHA" >> .artifacts/bisect/bisect_log.csv
```
Then adjust range manually (treat as bad unless proven otherwise).

**Step 4: Run the integration test**

Run:
```bash
python -m pytest -v -m integration tests/torch/test_grid_lines_hybrid_resnet_integration.py \
  | tee ".artifacts/bisect/test_${SHA}.log"
```

**Step 5: Record PASS/FAIL and update range**

Example:
```bash
printf "%s,pass,\n" "$SHA" >> .artifacts/bisect/bisect_log.csv
# or
printf "%s,fail,\n" "$SHA" >> .artifacts/bisect/bisect_log.csv
```

Update START/END indices accordingly and repeat Steps 2–5 until the range narrows to a single commit.

**Step 6: Clean up branch**

Run:
```bash
git checkout fno-stable
git branch -D bisect-temp
```

---

### Task 4: Verify First Bad Commit

**Files:**
- Modify: `.artifacts/bisect/bisect_log.csv`

**Step 1: Test parent of suspected commit (last good)**

Repeat Task 3 steps using the parent commit hash. Expect PASS.

**Step 2: Test suspected commit (first bad)**

Repeat Task 3 steps using the suspected commit hash. Expect FAIL.

---

### Task 5: Summarize Root Cause Candidate

**Files:**
- Create: `.artifacts/bisect/summary.md`

**Step 1: Capture diff summary**

Run:
```bash
git show --stat <FIRST_BAD_SHA> > .artifacts/bisect/summary.md
```

**Step 2: Add a short human summary**

Append:
```bash
cat <<'NOTE' >> .artifacts/bisect/summary.md

Notes:
- First bad commit: <FIRST_BAD_SHA>
- Last good commit: <LAST_GOOD_SHA>
- Test: tests/torch/test_grid_lines_hybrid_resnet_integration.py
NOTE
```

---

Plan complete and saved to `docs/plans/2026-02-05-hybrid-resnet-bisect.md`. Two execution options:

1. Subagent-Driven (this session) — I dispatch fresh subagent per task, review between tasks, fast iteration
2. Parallel Session (separate) — Open new session with executing-plans, batch execution with checkpoints

Which approach?
