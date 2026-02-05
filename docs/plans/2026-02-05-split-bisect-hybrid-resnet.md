# Hybrid ResNet Regression Split-Bisect Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Split commits `3ed15fa5611acc1fa932554520d761d0bfe25c31` and its child (`407d2bbb59ec2c9c51cbf38d25f4d67dfc6afebe`) into smaller logical commits, then continue the bisect on the finer-grained history to pinpoint the regression source.

**Architecture:** Rewrite history on a new local branch based on `fno2-phase8-optimizers`, without worktrees. Use interactive rebase with `edit` to split commits, then rerun the integration bisect flow against the new commit list using the existing integration test.

**Tech Stack:** git, pytest (`python -m pytest`), bash.

---

### Task 1: Inspect Target Commits and Define Split Boundaries

**Files:**
- Modify: none
- Inspect: git history only

**Step 1: Capture diff stats for both commits**

Run:
```bash
git show --stat 3ed15fa5611acc1fa932554520d761d0bfe25c31
```
```bash
git show --stat 407d2bbb59ec2c9c51cbf38d25f4d67dfc6afebe
```

**Step 2: Decide split groups**

Define logical groups (example: `ptycho_torch/inference+workflows`, `grid_lines_torch_runner`, `tests/torch`, `docs/plans`). These groupings will become separate commits during the split.

---

### Task 2: Create a Split Branch (No Worktrees)

**Files:**
- Modify: none

**Step 1: Create branch from `fno2-phase8-optimizers`**

Run:
```bash
git checkout fno2-phase8-optimizers
git checkout -b split-bisect-hybrid-resnet
```

---

### Task 3: Split Commit `3ed15fa...`

**Files:**
- Modify: files touched by `3ed15fa...`

**Step 1: Interactive rebase to edit the target commit**

Run:
```bash
git rebase -i e695f2dc8720952816a5a0f692ca3c8b1887d20e
```

In the editor, mark `3ed15fa...` as `edit` and keep later commits as `pick` for now.

**Step 2: Uncommit and split**

Run:
```bash
git reset HEAD^
```

**Step 3: Create grouped commits**

For each logical group:
```bash
git add -p <files>
git commit -m "split: <group description>"
```
Repeat until all changes are committed.

**Step 4: Continue rebase**

Run:
```bash
git rebase --continue
```

---

### Task 4: Split Commit `407d2bbb...`

**Files:**
- Modify: files touched by `407d2bbb...`

**Step 1: Rebase will stop at `407d2bbb...` if marked edit**

If not already, re-run:
```bash
git rebase -i <parent-of-407d2bbb>
```
Mark `407d2bbb...` as `edit`.

**Step 2: Uncommit and split**

Run:
```bash
git reset HEAD^
```

**Step 3: Create grouped commits**

For each logical group:
```bash
git add -p <files>
git commit -m "split: <group description>"
```

**Step 4: Continue rebase**

Run:
```bash
git rebase --continue
```

---

### Task 5: Rebuild Bisect Range and Re-run Bisect on Split History

**Files:**
- Create: `.artifacts/bisect_split/commit_list.txt`
- Modify: `.artifacts/bisect_split/bisect_log.csv`

**Step 1: Create bisect artifacts dir**

Run:
```bash
mkdir -p .artifacts/bisect_split
```

**Step 2: Build commit list from last good to HEAD**

Run:
```bash
LCA=$(git merge-base fno-stable split-bisect-hybrid-resnet)

git rev-list --reverse "${LCA}..split-bisect-hybrid-resnet" \
  | tee .artifacts/bisect_split/commit_list.txt
```

**Step 3: Initialize bisect log**

Run:
```bash
echo "sha,status,notes" > .artifacts/bisect_split/bisect_log.csv
```

**Step 4: Manual bisect loop**

Use the same manual merge-and-test method (no worktrees):
```bash
git checkout fno-stable
git branch -D bisect-temp 2>/dev/null || true
git checkout -b bisect-temp
git merge --no-ff <SHA>
python -m pytest -v -m integration tests/torch/test_grid_lines_hybrid_resnet_integration.py \
  | tee .artifacts/bisect_split/test_<SHA>.log
```
Record PASS/FAIL in `bisect_log.csv`, update the range, repeat until the first bad split commit is found.

---

### Task 6: Summarize Results

**Files:**
- Create: `.artifacts/bisect_split/summary.md`

**Step 1: Capture diff summary for first bad split commit**

Run:
```bash
git show --stat <FIRST_BAD_SPLIT_SHA> > .artifacts/bisect_split/summary.md
```

**Step 2: Add a short human summary**

Append the failure signature, PASS/FAIL evidence, and the narrowed commit identifier.

---

**Execution Options**

Plan complete and saved to `docs/plans/2026-02-05-split-bisect-hybrid-resnet.md`.

Two execution options:

1. **Subagent-Driven (this session)** — I dispatch a fresh subagent per task, review between tasks.
2. **Parallel Session (separate)** — open a new session and use `superpowers:executing-plans` with checkpoints.

Which approach?
