# Git Setup For Supervisor/Loop Orchestration (Agent Runbook)

Purpose: Make Git behavior deterministic for supervisor/loop/orchestrator flows. Prevent dirty submodule noise, rebase/pull stalls, and untracked-file collisions. Follow steps in order; all commands are idempotent.

Prereqs
- Git installed and repository cloned
- Remote auth works (SSH or HTTPS)

Step 1 — Detect Environment
1) Print versions and branch
   - git --version
   - git rev-parse --abbrev-ref HEAD
   - git remote -v
2) Verify working tree is clean (otherwise stash)
   - git status --porcelain
   - If not empty: git stash push -u -m "pre-orchestrator"

Step 2 — Global Defaults (safe for automation)
Run once per machine. Re-run is safe.
- git config --global pull.rebase true
- git config --global rebase.autoStash true
- git config --global fetch.prune true
- git config --global rerere.enabled true
- Optional: git config --global rebase.autosquash true
- Optional (CI prefers HTTPS): git config --global url."https://github.com/".insteadOf git@github.com:
Verify:
- git config -l --global | rg -n "(pull.rebase|rebase.autoStash|fetch.prune|rerere.enabled)"

Step 3 — Submodules (initialize, sync, and ignore dirty for tooling)
1) Initialize/sync
   - git submodule update --init
   - git submodule sync --recursive
2) Mark tooling submodules as ignore-dirty to avoid false dirty states
   This is commonly required for ".claude" or "claude" automation helpers.
   Prefer git config edits; if they are awkward due to dotted names, you may edit .gitmodules directly.
   - If .claude exists:
     - EITHER: git config -f .gitmodules "submodule..claude.ignore" dirty
     - OR edit .gitmodules and add this line under the [submodule ".claude"] block:
       - ignore = dirty
   - If claude exists:
     - git config -f .gitmodules submodule.claude.ignore dirty
   - For other tooling-only submodules that may change at runtime, repeat the pattern.
3) Sync and persist
   - git submodule sync
   - git add .gitmodules && git commit -m "submodule hygiene: ignore dirty for tooling" || true
4) Verify
   - git config -f .gitmodules --get-regexp ^submodule\..*\.ignore || true
   - git submodule status

Notes
- In this repository, the fix that stabilized automation was adding ignore = dirty to the .claude submodule (example: .gitmodules entry with ignore = dirty).

Step 4 — Ignore Transient/Generated Files (keep the tree clean)
Add these patterns if missing (append-only). Do not remove repo-specific ignores.
- __pycache__/
- *.pyc
- tmp/
- *.log
- *.egg-info/
- outputs/
- results/
- checkpoints/
- saved_models/
- datasets/
- data/
- *.npz
- *.npy
- *.h5
- *.hdf5
- *.pkl
- *.pickle
- *.dill
Agent action
- If .gitignore exists, ensure lines above are present (add if missing)
- If new lines added: git add .gitignore && git commit -m "git hygiene: ignore transient/data artifacts" || true

Step 5 — Orchestrator Safety (pull/rebase guard)
Shell orchestrators must define a safe pull wrapper to avoid rebase stalls.
Check for the pattern; if missing, add it to both scripts.
Pattern to search:
- rg -n "git pull --rebase" supervisor.sh loop.sh || true
Canonical function (use exactly):
```
git_safe_pull() {
  if ! timeout 30 git pull --rebase; then
    echo "WARNING: git pull --rebase failed or timed out. Attempting recovery..."
    git rebase --abort || true
    git pull --no-rebase || true
  fi
}
```
Python alternative already available (prefer if using Python orchestrators):
- scripts/orchestration/git_bus.py: safe_pull()

Step 6 — Handoff State Discipline
- Track and commit only the turn-taking state file between actors (e.g., sync/state.json)
- Keep logs and temp artifacts ignored (tmp/supervisorlog*, tmp/claudelog*)
- Confirm no other files are staged before push
  - git diff --cached --name-only

Step 7 — Sanity Cycle (dry run)
1) Fetch and reconcile
   - git fetch --all --prune
   - Use orchestrator’s git_safe_pull or: git pull --rebase || (git rebase --abort && git pull --no-rebase)
2) Submodules healthy
   - git submodule status
3) Working tree clean
   - git status --porcelain (should be empty)

Step 8 — Recovery Playbook
- Untracked file would be overwritten by merge/checkout/pull
  - Move/rename offending files OR: git stash push -u -m "untracked-collision"; retry pull
  - If disposable: git clean -fd (destructive)
- Rebase stuck/in-progress
  - git rebase --abort; then safe pull
- Submodule mismatch/dirty when not expected
  - git submodule update --init
  - Ensure .gitmodules has ignore = dirty for tooling submodules

Step 9 — CI/Headless Specifics
- Prefer HTTPS submodule URLs in CI runners
- Batch sync where possible (avoid pull/push on every poll); still use safe_pull wrapper when reconciling

Optional — One‑Shot Bootstrap Macro (run from repo root)
```
git submodule update --init && \
git submodule sync --recursive && \
{ git config -f .gitmodules "submodule..claude.ignore" dirty 2>/dev/null || true; } && \
{ git config -f .gitmodules submodule.claude.ignore dirty 2>/dev/null || true; } && \
git submodule sync && \
git add .gitmodules && git commit -m "submodule hygiene: ignore dirty for .claude/claude" || true
```

Success Criteria
- safe_pull available in shell/Python orchestrators
- .claude/claude submodules, if present, have ignore = dirty in .gitmodules
- .gitignore blocks logs, eggs, outputs, and large data artifacts
- git status clean; pull/rebase cycle succeeds without manual intervention
