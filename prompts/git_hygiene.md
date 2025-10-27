# Git Hygiene Guidelines (Reusable Across Repos)

Purpose: Reduce incidental Git friction during automation loops (supervisor/loop), keep history clean, and prevent failures due to dirty working trees or submodule drift.

Core Principles
- Keep working trees clean between automated turns
- Ignore generated artifacts and large data
- Treat automation/tooling submodules differently from code submodules

Recommended Global Config
- pull.rebase=true — favor linear history for small handoffs
- rebase.autoStash=true — stash local changes before rebase, restore after
- fetch.prune=true — remove deleted remote refs on fetch
- rerere.enabled=true — reuse recorded resolutions
- rebase.autosquash=true — tidy fixup!/squash! commits (optional)

Submodule Policy
- Tooling/agent submodules (e.g., .claude or claude): set ignore = dirty
  - Rationale: these may change locally at runtime and should not block pulls
- Library/code submodules: keep default (do not ignore dirty)
- Keep submodule URLs suitable for CI (HTTPS preferred where SSH is unavailable)

Ignore Lists (.gitignore)
- Temporary/logs: tmp/, *.log, *.egg-info/
- Outputs/checkpoints: outputs/, results/, checkpoints/, saved_models/
- Large data: datasets/, data/, *.npz, *.npy, *.h5, *.hdf5, *.pkl, *.pickle, *.dill
- Add project-specific transient files as discovered

Safe Pull Pattern
- Shell wrapper to avoid rebase stalls and untracked-file collisions:
```
git_safe_pull() {
  if ! timeout 30 git pull --rebase; then
    echo "WARNING: git pull --rebase failed or timed out. Attempting recovery..."
    git rebase --abort || true
    git pull --no-rebase || true
  fi
}
```
- Python alternative exists in scripts/orchestration/git_bus.py::safe_pull()

Recovery Playbook
- Untracked-file overwrite on pull: move/rename, or `git stash -u`, or `git clean -fd` if safe
- Stuck rebase: `git rebase --abort` then run safe_pull
- Submodule mismatch: `git submodule update --init` and re-sync `.gitmodules`

CI/Headless Guidance
- Prefer HTTPS submodule URLs; ensure tokens/credentials are configured
- Batch sync/push where possible; avoid per-poll push cycles

Data Management
- Never commit large datasets or model artifacts directly; keep them ignored and documented
- See docs/DATA_MANAGEMENT_GUIDE.md for repository policy and examples

Verification Checklist (Before Running Automation)
- `git status` is clean
- `git fetch --all --prune` ok; `git_safe_pull` succeeds
- `.gitmodules` uses `ignore = dirty` for tooling submodules only
- `.gitignore` includes transient and data patterns

