#!/usr/bin/env bash
set -euo pipefail

# Bootstrap Git settings for supervisor/loop/orchestrator repos.
# - Initializes/syncs submodules
# - Sets ignore=dirty for tooling submodules (.claude/claude) if present
# - Applies repo-local git configs for automation-friendly pulls
# - Optionally hardens .gitignore with transient/data patterns
#
# Usage:
#   scripts/bootstrap_git.sh [--force] [--update-gitignore] [--global]
#
# Flags:
#   --force             Proceed even if working tree is dirty (default: require clean)
#   --update-gitignore  Append recommended ignore patterns if missing
#   --global            Also set recommended configs in global scope

FORCE=0
UPDATE_GITIGNORE=0
SET_GLOBAL=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force) FORCE=1; shift ;;
    --update-gitignore) UPDATE_GITIGNORE=1; shift ;;
    --global) SET_GLOBAL=1; shift ;;
    *) echo "Unknown flag: $1" >&2; exit 2 ;;
  esac
done

echo "[bootstrap] Current branch: $(git rev-parse --abbrev-ref HEAD)"
echo "[bootstrap] Remote(s):"; git remote -v | sed 's/^/  /'

# Require clean working tree unless forced
if [[ $FORCE -ne 1 ]]; then
  if ! git diff --quiet || ! git diff --cached --quiet; then
    echo "ERROR: Working tree is not clean. Commit/stash changes or re-run with --force." >&2
    exit 2
  fi
fi

# Initialize and sync submodules
echo "[bootstrap] Initializing submodules..."
git submodule update --init
git submodule sync --recursive

# Configure ignore=dirty for tooling submodules if present
did_edit_gitmodules=0
if git config -f .gitmodules --get-regexp '^submodule\..*\.path$' >/dev/null 2>&1; then
  while IFS=$' \t' read -r key path; do
    # key example: submodule.".claude".path or submodule.claude.path
    name=${key%.*}          # submodule.".claude"
    name=${name#submodule.} # ".claude"
    # Normalize quotes
    unquoted=${name%"}
    unquoted=${unquoted#"}
    if [[ "$unquoted" == ".claude" || "$unquoted" == "claude" ]]; then
      echo "[bootstrap] Marking tooling submodule '$unquoted' as ignore=dirty"
      git config -f .gitmodules submodule."$unquoted".ignore dirty || true
      did_edit_gitmodules=1
    fi
  done < <(git config -f .gitmodules --get-regexp '^submodule\..*\.path$')
fi

if [[ $did_edit_gitmodules -eq 1 ]]; then
  git submodule sync
  if ! git diff --quiet -- .gitmodules; then
    git add .gitmodules
    git commit -m "submodule hygiene: ignore dirty for tooling (.claude/claude)" || true
  fi
fi

# Apply repo-local automation-friendly configs
echo "[bootstrap] Applying repo-local git configs"
git config pull.rebase true
git config rebase.autoStash true
git config fetch.prune true
git config rerere.enabled true

# Optionally apply global configs
if [[ $SET_GLOBAL -eq 1 ]]; then
  echo "[bootstrap] Applying global git configs"
  git config --global pull.rebase true
  git config --global rebase.autoStash true
  git config --global fetch.prune true
  git config --global rerere.enabled true
fi

# Optionally harden .gitignore
if [[ $UPDATE_GITIGNORE -eq 1 ]]; then
  echo "[bootstrap] Updating .gitignore with recommended patterns"
  touch .gitignore
  declare -a PATTERNS=(
    "tmp/" "*.log" "*.egg-info/" "outputs/" "results/" "checkpoints/" "saved_models/"
    "datasets/" "data/" "*.npz" "*.npy" "*.h5" "*.hdf5" "*.pkl" "*.pickle" "*.dill"
  )
  updated=0
  for p in "${PATTERNS[@]}"; do
    if ! grep -qxF "$p" .gitignore; then
      echo "$p" >> .gitignore
      updated=1
    fi
  done
  if [[ $updated -eq 1 ]]; then
    git add .gitignore
    git commit -m "git hygiene: ignore transient/data artifacts" || true
  fi
fi

# Final verification
echo "[bootstrap] Verifying pull with rebase (with fallback)"
if ! timeout 30 git pull --rebase; then
  echo "[bootstrap] WARNING: git pull --rebase failed; attempting recovery"
  git rebase --abort || true
  git pull --no-rebase || true
fi

echo "[bootstrap] Submodule status:"; git submodule status || true
echo "[bootstrap] Completed."

