#!/usr/bin/env bash
set -euo pipefail

# Prefer Python implementation; keep legacy below for fallback
if [[ "${ORCHESTRATION_PYTHON:-1}" == "1" ]]; then
  # --- Activate the conda environment required for Claude automation ---
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate pytorch
  # --------------------------------------------------------------------
  # Resolve interpreter to the active environment unless overridden
  if [[ -z "${PYTHON_BIN:-}" ]]; then
    if command -v python >/dev/null 2>&1; then
      PYTHON_BIN="$(python - <<'PY'
import sys
print(sys.executable)
PY
)"
    elif command -v python3 >/dev/null 2>&1; then
      PYTHON_BIN="$(python3 - <<'PY'
import sys
print(sys.executable)
PY
)"
    else
      PYTHON_BIN=python3
    fi
  fi
  exec "$PYTHON_BIN" -m scripts.orchestration.loop "$@"
fi

# --- Args & defaults ---
SYNC_VIA_GIT=0
POLL_INTERVAL=${POLL_INTERVAL:-5}
MAX_WAIT_SEC=${MAX_WAIT_SEC:-0}
STATE_FILE="sync/state.json"
SYNC_LOOPS=${SYNC_LOOPS:-20}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sync-via-git)
      SYNC_VIA_GIT=1; shift ;;
    --sync-loops)
      SYNC_LOOPS="$2"; shift 2 ;;
    --poll-interval)
      POLL_INTERVAL="$2"; shift 2 ;;
    --max-wait-sec)
      MAX_WAIT_SEC="$2"; shift 2 ;;
    *)
      break ;;
  esac
done

# --- Activate the conda environment required for Claude automation ---
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate pytorch
# --------------------------------------------------------------------

mkdir -p tmp
TS=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="tmp/claudelog${TS}.txt"
ln -sf "${LOG_FILE}" tmp/claudelog-latest.txt

CLAUDE_CMD="/home/ollie/.claude/local/claude"

git_safe_pull() {
  if ! timeout 30 git pull --rebase; then
    echo "WARNING: git pull --rebase failed or timed out. Attempting recovery..." | tee -a "$LOG_FILE"
    git rebase --abort || true
    git pull --no-rebase || true
  fi
}

read_state() {
  python - <<'PY'
import json,sys
path = sys.argv[1]
try:
  with open(path,'r') as f:
    st=json.load(f)
except Exception:
  st={}
print(st.get('expected_actor',''))
print(st.get('status',''))
print(st.get('iteration',1))
print(st.get('lease_expires_at',''))
PY
}

write_state() {
  # args: expected_actor status increment ralph_commit(optional)
  local expected_actor="$1"; shift
  local status="$1"; shift
  local increment="$1"; shift
  local ralph_commit="${1:-}"
  python - <<PY
import json,sys,datetime
path = sys.argv[1]
expected_actor = sys.argv[2]
status = sys.argv[3]
increment = sys.argv[4] == '1'
ralph_commit = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None

try:
  with open(path,'r') as f:
    st=json.load(f)
except Exception:
  st={}

it = int(st.get('iteration',1))
if increment:
  it += 1

st.update({
  'iteration': it,
  'expected_actor': expected_actor,
  'status': status,
  'last_update': datetime.datetime.utcnow().isoformat()+"Z",
  'lease_expires_at': (datetime.datetime.utcnow()+datetime.timedelta(minutes=10)).isoformat()+"Z",
})
if ralph_commit:
  st['ralph_commit'] = ralph_commit

with open(path,'w') as f:
  json.dump(st,f,indent=2)
  f.write("\n")
PY
}

for i in $(seq 1 "$SYNC_LOOPS"); do
  echo "[loop] Iteration ${i}/${SYNC_LOOPS}" | tee -a "$LOG_FILE"
  if [[ "$SYNC_VIA_GIT" -eq 1 ]]; then
  mkdir -p "$(dirname "$STATE_FILE")"
  git_safe_pull

  # Wait for our turn
  echo "[SYNC] Waiting for expected_actor=ralph..." | tee -a "$LOG_FILE"
  start_ts=$(date +%s)
  while true; do
    git_safe_pull
    mapfile -t S < <(read_state "$STATE_FILE")
    EXPECTED="${S[0]}"; STATUS="${S[1]}"; ITER="${S[2]}"
    if [[ "$EXPECTED" == "ralph" ]]; then
      break
    fi
    if [[ "$MAX_WAIT_SEC" -gt 0 ]]; then
      now=$(date +%s)
      if (( now - start_ts > MAX_WAIT_SEC )); then
        echo "[SYNC] Timeout waiting for turn; exiting." | tee -a "$LOG_FILE"
        exit 1
      fi
    fi
    sleep "$POLL_INTERVAL"
  done

  # Mark running
  write_state "ralph" "running-ralph" 0 ""
  git add "$STATE_FILE"
  git commit -m "[SYNC i=${ITER}] actor=ralph status=running" || true
  git push || true
fi

# Sync with remote before starting work (always keep up to date)
git_safe_pull

# Execute main prompt once per invocation
cat prompts/main.md | "${CLAUDE_CMD}" -p --dangerously-skip-permissions --verbose --output-format stream-json | tee -a "${LOG_FILE}"
rc=$?

git_safe_pull
SHA=$(git rev-parse --short HEAD || echo unknown)

if [[ "$SYNC_VIA_GIT" -eq 1 ]]; then
  if [[ "$rc" -eq 0 ]]; then
    # On success: complete iteration and handoff to galph; increment iteration
    write_state "galph" "complete" 1 "$SHA"
    git add "$STATE_FILE"
    git commit -m "[SYNC i=${ITER}] actor=ralph → next=galph status=ok ralph_commit=${SHA}" || true
    git push || true
  else
    write_state "ralph" "failed" 0 "$SHA"
    git add "$STATE_FILE"
    git commit -m "[SYNC i=${ITER}] actor=ralph status=fail ralph_commit=${SHA}" || true
    git push || true
  fi
fi

# Conditional push: only if there are commits to push and no errors occurred
if [[ "$rc" -eq 0 ]]; then
  if git diff --quiet origin/$(git rev-parse --abbrev-ref HEAD)..HEAD 2>/dev/null; then
    echo "No new commits to push."
  else
    echo "Pushing commits to remote..."
    git push || {
      echo "WARNING: git push failed. Please push manually."
      exit 1
    }
  fi
else
  echo "Loop execution failed with rc=$rc" | tee -a "$LOG_FILE"
  exit "$rc"
fi
done
