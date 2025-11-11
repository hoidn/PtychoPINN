#!/usr/bin/env bash
set -euo pipefail

# Prefer Python implementation; keep legacy below for fallback
if [[ "${ORCHESTRATION_PYTHON:-1}" == "1" ]]; then
  # Resolve the interpreter to the active environment unless overridden
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
  exec "$PYTHON_BIN" -m scripts.orchestration.supervisor "$@"
fi


# Prepare a timestamped log for supervisor runs.
mkdir -p tmp
TS=$(date '+%Y%m%d_%H%M%S')
LOG_FILE="tmp/supervisorlog${TS}.txt"
ln -sf "${LOG_FILE}" tmp/supervisorlog-latest.txt

CODEX_CMD="codex"

# --- Args & defaults ---
SYNC_VIA_GIT=0
SYNC_LOOPS=${SYNC_LOOPS:-20}
POLL_INTERVAL=${POLL_INTERVAL:-5}
MAX_WAIT_SEC=${MAX_WAIT_SEC:-0}  # 0 = no timeout
STATE_FILE="sync/state.json"

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
      # ignore unknowns to remain backward-compatible
      shift ;;
  esac
done

mkdir -p "$(dirname "$STATE_FILE")"

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
  # args: expected_actor status increment galph_commit(optional)
  local expected_actor="$1"; shift
  local status="$1"; shift
  local increment="$1"; shift
  local galph_commit="${1:-}"
  python - <<PY
import json,sys,datetime
path = sys.argv[1]
expected_actor = sys.argv[2]
status = sys.argv[3]
increment = sys.argv[4] == '1'
galph_commit = sys.argv[5] if len(sys.argv) > 5 and sys.argv[5] else None

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
if galph_commit:
  st['galph_commit'] = galph_commit

with open(path,'w') as f:
  json.dump(st,f,indent=2)
  f.write("\n")
PY
}

git_safe_pull() {
  if ! timeout 30 git pull --rebase; then
    echo "WARNING: git pull --rebase failed or timed out. Attempting recovery..." | tee -a "$LOG_FILE"
    git rebase --abort || true
    git pull --no-rebase || true
  fi
}

if [[ "$SYNC_VIA_GIT" -eq 0 ]]; then
  # Run the supervisor prompt repeatedly to manage Ralph's loops (legacy async mode).
  for i in {1..20}; do
    ${CODEX_CMD} exec -m gpt-5-codex -c model_reasoning_effort="high" --dangerously-bypass-approvals-and-sandbox < prompts/supervisor.md | tee -a "${LOG_FILE}"
  done
  exit 0
fi

# --- Synchronous via Git mode ---
echo "Running supervisor in SYNC via git mode for ${SYNC_LOOPS} iteration(s)" | tee -a "$LOG_FILE"

for i in $(seq 1 "$SYNC_LOOPS"); do
  git_safe_pull

  # Initialize state if missing
  if [[ ! -f "$STATE_FILE" ]]; then
    echo "Initializing $STATE_FILE (expected_actor=galph)" | tee -a "$LOG_FILE"
    write_state "galph" "idle" 0 ""
    git add "$STATE_FILE"
    git commit -m "[SYNC init] actor=galph status=idle" || true
    git push || true
  fi

  # Wait for our turn
  echo "Waiting for expected_actor=galph..." | tee -a "$LOG_FILE"
  start_ts=$(date +%s)
  while true; do
    git_safe_pull
    mapfile -t S < <(read_state "$STATE_FILE")
    EXPECTED="${S[0]}"; STATUS="${S[1]}"; ITER="${S[2]}"
    if [[ "$EXPECTED" == "galph" ]]; then
      break
    fi
    if [[ "$MAX_WAIT_SEC" -gt 0 ]]; then
      now=$(date +%s)
      if (( now - start_ts > MAX_WAIT_SEC )); then
        echo "Timeout waiting for turn; exiting." | tee -a "$LOG_FILE"
        exit 1
      fi
    fi
    sleep "$POLL_INTERVAL"
  done

  # Mark running lease and push
  write_state "galph" "running-galph" 0 ""
  git add "$STATE_FILE"
  git commit -m "[SYNC i=${ITER}] actor=galph status=running" || true
  git push || true

  # Execute one supervisor iteration
  set +e
  ${CODEX_CMD} exec -m gpt-5-codex -c model_reasoning_effort="high" --dangerously-bypass-approvals-and-sandbox < prompts/supervisor.md | tee -a "${LOG_FILE}"
  rc=$?
  set -e

  git_safe_pull
  SHA=$(git rev-parse --short HEAD || echo unknown)

  if [[ "$rc" -eq 0 ]]; then
    write_state "ralph" "waiting-ralph" 0 "$SHA"
    git add "$STATE_FILE"
    git commit -m "[SYNC i=${ITER}] actor=galph → next=ralph status=ok galph_commit=${SHA}" || true
    git push || true
  else
    write_state "galph" "failed" 0 "$SHA"
    git add "$STATE_FILE"
    git commit -m "[SYNC i=${ITER}] actor=galph status=fail galph_commit=${SHA}" || true
    git push || true
    echo "Supervisor iteration failed (rc=$rc). Halting sync loop." | tee -a "$LOG_FILE"
    exit "$rc"
  fi

  # Wait here until Ralph completes and flips turn (optional but ensures strict alternation)
  echo "Waiting for Ralph to complete i=${ITER}..." | tee -a "$LOG_FILE"
  while true; do
    git_safe_pull
    mapfile -t S2 < <(read_state "$STATE_FILE")
    NEXT_EXPECTED="${S2[0]}"; NEXT_STATUS="${S2[1]}"; NEXT_ITER="${S2[2]}"
    if [[ "$NEXT_EXPECTED" == "galph" && "$NEXT_ITER" -gt "$ITER" ]]; then
      echo "Ralph completed iteration $ITER; proceeding to $NEXT_ITER" | tee -a "$LOG_FILE"
      break
    fi
    sleep "$POLL_INTERVAL"
  done
done

echo "Supervisor SYNC loop finished." | tee -a "$LOG_FILE"
