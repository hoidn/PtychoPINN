#!/usr/bin/env bash
set -euo pipefail
HUB="outputs/dense_exec_100k_2025-11-12T224540Z"
PHASE_C_ROOT="$HUB/data/phase_c"
PHASE_E_ROOT="$HUB/data/phase_e"
PHASE_F_ROOT="$HUB/data/phase_f"
CLI_DIR="$HUB/cli"
LOG="$CLI_DIR/phase_g_rerun.log"
BASE_BUNDLE="$PHASE_E_ROOT/dose_100000/baseline/gs1/wts.h5.zip"
DENSE_BUNDLE="$PHASE_E_ROOT/dose_100000/dense/gs2/wts.h5.zip"
for i in {1..720}; do  # wait up to 12 hours, check each minute
  if [ -f "$BASE_BUNDLE" ] && [ -f "$DENSE_BUNDLE" ]; then
    echo "$(date -Is) Both bundles present. Starting Phase G..." > "$LOG"
    python -m studies.fly64_dose_overlap.comparison \
      --phase-c-root "$PHASE_C_ROOT" \
      --phase-e-root "$PHASE_E_ROOT" \
      --phase-f-root "$PHASE_F_ROOT" \
      --artifact-root "$HUB/analysis" \
      --dose 100000 \
      --view dense \
      --split train >> "$LOG" 2>&1 || true
    python -m studies.fly64_dose_overlap.comparison \
      --phase-c-root "$PHASE_C_ROOT" \
      --phase-e-root "$PHASE_E_ROOT" \
      --phase-f-root "$PHASE_F_ROOT" \
      --artifact-root "$HUB/analysis" \
      --dose 100000 \
      --view dense \
      --split test >> "$LOG" 2>&1 || true
    echo "$(date -Is) Phase G attempts complete" >> "$LOG"
    exit 0
  fi
  sleep 60
done
exit 1
