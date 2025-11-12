#!/bin/bash
set -e

# Ensure project root in PYTHONPATH; rely on environment's python
export PYTHONPATH="$PWD:${PYTHONPATH}"

export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
export HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier"

echo "=== Phase G Dense Full Run Execution ==="
echo "Python: $(command -v python)"
echo "PYTHONPATH: $PYTHONPATH"
echo "Hub: $HUB"
echo "Dose: 1000"
echo "View: dense"
echo "Splits: train test"
echo ""

# Ensure hub subdirectories exist before tee attempts
mkdir -p "$HUB/cli" "$HUB/green" "$HUB/red" "$HUB/analysis"

# Execute Phase G orchestrator
python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
    --hub "$HUB" \
    --dose 1000 \
    --view dense \
    --splits train test \
    --clobber 2>&1 | tee "$HUB"/cli/run_phase_g_dense_stdout.log

echo ""
echo "=== Phase G Dense Post-Verify Only ==="
# Execute post-verify-only to regenerate verification artifacts
python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py \
    --hub "$HUB" \
    --post-verify-only 2>&1 | tee "$HUB"/cli/run_phase_g_dense_post_verify_only.log

echo ""
echo "Phase G execution completed!"
