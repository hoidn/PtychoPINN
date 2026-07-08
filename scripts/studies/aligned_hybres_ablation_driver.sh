#!/usr/bin/env bash
# aligned_hybres_ablation_driver.sh - Aligned-regime hybrid_resnet ablation driver.
#
# Purpose: rerun the hybrid_resnet dyad + representation-axis ablation knobs
# (training-time probe weighting, rectangular_scaled forward physics,
# amp_phase output mode) on the EXACT config used by the trusted integration
# gate, varying only the ablation knobs one at a time. This resolves the
# earlier misaligned-regime false negative where hybrid_resnet looked
# degenerate under torch_loss_mode='poisson', lr 1e-3 (no scheduler), N=64
# fly001 data instead of the integration test's unsupervised-PINN/MAE
# objective, lr 2e-4 + ReduceLROnPlateau, N=128 grid-lines/Run1084 data.
#
# Alignment contract: every BASE flag below is copied verbatim from
# tests/torch/test_grid_lines_hybrid_resnet_integration.py::test_grid_lines_hybrid_resnet_metrics
# (seed 3). Only the per-arm knob(s) listed in each run_arm call vary.
#
# Dataset expectation: $ROOT/data/{train,test}.npz must be the SAME N=128,
# gridsize=1 dataset the integration test generates via its
# `_ensure_dataset` helper (Run1084 probe, nimgs_train=2, nimgs_test=1,
# nphotons=1e9, probe_smoothing_sigma=0.5, pad_extrapolate, set_phi=True).
# Hardlink or copy the integration test's generated npz's into $ROOT/data
# before running this driver; the integration test's own scratch root is
# .artifacts/integration/grid_lines_hybrid_resnet/datasets/N128/gs1/{train,test}.npz.
#
# Cross-references:
#   - Knowledge-base entry: docs/findings.md, HYBRES-ALIGN-001
#   - Executable regression coverage: tests/torch/test_grid_lines_hybrid_resnet_aligned_ablation.py
#     (marker: grid_lines_hybrid_resnet_aligned_ablation)
set -uo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO"
export PYTHONPATH="$REPO"
ROOT=${ALIGNED_ABLATION_ROOT:-.artifacts/varpro_ablation/ext_matrix_aligned}

BASE=(python scripts/studies/grid_lines_torch_runner.py
  --train-npz "$ROOT/data/train.npz" --test-npz "$ROOT/data/test.npz"
  --architecture hybrid_resnet
  --N 128 --gridsize 1
  --epochs "${ALIGNED_ABLATION_EPOCHS:-5}" --batch-size 16 --infer-batch-size 16
  --learning-rate 2e-4 --scheduler ReduceLROnPlateau
  --plateau-factor 0.5 --plateau-patience 2 --plateau-min-lr 1e-4 --plateau-threshold 0.0
  --seed 3 --optimizer adam --weight-decay 0.0 --beta1 0.9 --beta2 0.999
  --torch-loss-mode mae
  --probe-source custom
  --fno-modes 12 --fno-width 32 --fno-blocks 4 --fno-cnn-blocks 2
  --torch-logger mlflow)

run_arm () {
  local arm="$1"; shift
  if [[ -n "${ALIGNED_ABLATION_ARMS:-}" ]] && [[ " $ALIGNED_ABLATION_ARMS " != *" $arm "* ]]; then
    echo "=== [$arm] not in ALIGNED_ABLATION_ARMS, skipping ==="
    return 0
  fi
  if [[ -f "$ROOT/$arm.exit" ]] && [[ "$(cat "$ROOT/$arm.exit")" == "0" ]]; then
    echo "=== [$arm] already done, skipping ==="
    return 0
  fi
  echo "=== [$arm] start $(date) ==="
  "${BASE[@]}" --output-dir "$ROOT/$arm" "$@" > "$ROOT/$arm.log" 2>&1 &
  local pid=$!
  wait "$pid"
  local rc=$?
  echo "$rc" > "$ROOT/$arm.exit"
  echo "=== [$arm] exit=$rc $(date) ==="
}

run_arm neither      --output-mode real_imag --training-patch-weighting central_mask --physics-forward-mode amplitude
run_arm weight_only  --output-mode real_imag --training-patch-weighting probe        --physics-forward-mode amplitude
# rect_only and both dropped: rectangular_scaled + MAE broken by construction, see docs/findings.md RECT-MAE-UNITS-001.
run_arm ampphase_out --output-mode amp_phase --training-patch-weighting central_mask --physics-forward-mode amplitude
run_arm cnn          --output-mode real_imag --training-patch-weighting central_mask --physics-forward-mode amplitude --architecture cnn
run_arm l2match_on   --output-mode real_imag --training-patch-weighting central_mask --physics-forward-mode amplitude --torch-mae-pred-l2-match-target

echo done > "$ROOT/driver.done"
echo "=== all arms complete $(date) ==="
