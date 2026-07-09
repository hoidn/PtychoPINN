#!/usr/bin/env bash
# Multi-seed Stage A comparison: Default vs ReduceLROnPlateau
# Task 5 of docs/plans/2026-01-29-reduce-lr-plateau.md
#
# Prerequisites: per-seed directories already prepped with datasets.
# Run this when the GPU is free.
set -euo pipefail

SEEDS=(20260128 20260129 20260130)
HUB="plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T220000Z"
COMMON="--N 64 --gridsize 1 --set-phi --nimgs-train 1 --nimgs-test 1 --nphotons 1e9 --nepochs 20 --torch-epochs 20 --torch-loss-mode mae --fno-blocks 4 --torch-infer-batch-size 8"

echo "=== Multi-Seed ReduceLROnPlateau A/B ==="

# Control arm (Default scheduler)
for seed in "${SEEDS[@]}"; do
  outdir="outputs/grid_lines_stage_a/arm_control_seed${seed}"
  log="${HUB}/stage_a_arm_control_seed${seed}.log"
  if [ -f "${outdir}/runs/pinn_hybrid/metrics.json" ]; then
    echo "SKIP: Control seed ${seed} already complete"
    continue
  fi
  rm -rf "${outdir}/runs" training_outputs/checkpoints/* 2>/dev/null || true
  echo "Running: Control seed ${seed}"
  python scripts/studies/grid_lines_compare_wrapper.py \
    ${COMMON} \
    --output-dir "${outdir}" \
    --architectures hybrid \
    --seed "${seed}" \
    --torch-scheduler Default \
    2>&1 | tee "${log}"
  echo "DONE: Control seed ${seed}"
done

# Plateau arm
for seed in "${SEEDS[@]}"; do
  outdir="outputs/grid_lines_stage_a/arm_plateau_seed${seed}"
  log="${HUB}/stage_a_arm_plateau_seed${seed}.log"
  if [ -f "${outdir}/runs/pinn_hybrid/metrics.json" ]; then
    echo "SKIP: Plateau seed ${seed} already complete"
    continue
  fi
  rm -rf "${outdir}/runs" training_outputs/checkpoints/* 2>/dev/null || true
  echo "Running: Plateau seed ${seed}"
  python scripts/studies/grid_lines_compare_wrapper.py \
    ${COMMON} \
    --output-dir "${outdir}" \
    --architectures hybrid \
    --seed "${seed}" \
    --torch-scheduler ReduceLROnPlateau \
    2>&1 | tee "${log}"
  echo "DONE: Plateau seed ${seed}"
done

echo "=== All runs complete. Now dumping stats... ==="

# Dump per-run stats
for arm in control plateau; do
  for seed in "${SEEDS[@]}"; do
    run_dir="outputs/grid_lines_stage_a/arm_${arm}_seed${seed}/runs/pinn_hybrid"
    python scripts/internal/stage_a_dump_stats.py \
      --run-dir "${run_dir}" \
      --out-json "${HUB}/stage_a_arm_${arm}_seed${seed}_stats.json"
  done
done

# Aggregate summary
python - <<'PY'
import json, pathlib
hub = pathlib.Path("plans/active/FNO-STABILITY-OVERHAUL-001/reports/2026-01-29T220000Z")
summary = {"control": [], "plateau": []}
for arm in summary:
    for stats_file in sorted(hub.glob(f"stage_a_arm_{arm}_seed*_stats.json")):
        data = json.loads(stats_file.read_text())
        summary[arm].append({"seed": stats_file.stem.split('seed')[-1].split('_')[0], **data})
hub.joinpath("stage_a_plateau_multiseed_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
PY

echo "=== Multi-seed comparison complete ==="
