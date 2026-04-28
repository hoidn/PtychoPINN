#!/usr/bin/env bash
set -euo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ptycho311
cd /home/ollie/Documents/PtychoPINN
set +e
python scripts/studies/run_pdebench_image128_suite.py   --task 2d_cfd_cns   --mode pilot   --data-root /home/ollie/Documents/pdebench-data   --output-root ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z"   --profiles spectral_resnet_bottleneck_base,spectral_resnet_bottleneck_shared_blocks8,spectral_resnet_bottleneck_shared_blocks10   --history-len 2   --epochs 40   --batch-size 4   --max-train-trajectories 512   --max-val-trajectories 64   --max-test-trajectories 64   --max-windows-per-trajectory 8   --device cuda   --num-workers 0 &
pid=$!
echo "$pid" > ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z.launch/python_pid.txt"
wait "$pid"
status=$?
set -e
echo "$status" > ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z.launch/exit_code.txt"
echo "$status" > ".artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-depth-shared-40ep-20260428T043715Z/exit_status.txt"
exit "$status"
