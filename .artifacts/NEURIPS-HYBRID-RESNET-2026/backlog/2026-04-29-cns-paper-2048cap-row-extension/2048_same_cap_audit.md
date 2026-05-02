# 2048 Same-Cap Audit

- Outcome: `fallback_to_512_required`
- Target split: `2048cap`
- Locked fallback manifest: `/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`
- Rerun manifest: `/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/2048_rerun_manifest.json`

## Compatible 2048 Rows
- `spectral_resnet_bottleneck_base`: `/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-scaling-2048cap/cns-hybrid-spectral-finalists-2048cap-40ep-20260428T201926Z`

## Missing Or Incompatible Rows
- `fno_base`: `missing_same_contract_2048_row`
  rerun: `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot --data-root /home/ollie/Documents/pdebench-data --output-root /home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/rerun_candidates/fno-2048cap-40ep --profiles fno_base --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 2048 --max-val-trajectories 256 --max-test-trajectories 256 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
- `unet_strong`: `missing_same_contract_2048_row`
  rerun: `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot --data-root /home/ollie/Documents/pdebench-data --output-root /home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/rerun_candidates/unet_strong-2048cap-40ep --profiles unet_strong --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 2048 --max-val-trajectories 256 --max-test-trajectories 256 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
- `author_ffno_cns_base`: `missing_same_contract_2048_row`
  rerun: `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode pilot --data-root /home/ollie/Documents/pdebench-data --output-root /home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-2048cap-row-extension/rerun_candidates/author_ffno_cns-2048cap-40ep --profiles author_ffno_cns_base --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 2048 --max-val-trajectories 256 --max-test-trajectories 256 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
