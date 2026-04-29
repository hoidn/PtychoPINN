# 1024 Same-Cap Audit

- Outcome: `fallback_to_512_required`
- Locked fallback manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-benchmark-rows/cns_paper_locked_rows.json`
- Rerun manifest: `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/1024_rerun_manifest.json`

## Compatible 1024 Rows
- `spectral_resnet_bottleneck_base`: `/home/ollie/Documents/PtychoPINN/.artifacts/NEURIPS-HYBRID-RESNET-2026/phase-2-pdebench-cns-hybrid-spectral-architecture-ablation/cns-hybrid-spectral-finalists-1024cap-40ep-20260428T054559Z`

## Missing Or Incompatible Rows
- `fno_base`: `missing_same_contract_1024_row`
  rerun: `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode readiness --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/rerun_candidates/fno-1024cap-40ep --profiles fno_base --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 1024 --max-val-trajectories 128 --max-test-trajectories 128 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
- `unet_strong`: `missing_same_contract_1024_row`
  rerun: `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode readiness --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/rerun_candidates/unet_strong-1024cap-40ep --profiles unet_strong --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 1024 --max-val-trajectories 128 --max-test-trajectories 128 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
- `author_ffno_cns_base`: `missing_same_contract_1024_row`
  rerun: `python scripts/studies/run_pdebench_image128_suite.py --task 2d_cfd_cns --mode readiness --data-root /home/ollie/Documents/pdebench-data --output-root .artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cns-paper-table-figure-bundle/rerun_candidates/author_ffno_cns-1024cap-40ep --profiles author_ffno_cns_base --history-len 2 --epochs 40 --batch-size 4 --max-train-trajectories 1024 --max-val-trajectories 128 --max-test-trajectories 128 --max-windows-per-trajectory 8 --device cuda --num-workers 0`
