# NeurIPS Hybrid ResNet CDI FFNO Generator Lines Best-Config Summary

- Backlog item: `2026-04-27-cdi-ffno-generator-lines-best-config`
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/execution_plan.md`
- Downstream design authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
- Recovery preflight: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`
- Study-contract authority: `docs/studies/index.md#grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3`
- Current state: `completed` on `2026-04-29`

## Fixed Contract

- `N=128`, `gridsize=1`
- synthetic grid-lines with `set_phi=True`
- custom Run1084 probe
- `probe_scale_mode=pad_extrapolate`
- `probe_smoothing_sigma=0.5`
- `nimgs_train=2`, `nimgs_test=2`
- `nphotons=1e9`
- `seed=3`
- `torch_epochs=40`
- `torch_learning_rate=2e-4`
- `torch_scheduler=ReduceLROnPlateau`
- `torch_plateau_factor=0.5`
- `torch_plateau_patience=2`
- `torch_plateau_min_lr=1e-4`
- `torch_plateau_threshold=0.0`
- `torch_loss_mode=mae`
- `torch_mae_pred_l2_match_target=off`
- `torch_output_mode=real_imag`
- `probe_mask=off`
- `fno_modes=12`, `fno_width=32`, `fno_blocks=4`, `fno_cnn_blocks=2`

## Fresh Artifact Root

- Stable compare root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`

## Final Metrics

- `pinn_hybrid_resnet`
  - amp/phase MAE: `0.026939474 / 0.072063477`
  - amp/phase SSIM: `0.988114297 / 0.994739987`
  - amp/phase PSNR: `77.519370679 / 69.216286834`
- `pinn_ffno`
  - amp/phase MAE: `0.062772475 / 0.082838669`
  - amp/phase SSIM: `0.934830340 / 0.981591519`
  - amp/phase PSNR: `70.190080563 / 67.775916878`

## Final Artifact Set

- Wrapper outputs:
  - `metrics.json`
  - `metrics_table.csv`
  - `metrics_table.tex`
  - `metrics_table_best.tex`
  - `visuals/compare_amp_phase.png`
- Row outputs:
  - `runs/pinn_hybrid_resnet/{invocation.json,invocation.sh,metrics.json,history.json,model.pt,randomness_contract.json}`
  - `runs/pinn_ffno/{invocation.json,invocation.sh,metrics.json,history.json,model.pt,randomness_contract.json}`
  - `recons/pinn_hybrid_resnet/recon.npz`
  - `recons/pinn_ffno/recon.npz`
- Fresh verification evidence:
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/20260429T111254Z_pytest.log`
  - `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/verification/20260429T111848Z_compileall.log`

## Compare Command

```bash
python scripts/studies/grid_lines_compare_wrapper.py \
  --N 128 \
  --gridsize 1 \
  --output-dir .artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet \
  --architectures hybrid_resnet,ffno \
  --nimgs-train 2 \
  --nimgs-test 2 \
  --nphotons 1e9 \
  --batch-size 16 \
  --seed 3 \
  --probe-source custom \
  --probe-scale-mode pad_extrapolate \
  --probe-smoothing-sigma 0.5 \
  --set-phi \
  --torch-epochs 40 \
  --torch-batch-size 16 \
  --torch-learning-rate 2e-4 \
  --torch-scheduler ReduceLROnPlateau \
  --torch-plateau-factor 0.5 \
  --torch-plateau-patience 2 \
  --torch-plateau-min-lr 1e-4 \
  --torch-plateau-threshold 0.0 \
  --torch-loss-mode mae \
  --torch-no-mae-pred-l2-match-target \
  --torch-output-mode real_imag \
  --fno-modes 12 \
  --fno-width 32 \
  --fno-blocks 4 \
  --fno-cnn-blocks 2
```

## Status Notes

- This row pair is a prerequisite CDI evidence slice for later `lines128`
  paper-benchmark packaging. It does not replace the planned four-row paper
  benchmark harness or table.
- The checked-in preflight note now records the recovered contract, the
  completed stable output root, and the no-relaunch completion decision.
- The historical `grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3` root is
  decision-support-only because it lacks full child invocation provenance. This
  execution is the repaired stable `hybrid_resnet` versus `ffno` row pair for
  prerequisite CDI evidence, with row-level invocation provenance caveats noted
  below.
- Implementation-review repair backfilled `metrics_table.csv`, the per-row
  `runs/.../invocation.*` artifacts, and wrapper completion metadata from the
  fixed contract without relaunching the finished compare.
- Deterministic verification evidence archived for closeout:
  - `20260429T111254Z_pytest.log` records the required backlog pytest command,
    stable root, start/finish timestamps, and passing output.
  - `20260429T111848Z_compileall.log` records the required `compileall`
    command, stable root, start/finish timestamps, and exit `0`.
- Residual provenance caveat:
  - the per-row invocation artifacts were backfilled from the fixed wrapper
    contract after the original in-process compare completed, so the row
    invocation timestamps reflect the repair pass rather than the original
    training start time.
