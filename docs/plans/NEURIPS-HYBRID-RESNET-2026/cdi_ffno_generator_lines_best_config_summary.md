# NeurIPS Hybrid ResNet CDI FFNO Generator Lines Best-Config Summary

- Backlog item: `2026-04-27-cdi-ffno-generator-lines-best-config`
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/execution_plan.md`
- Downstream design authority: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
- Study-contract authority: `docs/studies/index.md#grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3`
- Current state: `run_launched_pending_completion`

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
- The historical `grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3` root is
  decision-support-only because it lacks full child invocation provenance. This
  execution is the fresh auditable row pair for `hybrid_resnet` versus `ffno`.
- Deterministic code gates completed before launch:
  - `pytest -q tests/torch/test_grid_lines_hybrid_resnet_integration.py tests/torch/test_grid_lines_torch_runner.py tests/test_grid_lines_compare_wrapper.py`
  - `python -m compileall -q ptycho_torch scripts/studies`
