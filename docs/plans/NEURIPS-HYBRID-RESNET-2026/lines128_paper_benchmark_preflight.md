# NeurIPS Lines128 Paper Benchmark Preflight

- Date: `2026-04-29`
- Backlog item: `2026-04-27-cdi-ffno-generator-lines-best-config`
- Governing plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/execution_plan.md`
- Downstream benchmark design: `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
- Stable output root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet`

## Recovered Contract

- Study identity and legacy-best target:
  `docs/studies/index.md#grid-lines-n128-hybrid-resnet-legacy-best-e40-seed3`
- Fixed compare surface and claim boundary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_ffno_generator_lines_best_config_summary.md`
- Paper-benchmark contract and preflight requirement:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_design.md`
- Compare command and live invocation audit:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-27-cdi-ffno-generator-lines-best-config/lines128_ffno_vs_hybrid_resnet/invocation.json`

Recovered fixed contract fields:

- Dataset and probe contract:
  `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`, custom
  Run1084 probe, `probe_scale_mode=pad_extrapolate`,
  `probe_smoothing_sigma=0.5`
- Train/test split contract:
  `nimgs_train=2`, `nimgs_test=2`, `seed=3`
- Physics / measurement contract:
  `nphotons=1e9`, `probe_mask=off`
- Optimization contract:
  `torch_epochs=40`, `torch_learning_rate=2e-4`,
  `torch_scheduler=ReduceLROnPlateau`, `torch_plateau_factor=0.5`,
  `torch_plateau_patience=2`, `torch_plateau_min_lr=1e-4`,
  `torch_plateau_threshold=0.0`
- Loss / output contract:
  `torch_loss_mode=mae`, `torch_mae_pred_l2_match_target=off`,
  `torch_output_mode=real_imag`
- FFNO shape contract:
  `fno_modes=12`, `fno_width=32`, `fno_blocks=4`,
  `fno_cnn_blocks=2`

## Fixed Row List And Seed Policy

- Row list:
  - `pinn_hybrid_resnet`
  - `pinn_ffno`
- Fixed-seed policy:
  - use `seed=3` for both rows
  - do not broaden this item into a multi-seed benchmark

## Active Root Audit

Audit time: `2026-04-29` local checkout review.

- Active writer confirmed:
  - tracked shell PID `831894`
  - live wrapper PID `831899`
- Wrapper invocation present:
  - `invocation.json`
  - `invocation.sh`
- Shared dataset generation has started under:
  - `datasets/N128/gs1/`
- Partial runtime evidence exists:
  - `tmux.log`
  - `lightning_logs/`
  - transient `checkpoints/`
  - `recons/gt/recon.npz`
- Completion artifacts are still incomplete:
  - no wrapper-level merged `metrics.json`
  - no wrapper-level tables or combined visuals
  - no completed `runs/pinn_hybrid_resnet/`
  - no completed `runs/pinn_ffno/`

This root is therefore recoverable in-progress state, not proof of completion.

## Resume Command

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

## Go / No-Go Decision

- Decision: `GO - resume the active root; do not relaunch`
- Reason:
  - an active writer is still attached to the stable output root
  - the wrapper command matches the recovered fixed contract
  - launching another compare into the same root would violate the
    no-duplicate-run guardrail and risk mixing artifacts

## Partial-Output Preservation Policy

- While PID `831894` is active, preserve the stable root in place and treat all
  current files as transient partial outputs only.
- If the tracked PID exits non-zero or the wrapper root remains incomplete after
  exit, preserve `invocation.json`, `invocation.sh`, and `tmux.log` as the
  stale-attempt record before any relaunch decision.
- Do not allow wrapper-only files, checkpoints, or dataset byproducts to
  masquerade as completion without fresh merged metrics, visuals, and both row
  result trees.
