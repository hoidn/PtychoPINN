---
priority: 2
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md"),
        Path("ptycho_torch/generators/ffno_bottleneck.py"),
        Path("ptycho_torch/generators/hybrid_resnet.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
        Path("scripts/studies/run_pdebench_image128_suite.py"),
        Path("scripts/studies/pdebench_image128/models.py"),
        Path("scripts/studies/pdebench_image128/run_config.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing SRU-Net PtychoBlock->FFNO encoder-order inputs: {missing}")
    print("SRU-Net PtychoBlock->FFNO encoder-order inputs present")
    PY
  - pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"
  - pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"
  - pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno or hybrid_resnet"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The completed FFNO-to-PtychoBlock encoder-order row tests global factorized spectral mixing before the two SRU-Net encoder stages.
  - A reversed PtychoBlock-to-FFNO row is needed to separate whether the effect comes from the FFNO operator itself or from placing FFNO before versus after the two downsampling encoder stages.
  - This item is a bounded third-row extension of the same encoder-order study and should run before lower-priority WaveBench candidate items.
---

# Backlog Item: Add SRU-Net PtychoBlock-To-FFNO Encoder-Order Row

## Objective

- Extend the completed SRU-Net FFNO encoder-order study with one additional
  architecture row:
  - two ordinary SRU-Net `PtychoBlock` encoder stages with the existing
    downsampling schedule; then
  - a full 24-layer FFNO-style factorized spectral stack at the post-downsample encoder
    resolution; then
  - the unchanged SRU-Net bottleneck, decoder, skip structure, output mode, and
    training recipe.
- Compare this reversed-order row against the already completed
  `FFNO -> 2x(PtychoBlock + downsample)` row and the regular SRU-Net lineage
  rows on:
  - the fixed `lines128` CDI benchmark; and
  - the small-cap PDEBench `2d_cfd_cns` matched-condition benchmark.

## Scope

- Implement one explicit architecture/profile family, for example
  `hybrid_resnet_ptychoblock_ffno_encoder`.
- Preserve the SRU-Net shell outside the encoder-order change:
  - same lifter/input transform policy as the regular SRU-Net row;
  - same two downsampling stages, downsample operator, skip taps, skip fusion
    style, bottleneck family, decoder family, residual-scaling policy, loss,
    scheduler, seed policy, visual sample policy, and metric schema as the
    completed FFNO-to-PtychoBlock item;
  - no hyperparameter tuning after seeing metrics.
- Use the same FFNO-close component family as the completed
  FFNO-to-PtychoBlock item unless the implementation plan justifies a narrower
  compatibility adapter:
  - `SharedFactorizedFfnoBottleneck` / `FactorizedFfnoBlock`;
  - `ffno_encoder_blocks=24`;
  - `ffno_encoder_modes=12` unless the implementation records a shape-based
    reason for an adjusted post-downsample mode count;
  - `ffno_encoder_share_weights=true`;
  - `ffno_encoder_gate_init=0.1`;
  - `ffno_encoder_norm=instance`;
  - `ffno_encoder_mlp_ratio=2.0`;
  - `local_conv_kernel_size=None`.
- Place the FFNO stack after the two encoder stages and their downsampling
  layers, before the existing SRU-Net bottleneck. Do not replace the bottleneck
  with FFNO; that is the already completed FFNO-bottleneck bridge mechanism.
- Launch only the new rows:
  - `pinn_hybrid_resnet_ptychoblock_ffno_encoder` or equivalent on the fixed
    Lines128 CDI contract;
  - a matching small-cap CNS profile under the existing matched-condition CNS
    contract.
- Reuse all prior rows by lineage:
  - regular `pinn_hybrid_resnet`;
  - `pinn_hybrid_resnet_ffno_ptychoblock_encoder`;
  - `pinn_hybrid_resnet_encoder_spectral_only`;
  - `pinn_ffno`;
  - CNS `spectral_resnet_bottleneck_base` / SRU-Net*, `author_ffno_cns_base`,
    `fno_base`, and `unet_strong`.
- Do not rerun completed baseline or FFNO-to-PtychoBlock rows just to assemble
  the third-row comparison.

## Fixed Benchmark Contracts

### CDI Lines128

- Use the same fixed `cdi_lines128_seed3` contract as the completed
  FFNO-to-PtychoBlock item:
  - `N=128`, `gridsize=1`, `seed=3`;
  - `40` epochs, batch `16`, Adam `2e-4`;
  - `torch_loss_mode=mae`, `torch_output_mode=real_imag`;
  - Run1084 fixed-probe lineage, fixed sample ids, and metric schema unchanged.

### CNS Small-Cap

- Use the matched-condition CNS lane:
  - task: PDEBench `2d_cfd_cns`;
  - `history_len=5`;
  - split caps `512 / 64 / 64`;
  - `40` epochs, batch size `4`, Adam `2e-4`;
  - `max_windows_per_trajectory=8`;
  - training loss: `mse`;
  - metric family: `err_RMSE`, `err_nRMSE`, `relative_l2`,
    `fRMSE_low`, `fRMSE_mid`, `fRMSE_high`.

## Required Interpretation

- Frame this as an encoder-order ablation, not a new default SRU-Net family.
- The causal question is whether a full 24-layer FFNO stack is more useful before the two
  `PtychoBlock` downsampling stages or after them.
- Do not describe the local FFNO-close encoder stack as identical to the
  end-to-end CDI `pinn_ffno` row or the authored CNS FFNO baseline. Record the
  exact FFNO component and configuration in manifests and summaries.
- Keep CDI and CNS conclusions separate. CNS remains bounded capped
  decision-support unless a later roadmap decision reopens full-training CNS
  evidence.
- If the reversed-order row helps one benchmark and hurts the other, report the
  domain-dependent result directly rather than averaging across benchmarks.

## Outputs

- Row-local invocation/config/history/metrics/reconstruction artifacts for the
  Lines128 CDI row.
- Row-local invocation/config/history/metrics/field-visual artifacts for the
  CNS small-cap row.
- A concise durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_encoder_order_ffno_vs_ptychoblock_summary.md`
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md` only as
    bounded mechanism evidence;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`;
  - `docs/studies/index.md`.

## Completion Gate

- The new row must differ from the completed FFNO-to-PtychoBlock row only by
  encoder ordering and any documented shape adapter needed to place the FFNO
  stack after downsampling.
- The summary must publish a three-row encoder-order table containing regular
  SRU-Net, FFNO-to-PtychoBlock, and PtychoBlock-to-FFNO for CDI and CNS where
  each row is available under the fixed benchmark contract.
- Any missing or incompatible row must be represented as a precise row-level
  blocker, not as a rerun of unrelated baselines.

## Notes For Reviewer

- Reject implementations that replace the SRU-Net bottleneck with FFNO while
  claiming to test encoder order.
- Reject implementations that change skip wiring, residual scaling, loss,
  scheduler, probe/data contract, CNS history length, CNS split cap, or metric
  definitions.
- Reject implementations that use a two-block FFNO proxy for the reversed-order
  row; two-block artifacts belong only in misconfigured diagnostic context.
- Reject summaries that imply the local FFNO-close stack is the same model as
  `author_ffno_cns_base` or the full CDI `pinn_ffno` generator.
- Require manifest fields for:
  `encoder_variant`, `encoder_order`, `ptychoblock_stage_count`,
  `downsample_steps`, `downsample_op`, `ffno_encoder_blocks`,
  `ffno_encoder_modes`, `ffno_encoder_share_weights`,
  `ffno_encoder_gate_init`, `ffno_encoder_norm`, and
  `ffno_encoder_mlp_ratio`.
