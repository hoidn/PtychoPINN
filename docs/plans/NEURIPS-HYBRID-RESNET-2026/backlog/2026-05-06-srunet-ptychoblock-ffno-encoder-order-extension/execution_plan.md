# SRU-Net PtychoBlock-To-FFNO Encoder-Order Extension Plan

- Backlog item:
  `docs/backlog/active/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension.md`
- Summary target:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_encoder_order_ffno_vs_ptychoblock_summary.md`
- Artifact root:
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-srunet-ptychoblock-ffno-encoder-order-extension/`

## Intent

Add one reversed-order SRU-Net encoder row:

`2x(PtychoBlock + downsample) -> 24-layer FFNO -> unchanged SRU-Net bottleneck/decoder`

The completed comparison row is:

`24-layer FFNO -> 2x(PtychoBlock + downsample) -> unchanged SRU-Net bottleneck/decoder`

The implementation must isolate encoder order. It must not rerun completed
baselines, replace the bottleneck with FFNO, or collapse the FFNO stack into a
lightweight two-block proxy.

## Required Inputs

- Completed FFNO-to-PtychoBlock item:
  `docs/backlog/done/2026-05-05-srunet-ffno-ptychoblock-encoder-cdi-cns-smallcap.md`
- Completed FFNO-to-PtychoBlock summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/srunet_ffno_ptychoblock_encoder_cdi_cns_smallcap_summary.md`
- Current model surfaces:
  - `ptycho_torch/generators/hybrid_resnet.py`
  - `ptycho_torch/generators/ffno_bottleneck.py`
  - `ptycho_torch/generators/registry.py`
  - `ptycho_torch/model.py`
  - `scripts/studies/grid_lines_torch_runner.py`
  - `scripts/studies/grid_lines_compare_wrapper.py`
  - `scripts/studies/pdebench_image128/models.py`
  - `scripts/studies/pdebench_image128/run_config.py`

## Implementation Steps

1. Add the reversed-order architecture.
   - Introduce a clearly named CDI architecture such as
     `hybrid_resnet_ptychoblock_ffno_encoder`.
   - Introduce a matching CNS profile such as
     `hybrid_resnet_ptychoblock_ffno_encoder_cns`.
   - Reuse the regular SRU-Net lifter, two encoder stages, downsampling,
     skip taps, bottleneck, decoder, residual scaling, and output path.
   - Insert the 24-layer FFNO-close stack after the second downsample and before the
     existing SRU-Net bottleneck.

2. Keep the FFNO recipe fixed.
   - Use `SharedFactorizedFfnoBottleneck` / `FactorizedFfnoBlock`.
   - Default recipe:
     `ffno_encoder_blocks=24`, `ffno_encoder_modes=12`,
     `ffno_encoder_share_weights=true`, `ffno_encoder_gate_init=0.1`,
     `ffno_encoder_norm=instance`, `ffno_encoder_mlp_ratio=2.0`,
     `local_conv_kernel_size=None`.
   - If post-downsample resolution forces a mode-count adjustment, record the
     adjustment as a shape-compatibility decision, not tuning.

3. Wire study entrypoints.
   - Register the CDI row id, display label, route, config capture, and wrapper
     validation for `pinn_hybrid_resnet_ptychoblock_ffno_encoder`.
   - Register the CNS profile as manual/append-only, not part of primary
     CNS bundles.
   - Ensure checkpoint/config reconstruction preserves:
     `encoder_variant`, `encoder_order`, `ptychoblock_stage_count`,
     `downsample_steps`, `downsample_op`, `ffno_encoder_blocks`,
     `ffno_encoder_modes`, `ffno_encoder_share_weights`,
     `ffno_encoder_gate_init`, `ffno_encoder_norm`,
     `ffno_encoder_mlp_ratio`.

4. Run only the two new rows.
   - CDI: fixed `cdi_lines128_seed3` contract, `N=128`, `40` epochs,
     batch `16`, Adam `2e-4`, MAE loss, fixed probe/sample policy.
   - CNS: `2d_cfd_cns`, `history_len=5`, `512 / 64 / 64`, `40` epochs,
     batch `4`, Adam `2e-4`, `max_windows_per_trajectory=8`, MSE loss.

5. Publish an append-only comparison.
   - Reuse prior rows by lineage:
     regular SRU-Net, FFNO-to-PtychoBlock, spectral-only encoder, FFNO, and
     matched CNS comparator rows.
   - Emit a three-row encoder-order table for CDI and CNS where same-contract
     rows exist.
   - Update evidence/discoverability indexes without overwriting the existing
     Lines128 or CNS authorities.

## Verification

Run narrow tests first:

```bash
pytest -q tests/torch/test_fno_generators.py -k "PtychoBlock or hybrid_resnet or ffno"
pytest -q tests/torch/test_generator_registry.py -k "hybrid_resnet or ffno"
pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno"
pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno or hybrid_resnet"
pytest -q tests/studies/test_pdebench_image128_models.py -k "hybrid_resnet or ffno"
python -m compileall -q ptycho_torch scripts/studies
```

After the rows run, verify:

- row invocations exit `0`;
- CDI metrics and reconstruction artifacts exist;
- CNS metrics, comparison PNG/NPZ, and profile manifest exist;
- the durable summary cites source paths for reused rows;
- index updates preserve claim boundaries and append-only status.

## Review Checks

- Reject any implementation that changes the bottleneck, decoder, skip fusion,
  loss, scheduler, probe/data contract, CNS history length, CNS split cap, or
  metric definitions while claiming to test encoder order.
- Reject any implementation that uses a two-block FFNO proxy for the intended
  full FFNO encoder row.
- Reject any summary that treats the local FFNO-close stack as identical to
  `pinn_ffno` or `author_ffno_cns_base`.
- Reject reruns of completed baseline rows unless there is a documented
  artifact-integrity problem unrelated to this new row.
