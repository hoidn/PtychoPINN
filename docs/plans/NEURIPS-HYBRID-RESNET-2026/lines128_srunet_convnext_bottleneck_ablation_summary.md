# NeurIPS Lines128 SRU-Net ConvNeXt-Bottleneck Ablation Summary

- Date: `2026-05-04`
- Backlog item: `2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation`
- Plan: `docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation/execution_plan.md`
- State: `decision_support_append_only`
- Fixed contract id: `cdi_lines128_seed3`
- Authoritative ablation root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation/runs/ablation_20260505T035147Z`
- Bundle artifact:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation/bundle/`
- Authoritative baseline source root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux`

## Claim Boundary

This bundle is `decision_support_append_only`. It does not replace the
immutable six-row complete CDI authority
(`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`),
the U-NO append-only extension
(`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`),
the SRU-Net branch / objective ablation
(`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_srunet_branch_objective_ablation_summary.md`),
or the skip / residual ablation
(`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_hybrid_resnet_skip_residual_ablation_summary.md`).
It is mechanism evidence around the SRU-Net (Hybrid ResNet) bottleneck on the
locked `lines128` CDI contract.

## Fixed Contract

- dataset contract id: `cdi_lines128_seed3`
- `N = 128`, `gridsize = 1`, `seed = 3`
- `40` epochs, `batch_size = 16`, `learning_rate = 2e-4`
- `ReduceLROnPlateau` (factor `0.5`, patience `2`, min_lr `1e-4`, threshold `0.0`)
- `torch_loss_mode = mae`, `torch_output_mode = real_imag`
- `fno_modes = 12`, `fno_width = 32`, `fno_blocks = 4`, `fno_cnn_blocks = 2`
- `nimgs_train = 2`, `nimgs_test = 2`, `nphotons = 1e9`
- `set_phi = True`, probe `Run1084_recon3_postPC_shrunk_3.npz`,
  `probe_source = custom`, `probe_scale_mode = pad_extrapolate`,
  `probe_smoothing_sigma = 0.5`
- fixed sample ids: `0`, `1`

## Row Roster

| model_id | label | architecture | training | evidence_source |
|---|---|---|---|---|
| `pinn_hybrid_resnet` | Hybrid ResNet + PINN | `hybrid_resnet` | pinn | promoted_by_lineage |
| `pinn_hybrid_resnet_convnext_bottleneck` | Hybrid ResNet (ConvNeXt bottleneck) + PINN | `hybrid_resnet_convnext_bottleneck` | pinn | fresh_run |

## Reused Baseline Lineage

The `pinn_hybrid_resnet` row is reused by lineage from the authoritative
six-row CDI bundle root (`runs/pinn_hybrid_resnet/`). It is not rerun for this
backlog item.

## Implementation Surfaces Touched

- `ptycho_torch/generators/resnet_components.py`:
  - new `ConvNextBottleneckBlock` (depthwise 7x7 with reflect padding ->
    `InstanceNorm2d` channels-first norm -> 1x1 expand 4x -> GELU -> 1x1
    project, residual with shared LayerScale gamma)
  - new `ConvNextBottleneck` stack (constant-resolution, learned/fixed
    LayerScale, shared per-block gamma)
- `ptycho_torch/generators/hybrid_resnet.py`:
  - new `HybridResnetConvNextBottleneckGeneratorModule` that subclasses
    `HybridResnetGeneratorModule` and replaces only `self.resnet` with
    `ConvNextBottleneck`
  - new `HybridResnetConvNextBottleneckGenerator` registry wrapper
- `ptycho_torch/generators/registry.py`:
  - registers `hybrid_resnet_convnext_bottleneck`
- `ptycho/config/config.py`, `ptycho_torch/config_params.py`:
  - extends `architecture` Literal and `valid_arches` set so the new id is
    accepted in CDI / runner / wrapper validation paths
  - reuses the SRU-Net branch (`hybrid_resnet`) `fno_blocks >= 3` shell guard
- `scripts/studies/grid_lines_torch_runner.py`:
  - `architecture` choices and `arch_literal` cast extended
  - `PAPER_MODEL_LABELS` adds the new label
- `scripts/studies/grid_lines_compare_wrapper.py`:
  - `TORCH_MODEL_IDS`, `PAPER_MODEL_LABELS`, `DEFAULT_TORCH_ROW_SPECS` register
    `pinn_hybrid_resnet_convnext_bottleneck` with `lock_row_status: True` and
    `row_status: decision_support_append_only`
- new helper `scripts/studies/lines128_srunet_convnext_bottleneck_ablation.py`:
  - promotes the baseline `pinn_hybrid_resnet` row by lineage
  - collates the fresh ConvNeXt row under a fresh append-only bundle leaf
  - fails closed if baseline lineage or fresh-row completion proof is missing

## Tests Added

- `tests/torch/test_fno_generators.py::TestConvNextBottleneck` (9 cases) and
  `TestHybridResnetConvNextBottleneckGenerator` (4 cases): block / bottleneck
  shape preservation, shared-LayerScale invariance, default LayerScale init
  matches SRU-Net convention (`0.1`), fixed-mode disables grad, invalid kernel
  size / layerscale_init / n_blocks raise, expand-act-project topology, end-
  to-end module shape, shell preserved (only `self.resnet` swapped), registry
  resolves the new architecture.
- `tests/studies/test_lines128_srunet_convnext_bottleneck_ablation.py`:
  baseline lineage promotion + fresh row collation, completion-proof
  recording, legacy `launcher_completion.json` acceptance, missing baseline
  raises, missing fresh row raises, missing completion proof raises, refuses
  to overwrite existing bundle outputs.

## Metric Deltas vs. `pinn_hybrid_resnet`

Per-row metrics are pulled from `runs/<row>/metrics.json` and the
ablation bundle. All entries use the `[amplitude, phase]` format for `amp_*`
/ `phase_*` derived components.

| Row | amp_mae | phase_mae | amp_ssim | phase_ssim | amp_psnr | phase_psnr | amp_frc50 | phase_frc50 |
|---|---|---|---|---|---|---|---|---|
| `pinn_hybrid_resnet` (baseline) | 0.0269 | 0.0721 | 0.9881 | 0.9947 | 77.52 | 69.22 | 135.46 | 106.80 |
| `pinn_hybrid_resnet_convnext_bottleneck` | 0.0282 | 0.0718 | 0.9863 | 0.9936 | 77.12 | 69.11 | 135.57 | 135.01 |

Δ vs. baseline (negative is better for MAE; positive is better for SSIM / PSNR / FRC50):

| Row | Δ amp_mae | Δ phase_mae | Δ amp_ssim | Δ phase_ssim | Δ amp_psnr | Δ phase_psnr | Δ amp_frc50 | Δ phase_frc50 |
|---|---|---|---|---|---|---|---|---|
| `pinn_hybrid_resnet_convnext_bottleneck` | +0.0013 | −0.0002 | −0.0018 | −0.0012 | −0.40 | −0.11 | +0.11 | +28.21 |

## Interpretation

- **Bottleneck block-family swap is approximately neutral on the global
  MAE/SSIM/PSNR axes.** Replacing the constant-resolution ResNet bottleneck
  with a ConvNeXt-style depthwise + LayerNorm + 1x1 expand/project stack — at
  the same width (`32 -> 128` after two stride-2 downsamples) and same block
  count (`6`) — moves amplitude MAE / SSIM / PSNR by less than the noise floor
  observed across same-contract Lines128 rows. The amplitude side regresses
  marginally (Δ amp_mae +0.0013, Δ amp_ssim −0.0018, Δ amp_psnr −0.40); the
  phase side moves slightly the other way on MAE (−0.0002) but slightly down
  on SSIM/PSNR. None of these deltas individually justify swapping the paper
  default away from the SRU-Net ResNet bottleneck.
- **High-frequency phase fidelity improves substantially.** Δ phase_frc50 =
  `+28.21` is a large improvement (`106.8 -> 135.0`). Together with the
  `pinn_hybrid_resnet_encoder_spectral_only` and
  `pinn_hybrid_resnet_encoder_conv_only` rows from the SRU-Net
  branch / objective ablation — both of which also pushed phase_frc50 above
  the baseline — this is consistent evidence that the both-branch SRU-Net
  bottleneck baseline is leaving high-frequency phase headroom on the table.
  The ConvNeXt swap captures a portion of that headroom while keeping
  amplitude almost unchanged.
- **Same-contract guard.** This row uses the `hybrid_resnet_convnext_bottleneck`
  architecture id with the current SRU-Net LayerScale convention
  (init `0.1`, learned, shared per block). The canonical tiny-ConvNeXt
  LayerScale (`1e-6`) initialization is intentionally deferred to a later
  optional item so that block family is the only intended axis here. Encoder,
  downsample, decoder, skip policy, output projection, loss, schedule, probe,
  dataset, and visual scales are unchanged; the run-level
  `model_manifest.json` records `claim_boundary == decision_support_append_only`
  and `row_status == decision_support_append_only` for the new row.
- **Non-promotion.** The completed six-row CDI bundle remains the headline
  authority. This item is append-only mechanism evidence and is not a
  candidate for replacing the SRU-Net + PINN paper row, but Δ phase_frc50
  motivates a follow-up bounded item to test the canonical tiny-ConvNeXt
  LayerScale (`1e-6`) initialization on the same locked contract.

## Verification

Verification logs are archived under
`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation/verification/`.

- Required deterministic gates (timestamp `20260505T035026Z`):
  - prerequisite presence — `prereq_20260505T035026Z.log` (exit 0)
  - `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or convnext"` —
    `test_fno_generators_20260505T035026Z.log` (62 passed)
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or convnext"` —
    `test_grid_lines_torch_runner_20260505T035026Z.log` (6 passed)
  - `pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or convnext"` —
    `test_grid_lines_compare_wrapper_20260505T035026Z.log` (7 passed)
  - `pytest -q tests/studies/test_lines128_srunet_convnext_bottleneck_ablation.py` —
    `test_ablation_helper_20260505T035026Z.log` (7 passed)
  - `python -m compileall -q scripts/studies ptycho_torch` —
    `compileall_20260505T035026Z.log` (exit 0)
- Wrapper completion proof:
  - run-root `runs/ablation_20260505T035147Z/invocation.json`: `status=completed`,
    `exit_code=0`
  - per-fresh-row
    `runs/ablation_20260505T035147Z/runs/pinn_hybrid_resnet_convnext_bottleneck/exit_code_proof.json`:
    `exit_code=0`, `invocation_status=completed`
  - bundle `ablation_metrics.json` and `ablation_manifest.json` both record
    `completion_proof_present: true` and
    `completion_proof_filename: "exit_code_proof.json"` for the lineage-promoted
    baseline and the fresh ConvNeXt row.
- Run-level model manifest: `claim_boundary == "decision_support_append_only"`
  and the fresh row `pinn_hybrid_resnet_convnext_bottleneck` reports
  `row_status == "decision_support_append_only"`. No row is auto-promoted to
  `paper_grade` for this item.

## Residual Risks

- The ConvNeXt bottleneck has fewer parameters than the SRU-Net ResNet
  bottleneck at this width (`6.84M` vs `7.79M` total module params at
  `n_blocks=4`, `hybrid_downsample_steps=2`). The comparison is structural and
  parameter-count-honest; it is not normalized to match parameter count by
  design. Any future row should cite the row-local `model_manifest`
  parameter count.
- LayerScale init `0.1` was chosen so that block family is the only intended
  axis. The canonical tiny-ConvNeXt LayerScale (`1e-6`) initialization is
  deferred and is the most plausible follow-up to test whether the
  Δ phase_frc50 gain is robust across LayerScale conventions.
- The baseline `pinn_hybrid_resnet` row is held by lineage; its row-local
  metrics in this bundle are read-only references to the immutable authority.
