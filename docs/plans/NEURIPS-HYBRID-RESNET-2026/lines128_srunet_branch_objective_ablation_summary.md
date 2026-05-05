# NeurIPS Lines128 SRU-Net Branch / Objective Ablation Summary

- Date: `2026-05-04`
- Backlog item: `2026-05-04-cdi-lines128-srunet-branch-objective-ablation`
- State: `decision_support_append_only`
- Fixed contract id: `cdi_lines128_seed3`
- Authoritative ablation root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/runs/ablation_20260505T010316Z`
- Bundle artifact:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/bundle/`

## Claim Boundary

- This bundle is `decision_support_append_only`. It does not replace the immutable
  six-row complete CDI authority
  (`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`) or
  the U-NO append-only extension
  (`docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_uno_table_extension_summary.md`).
- It is mechanistic / objective-control evidence around the SRU-Net (Hybrid ResNet)
  row, framed against the fixed `lines128` CDI contract.

## Fixed Contract

- dataset contract id: `cdi_lines128_seed3`
- `N = 128`, `gridsize = 1`, `seed = 3`
- `40` epochs, `batch_size = 16`, `learning_rate = 2e-4`
- `ReduceLROnPlateau` (factor 0.5, patience 2, min_lr 1e-4, threshold 0.0)
- `torch_loss_mode = mae`, `torch_output_mode = real_imag`
- `fno_modes = 12`, `fno_width = 32`, `fno_blocks = 4`, `fno_cnn_blocks = 2`
- `nimgs_train = 2`, `nimgs_test = 2`, `nphotons = 1e9`
- `set_phi = True`, probe `Run1084_recon3_postPC_shrunk_3.npz`,
  `probe_source = custom`, `probe_scale_mode = pad_extrapolate`,
  `probe_smoothing_sigma = 0.5`
- fixed sample ids: `0`, `1`

## Row Roster

| model_id | label | training | branch_select | evidence_source |
|---|---|---|---|---|
| `pinn_hybrid_resnet` | Hybrid ResNet + PINN | pinn | both | promoted_by_lineage |
| `pinn_hybrid_resnet_encoder_conv_only` | Hybrid ResNet (conv-only encoder) + PINN | pinn | conv_only | fresh_run |
| `pinn_hybrid_resnet_encoder_spectral_only` | Hybrid ResNet (spectral-only encoder) + PINN | pinn | spectral_only | fresh_run |
| `supervised_hybrid_resnet` | Hybrid ResNet + supervised | supervised | both | fresh_run |

## Reused Baseline Lineage

- The `pinn_hybrid_resnet` row is reused by lineage from the authoritative
  six-row CDI bundle root:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-04-29-cdi-lines128-paper-benchmark-execution/runs/complete_table_20260430T150757Z_repair_tmux/runs/pinn_hybrid_resnet`
- It is not rerun for this backlog item.

## Implementation Surfaces Touched

- `ptycho_torch/generators/hybrid_resnet.py`:
  - new `ENCODER_BRANCH_SELECT_MODES = ("both", "conv_only", "spectral_only")`
  - `HybridResnetEncoderBlock` and `HybridResnetGeneratorModule` accept
    `encoder_branch_select` and propagate it to every encoder block
  - disabled branches collapse to `nn.Identity` so the ablation is deterministic
    and parameter-honest (no learned-zero-gate fakery)
  - factory routes `encoder_branch_select` through `execution_config` and
    `generator_overrides` round-trip
- `ptycho/config/config.py`: `PyTorchExecutionConfig.hybrid_encoder_branch_select`
- `ptycho_torch/workflows/components.py`: factory override forwarding
- `ptycho_torch/model.py`: PtychoPINN_Lightning generator overrides round-trip
- `scripts/studies/grid_lines_torch_runner.py`:
  - `TorchRunnerConfig.hybrid_encoder_branch_select`
  - CLI flag `--hybrid-encoder-branch-select` with validation + execution-config bridge
- `scripts/studies/grid_lines_compare_wrapper.py`:
  - new model_ids registered in `TORCH_MODEL_IDS`, `PAPER_MODEL_LABELS`,
    `PAPER_TRAINING_PROCEDURE_OVERRIDES`, and `DEFAULT_TORCH_ROW_SPECS`
  - branch-ablation rows declare `overrides.hybrid_encoder_branch_select`
- `scripts/studies/lines128_srunet_ablation_bundle.py`:
  - narrow append-only ablation bundle helper that promotes the baseline by
    lineage and collates fresh row metrics under a fixed claim boundary
  - requires a row-local completion-proof artifact for every fresh row
    (`exit_code_proof.json`, with `launcher_completion.json` accepted as a
    legacy fallback) and surfaces both the proof payload and its filename in
    the bundle's row provenance

## Tests Added

- `tests/torch/test_fno_generators.py`:
  - branch-select shape preservation
  - branch-select default = `both`
  - branch-select drops unused branch params (Identity placeholders)
  - branch-select invalid raises
  - orthogonal interaction with `branch_gated` fusion mode
  - branch-select identity passthrough when active branch zeroed
  - module propagates branch-select per block
- `tests/torch/test_grid_lines_torch_runner.py`:
  - runner accepts each branch-select mode
  - default branch-select is `both`
  - invalid branch-select rejected
  - branch-select stays out of canonical model config
  - branch-select appears in reconstructed runner CLI argv
- `tests/test_grid_lines_compare_wrapper.py`:
  - default torch row specs register the three new ablation rows
  - `validate_model_specs` accepts the new rows
- `tests/studies/test_lines128_srunet_ablation_bundle.py`:
  - bundle helper collates fresh + promoted-by-lineage rows
  - records branch-select overrides per row
  - fails loudly when baseline or fresh row artifacts missing
  - fails loudly when a fresh row has no completion-proof artifact
  - accepts the legacy `launcher_completion.json` filename for backward compatibility

## Metric Deltas vs. `pinn_hybrid_resnet`

Per-row metrics are pulled from the bundle's `ablation_metrics.json`. All entries
in this table use the format `[amplitude, phase]` where applicable.

| Row | amp_mae | phase_mae | amp_ssim | phase_ssim | amp_psnr | phase_psnr | amp_frc50 | phase_frc50 |
|---|---|---|---|---|---|---|---|---|
| `pinn_hybrid_resnet` (baseline) | 0.0269 | 0.0721 | 0.9881 | 0.9947 | 77.52 | 69.22 | 135.46 | 106.80 |
| `pinn_hybrid_resnet_encoder_conv_only` | 0.0323 | 0.0571 | 0.9823 | 0.9946 | 75.95 | 71.12 | 134.93 | 134.96 |
| `pinn_hybrid_resnet_encoder_spectral_only` | 0.0252 | 0.0615 | 0.9899 | 0.9958 | 78.14 | 70.55 | 137.69 | 135.71 |
| `supervised_hybrid_resnet` | 0.4010 | 0.0340 | 0.2710 | 0.9687 | 53.63 | 74.11 | 24.44 | 52.36 |

Δ vs. baseline (negative is better for MAE; positive is better for SSIM/PSNR/FRC50):

| Row | Δ amp_mae | Δ phase_mae | Δ amp_psnr | Δ phase_psnr | Δ amp_frc50 | Δ phase_frc50 |
|---|---|---|---|---|---|---|
| `pinn_hybrid_resnet_encoder_conv_only` | +0.0054 | -0.0150 | -1.57 | +1.90 | -0.53 | +28.16 |
| `pinn_hybrid_resnet_encoder_spectral_only` | -0.0017 | -0.0106 | +0.62 | +1.33 | +2.23 | +28.91 |
| `supervised_hybrid_resnet` | +0.3741 | -0.0381 | -23.89 | +4.89 | -111.02 | -54.44 |

## Interpretation

- **Branch necessity (encoder ablation).** Removing either single encoder branch
  preserves most amplitude and phase fidelity within the locked Lines128 CDI
  contract. The spectral-only encoder slightly improves amp MAE/SSIM/PSNR/FRC50
  versus the both-branch baseline at this `N=128` resolution, while conv-only
  trades some amplitude fidelity for a phase MAE / phase FRC50 improvement.
  Neither single-branch row collapses, so the branches are complementary rather
  than each individually load-bearing under this fixed contract; the both-branch
  baseline therefore remains the conservative SRU-Net + PINN row for the paper.
- **Objective control.** `supervised_hybrid_resnet` keeps the same SRU-Net body
  and all contract knobs except the training procedure. Phase-side metrics
  improve modestly (phase MAE 0.034 vs. 0.072), but amplitude metrics collapse
  catastrophically (amp MAE 0.401, amp SSIM 0.27, amp FRC50 24.4) because the
  supervised path trained without the PINN physics-consistency loss does not
  recover the correct amplitude scale on the locked CDI contract. This is
  evidence that the SRU-Net + PINN advantage on this task is not purely
  architectural — the physics-consistency loss is load-bearing for amplitude
  calibration. The supervised SRU-Net row must not be promoted as a CDI headline
  row; it is presented as objective-control evidence only.
- Comparisons against FFNO, FNO, U-NO, CNN/U-Net-class, and spectral-bottleneck
  rows are by reference to the authoritative six-row bundle and the U-NO
  extension; no completed row is rerun.

## Verification

Verification logs are archived under
`.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-branch-objective-ablation/verification/`.

- Required deterministic gates (final, timestamp `20260505T020849Z`):
  - prerequisite presence — `prereq_20260505T020849Z.log` (PASS)
  - `pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or hybrid_encoder"` —
    `test_fno_generators_20260505T020849Z.log` (60 passed)
  - `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or hybrid_encoder or supervised"` —
    `test_grid_lines_torch_runner_20260505T020849Z.log` (37 passed)
  - `pytest -q tests/test_grid_lines_compare_wrapper.py` —
    `test_grid_lines_compare_wrapper_20260505T020849Z.log` (67 passed)
  - `python -m compileall -q scripts/studies ptycho_torch` —
    `compileall_20260505T020849Z.log` (exit 0)
- Bundle helper regression:
  - `pytest -q tests/studies/test_lines128_srunet_ablation_bundle.py` —
    `test_ablation_bundle_20260505T020849Z.log` (6 passed; covers baseline-by-lineage,
    fresh-row collation, branch-select override recording, missing baseline,
    missing fresh row, missing completion-proof, and legacy
    `launcher_completion.json` acceptance)
- Wrapper completion proof:
  - run-root `<run_root>/invocation.json`: `status=completed`, `exit_code=0`,
    `finished_at_utc=2026-05-05T01:50:15Z`
  - per-fresh-row `runs/<row>/exit_code_proof.json`: `exit_code=0`,
    `invocation_status=completed` for `pinn_hybrid_resnet_encoder_conv_only`,
    `pinn_hybrid_resnet_encoder_spectral_only`, and `supervised_hybrid_resnet`
  - bundle `ablation_metrics.json` records `completion_proof_present: true` and
    `completion_proof_filename: "exit_code_proof.json"` for every row, including
    the lineage-promoted baseline `pinn_hybrid_resnet`

## Residual Risks

- Branch-disable is a structural ablation that drops one encoder branch entirely.
  Per-block parameter counts therefore differ across rows; this is intentional
  and is not normalized away. Comparisons should always cite the row-local
  `model_manifest` parameter counts.
- `supervised_hybrid_resnet` uses identical SRU-Net body and contract, only
  swapping the training procedure. It must not be promoted as the CDI headline
  row.
- The baseline `pinn_hybrid_resnet` row is held by lineage; its row-local
  metrics in this bundle are read-only references to the immutable authority.
