# CDI Lines128 FFNO Depth-24 Ablation Execution Plan

## Goal

Run one append-only CDI Lines128 FFNO row with `fno_blocks=24`, then compare it
against the corrected no-refiner default `pinn_ffno` row with `fno_blocks=4`
under the same fixed Lines128 contract.

## Non-Negotiable Contract

- New row id: `pinn_ffno_depth24` unless an implementation-local naming
  constraint requires a clearer equivalent.
- Baseline comparator: reuse corrected no-refiner `pinn_ffno` by lineage; do
  not compare against the historical local-refiner proxy row.
- Only intended row difference:
  - default: `fno_blocks=4`;
  - new row: `fno_blocks=24`.
- Hold fixed:
  - `N=128`, `gridsize=1`, synthetic grid-lines;
  - Run1084 fixed probe, `probe_scale_mode=pad_extrapolate`,
    `probe_smoothing_sigma=0.5`, `set_phi=True`;
  - `seed=3`, `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e9`;
  - `torch_epochs=40`, `torch_batch_size=16`, `torch_learning_rate=2e-4`;
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-4)`;
  - `torch_loss_mode=mae`, `torch_mae_pred_l2_match_target=false`,
    `torch_output_mode=real_imag`;
  - `fno_modes=12`, `fno_width=32`, `fno_cnn_blocks=0`.

## Implementation Tasks

1. Confirm the default FFNO authority.
   - Read `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md`.
   - Locate the corrected no-refiner `pinn_ffno` metrics/config/run root from
     `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`.
   - Record the exact baseline source path in the new summary.

2. Add a row-local depth override without mutating `pinn_ffno`.
   - Prefer extending the existing compare wrapper or Lines128 paper benchmark
     helper so it can run a second FFNO row id with `fno_blocks=24`.
   - Keep `pinn_ffno` permanently bound to `fno_blocks=4`.
   - Add tests proving the default row remains four blocks and the depth-24 row
     writes `fno_blocks=24` into invocation/config artifacts.

3. Run only the new row.
   - Use an item-local root:
     `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/`.
   - Produce invocation, config, history, metrics, checkpoint, reconstruction
     arrays, visual comparison PNGs, and row provenance for `pinn_ffno_depth24`.
   - Do not overwrite the completed Lines128 table root.

4. Publish append-only comparison artifacts.
   - Create a two-row comparison table and JSON summary containing:
     - reused corrected no-refiner `pinn_ffno`, `fno_blocks=4`,
       `fno_cnn_blocks=0`;
     - fresh `pinn_ffno_depth24`, `fno_blocks=24`, `fno_cnn_blocks=0`.
   - Include amplitude and phase MAE, MSE, PSNR, SSIM, MS-SSIM, FRC50,
     parameter count, and runtime/provenance fields where available.
   - Preserve source paths for both rows.

5. Update discoverability.
   - Write
     `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`.
   - Update:
     - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
     - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
     - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`;
     - `docs/studies/index.md`.

## Verification

- `pytest -q tests/torch/test_generator_registry.py -k "ffno"`
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`
- `python -m compileall -q ptycho_torch scripts/studies`

Also run a row-local artifact audit that fails if:

- `pinn_ffno_depth24` does not report `fno_blocks=24`;
- the reused default `pinn_ffno` source does not report `fno_blocks=4` and
  `fno_cnn_blocks=0`;
- any fixed contract field differs outside the allowed `fno_blocks` change.

## Review Rules

- Reject any implementation that relabels `pinn_ffno` as 24-block FFNO.
- Reject any implementation that reruns completed CDI table rows without a
  documented artifact-corruption reason.
- Reject any comparison that mixes this row with CNS authored-FFNO or SRU-Net
  encoder-variant claims.
