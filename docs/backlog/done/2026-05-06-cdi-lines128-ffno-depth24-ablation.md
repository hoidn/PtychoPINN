---
priority: 7
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_preflight.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json"),
        Path("ptycho_torch/generators/ffno.py"),
        Path("ptycho_torch/generators/ffno_bottleneck.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
        Path("scripts/studies/grid_lines_compare_wrapper.py"),
        Path("scripts/studies/lines128_paper_benchmark.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing CDI FFNO depth-24 ablation inputs: {missing}")
    print("CDI FFNO depth-24 ablation inputs present")
    PY
  - pytest -q tests/torch/test_generator_registry.py -k "ffno"
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"
  - pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun
  - 2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The corrected Lines128 CDI FFNO authority must use the no-refiner four-block `pinn_ffno` generator row.
  - Recent SRU-Net/FFNO encoder discussions exposed confusion between the CDI four-block FFNO default and deeper 24-layer FFNO profiles used elsewhere.
  - A direct Lines128 depth ablation is needed before treating "24-layer FFNO" as a better or worse CDI design choice.
  - This belongs to the later full-results wave and should run only after the four-block no-refiner rows have refreshed near-term paper figures, images, and metrics.
---

# Backlog Item: Run 24-Block CDI FFNO Lines128 Depth Ablation

## Objective

- Add one CDI Lines128 FFNO depth-ablation row with `fno_blocks=24`.
- Compare its metrics directly against the corrected no-refiner default
  `pinn_ffno` row with `fno_blocks=4`.
- Keep this as the later full-results wave; do not delay the four-block
  no-refiner table/figure refresh on this run.
- Keep the comparison isolated to FFNO depth; do not change the dataset,
  probe, loss, width, modes, CNN-refiner count, scheduler, seed, or metric
  definitions.

## Scope

- Use the corrected no-refiner Lines128 CDI FFNO rerun as the source of truth
  for the default four-block comparator:
  - row id: `pinn_ffno`;
  - `fno_blocks=4`;
  - `fno_modes=12`;
  - `fno_width=32`;
  - `fno_cnn_blocks=0`.
- Add one explicit new row id, preferably `pinn_ffno_depth24`, whose only
  architectural difference from `pinn_ffno` is:
  - `fno_blocks=24`.
- Preserve all other fixed CDI contract fields:
  - `N=128`, `gridsize=1`, synthetic grid-lines;
  - Run1084 fixed-probe lineage, `probe_scale_mode=pad_extrapolate`,
    `probe_smoothing_sigma=0.5`, `set_phi=True`;
  - `seed=3`, `nimgs_train=2`, `nimgs_test=2`;
  - `nphotons=1e9`, `probe_mask=off`;
  - `40` epochs, batch `16`, Adam `2e-4`;
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-4)`;
  - `torch_loss_mode=mae`, `torch_mae_pred_l2_match_target=off`,
    `torch_output_mode=real_imag`;
  - fixed sample ids and shared visual-scale policy unchanged.
- Run only the new depth-24 row after the corrected default no-refiner row and
  the four-block no-refiner table/figure refresh land. Reuse that corrected
  four-block row by lineage; do not compare against the historical
  `fno_cnn_blocks=2` local-refiner proxy row.
- If the existing wrapper cannot express a second FFNO row with a distinct row
  id and row-local `fno_blocks` override, add a narrow append-only row
  extension rather than mutating the default `pinn_ffno` row.

## Required Interpretation

- This is a CDI FFNO depth ablation, not an SRU-Net encoder ablation and not a
  CNS authored-FFNO comparison.
- The result should answer one question: under the fixed Lines128 CDI contract,
  does increasing the end-to-end CDI FFNO stack from four blocks to 24 blocks
  improve, degrade, or leave unchanged amplitude and phase reconstruction
  metrics?
- Report parameter count and runtime/provenance if available so any metric
  improvement is not interpreted without compute context.
- Keep the default `pinn_ffno` label reserved for the four-block authority.
  The 24-block row must have a distinct row id and display label such as
  `FFNO-24 + PINN`.
- Do not promote the 24-block row into the main paper table automatically. Final
  paper-facing promotion is owned by
  `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh` after both PINN and
  supervised depth-24 rows are available.

## Outputs

- Row-local artifacts under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-ffno-depth24-ablation/`.
- Metrics, history, invocation, config, checkpoint, reconstruction arrays, and
  visual comparisons for `pinn_ffno_depth24`.
- A derived comparison table containing at least:
  - corrected no-refiner `pinn_ffno` (`fno_blocks=4`, `fno_cnn_blocks=0`,
    reused by lineage);
  - `pinn_ffno_depth24` (`fno_blocks=24`, fresh row).
- A concise durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md`
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`;
  - `docs/studies/index.md`.

## Completion Gate

- The depth-24 row must complete under the same Lines128 CDI contract as the
  corrected no-refiner default four-block row, except for `fno_blocks=24`.
- The summary must include side-by-side values for the standard CDI metrics:
  amplitude MAE/MSE/PSNR/SSIM/MS-SSIM/FRC50 and phase
  MAE/MSE/PSNR/SSIM/MS-SSIM/FRC50.
- The summary must explicitly record whether the default four-block comparator
  was reused by lineage or rerun for a justified reason.
- Any failure to instantiate or train the 24-block row must be row-local and
  precise; it should not trigger reruns of existing CDI table rows.

## Notes For Reviewer

- Reject implementations that overwrite the default `pinn_ffno` row or change
  its meaning from four blocks to 24 blocks.
- Reject implementations that compare the 24-block row against a different
  probe, seed, loss, image count, epoch count, scheduler, or output mode.
- Reject implementations that change `fno_modes`, `fno_width`, or set
  `fno_cnn_blocks` to anything other than `0` while claiming this isolates
  depth.
- Reject summaries that describe the 24-block CDI FFNO as the authored CNS
  FFNO baseline or as an SRU-Net encoder variant.
