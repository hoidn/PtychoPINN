---
priority: 8
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
        Path("docs/backlog/done/2026-05-06-cdi-lines128-ffno-depth24-ablation.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_ablation_summary.md"),
        Path("ptycho_torch/generators/ffno.py"),
        Path("scripts/studies/grid_lines_compare_wrapper.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing supervised FFNO depth-24 inputs: {missing}")
    print("supervised FFNO depth-24 inputs present")
    PY
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"
  - pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-05-06-cdi-lines128-ffno-depth24-ablation
  - 2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - The cheap four-block no-refiner supervised row must refresh near-term paper assets before depth-24 work runs.
  - A final depth-24 paper update needs a training-procedure companion, not only the PINN depth-24 row.
  - This item belongs to the later full-results wave and should run after the depth-24 PINN row.
---

# Backlog Item: Rerun Supervised CDI Lines128 FFNO With 24 Blocks And No Refiner

## Objective

- Add one supervised Lines128 CDI FFNO row with `fno_blocks=24` and
  `fno_cnn_blocks=0`.
- Compare it against the corrected four-block no-refiner `supervised_ffno` row
  by lineage.
- Provide the supervised companion required before any final depth-24 paper
  refresh can replace the four-block no-refiner interim FFNO rows.

## Scope

- Use the same fixed Lines128 supervised-extension contract as
  `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`, except:
  - `fno_blocks=24`;
  - row id must be distinct, preferably `supervised_ffno_depth24`.
- Hold fixed:
  - `N=128`, `gridsize=1`, synthetic grid-lines, Run1084 fixed probe;
  - `probe_scale_mode=pad_extrapolate`, `probe_smoothing_sigma=0.5`,
    `set_phi=True`;
  - `seed=3`, `nimgs_train=2`, `nimgs_test=2`, fixed sample ids;
  - `40` epochs, batch `16`, Adam `2e-4`;
  - `ReduceLROnPlateau(factor=0.5, patience=2, threshold=0.0, min_lr=1e-4)`;
  - `torch_loss_mode=mae`, `torch_output_mode=real_imag`;
  - `fno_modes=12`, `fno_width=32`, `fno_cnn_blocks=0`.
- Run only the new supervised depth-24 row. Do not rerun the four-block
  no-refiner supervised row or any non-FFNO table rows.

## Outputs

- Item-local artifacts under
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cdi-lines128-supervised-ffno-depth24-no-refiner-rerun/`.
- Row-local invocation/config/history/metrics/checkpoint/reconstruction/visual
  artifacts for `supervised_ffno_depth24`.
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_depth24_no_refiner_summary.md`.
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`;
  - `docs/studies/index.md`.

## Completion Gate

- The new row must record `fno_blocks=24` and `fno_cnn_blocks=0` in invocation
  and row-local config artifacts.
- The summary must compare against the corrected four-block no-refiner
  `supervised_ffno` row by lineage and must not compare against historical
  `fno_cnn_blocks=2` proxy rows as canonical evidence.
- Any failure must be row-local and must not trigger reruns of non-FFNO rows.

## Notes For Reviewer

- Reject implementations that mutate the default `supervised_ffno` row from
  four blocks to 24 blocks.
- Reject comparisons that mix different probes, losses, seeds, sample ids,
  scheduler settings, or output modes.
- Reject summaries that promote depth-24 into final paper tables directly; the
  final paper update is owned by
  `2026-05-06-cdi-lines128-ffno-depth24-final-paper-refresh`.
