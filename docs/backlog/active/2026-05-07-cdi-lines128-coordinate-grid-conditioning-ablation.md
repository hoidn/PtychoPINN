---
priority: 20
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
        Path("docs/backlog/done/2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_no_refiner_row_rerun_summary.md"),
        Path("scripts/studies/grid_lines_compare_wrapper.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
        Path("ptycho_torch/generators/hybrid_resnet.py"),
        Path("ptycho_torch/generators/ffno.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing coordinate-grid conditioning inputs: {missing}")
    print("coordinate-grid conditioning inputs present")
    PY
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or ffno or coordinate or grid"
  - pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or ffno or coordinate or grid"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - CNS authored FFNO uses explicit unit-grid position channels, while CDI SRU-Net and FFNO currently do not.
  - A same-contract Lines128 ablation can test whether fixed spatial coordinates help or hurt CDI reconstructions.
  - This is append-only Phase 3 CDI mechanism evidence and should not rerun completed baseline rows.
---

# Backlog Item: Lines128 CDI Coordinate-Grid Conditioning Ablation

## Objective

- Add a controlled Lines128 CDI ablation that concatenates fixed spatial
  coordinate channels to SRU-Net and FFNO inputs.
- Test whether explicit position information improves the CDI setting where the
  current repo-local CDI FFNO and SRU-Net paths do not append a grid.
- Keep the comparison limited to same-contract SRU-Net and corrected pure FFNO
  baselines.

## Scope

- Preserve the locked Lines128 contract:
  - `N=128`, `gridsize=1`, synthetic grid-lines, `set_phi=True`;
  - Run1084 fixed probe with `probe_scale_mode=pad_extrapolate` and
    `probe_smoothing_sigma=0.5`;
  - fixed train/test split, seed, epoch budget, scheduler, loss, output mode,
    metric schema, and fixed visual sample policy.
- Add a config-controlled input path that appends two deterministic coordinate
  channels:
  - `y` coordinate on `[0, 1]`;
  - `x` coordinate on `[0, 1]`.
- Use the same coordinate convention as the PDEBench authored-FFNO adapter
  unless the implementation plan records a reviewed reason to choose a
  centered convention.
- Required fresh rows:
  - `pinn_hybrid_resnet_grid_channels`;
  - `pinn_ffno_grid_channels`.
- Do not combine this with probe-channel conditioning in the same row. The
  effect of position channels must be isolated.

## Outputs

- Item-local artifacts under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-07-cdi-lines128-coordinate-grid-conditioning-ablation/`.
- Row-local invocation/config/history/metrics/checkpoint/reconstruction/visual
  artifacts for the two fresh grid-conditioned rows.
- Durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_coordinate_grid_conditioning_ablation_summary.md`.
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`;
  - `docs/studies/index.md`.

## Completion Gate

- The fresh row configs must prove that coordinate channels were appended to
  the learned input, with channel ordering and normalization recorded.
- The summary must compare grid-conditioned SRU-Net and FFNO against the
  corresponding unconditioned same-contract baseline rows by lineage.
- The item must explicitly state whether coordinate channels helped SRU-Net,
  FFNO, both, or neither, and whether any gain is large enough to justify a
  later table-refresh item.

## Notes For Reviewer

- Reject implementations that silently add coordinate channels to default CDI
  rows or to unrelated model families.
- Reject rows that mix grid conditioning with probe conditioning, architecture
  changes, or training-contract changes.
- Reject claims that import CNS authored-FFNO coordinate-grid conclusions into
  CDI without the fresh CDI ablation rows.
