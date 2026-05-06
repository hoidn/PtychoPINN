---
priority: 38
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-srunet-convnext-bottleneck-ablation/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
        Path("docs/backlog/done/2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
        Path("scripts/studies/grid_lines_compare_wrapper.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
        Path("ptycho_torch/generators/hybrid_resnet.py"),
        Path("ptycho_torch/generators/resnet_components.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing SRU-Net ConvNeXt bottleneck ablation inputs: {missing}")
    print("SRU-Net ConvNeXt bottleneck ablation inputs present")
    PY
  - pytest -q tests/torch/test_fno_generators.py -k "hybrid_resnet or convnext"
  - pytest -q tests/torch/test_grid_lines_torch_runner.py -k "hybrid_resnet or convnext"
  - pytest -q tests/test_grid_lines_compare_wrapper.py -k "hybrid_resnet or convnext"
  - python -m compileall -q scripts/studies ptycho_torch
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-04-30-cdi-lines128-hybrid-resnet-skip-residual-ablation
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - This is a single-axis SRU-Net bottleneck-family ablation for the fixed Lines128 CDI benchmark.
  - It should run after the existing skip/residual ablation context and remain separate from the SRU-Net encoder branch/objective ablation.
  - It adds only a ConvNeXt-style bottleneck row and must not rerun completed Lines128 comparator rows.
---

# Backlog Item: Lines128 SRU-Net ConvNeXt Bottleneck Ablation

## Objective

- Add and run a ConvNeXt-style bottleneck replacement for SRU-Net on the fixed
  Lines128 CDI benchmark, then compare it against the completed
  `pinn_hybrid_resnet` / SRU-Net row by lineage.

## Scope

- Implement a narrow SRU-Net variant such as
  `hybrid_resnet_convnext_bottleneck`.
- Preserve the existing SRU-Net encoder, downsampling, decoder, skip policy,
  bottleneck width, block count, dataset, split, probe preprocessing, seed
  policy, epoch budget, scheduler, output mode, loss, metric schema, fixed
  visual samples, and shared visual scales.
- Replace only the current `ResnetBottleneck` body with a ConvNeXt-style
  constant-resolution bottleneck:
  - depthwise spatial convolution;
  - normalization;
  - pointwise channel expansion/projection;
  - GELU;
  - residual LayerScale.
- Use the current SRU-Net LayerScale convention for the first row so the result
  isolates block family rather than initialization policy.
- Launch only the missing `pinn_hybrid_resnet_convnext_bottleneck` row.
- Publish a concise append-only summary with metrics, visuals, provenance, and
  a direct comparison to the completed SRU-Net bottleneck row.

## Notes for Reviewer

- Reject implementations that rerun completed CNN, FNO, FFNO, U-NO, spectral
  bottleneck, or baseline SRU-Net rows just to assemble the comparison.
- Reject implementations that change encoder branch gates, skip connections,
  decoder family, loss, probe, schedule, or data contract in the ConvNeXt row.
- Keep this separate from the SRU-Net branch/objective ablation. That item
  tests encoder branch necessity and supervised objective controls; this one
  tests bottleneck block family.
- Treat canonical tiny ConvNeXt LayerScale initialization as follow-up context
  unless a later reviewed plan explicitly adds a second row.
