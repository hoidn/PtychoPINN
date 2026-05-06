---
priority: 36
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_fixedprobe_dataset_summary.md"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
        Path("scripts/studies/grid_lines_compare_wrapper.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing natural-patch expanded benchmark inputs: {missing}")
    print("natural-patch expanded benchmark inputs present")
    PY
  - pytest -q tests/studies/test_cdi_natural_patch_dataset.py tests/studies/test_cdi_natural_patch_benchmark.py
  - python -m compileall -q scripts/studies ptycho_torch
prerequisites:
  - 2026-05-04-cdi-natural-patch-fixedprobe-dataset
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-04-30-cdi-lines128-uno-table-extension
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - Expanded-object CDI evidence addresses the reviewer concern that the current CDI table uses too few objects.
  - This item depends on a frozen natural-patch fixed-probe dataset so model ranking cannot change the object distribution.
  - It should produce a standalone natural-patch CDI table rather than rewriting the existing lines128 authority.
---

# Backlog Item: CDI Natural-Patch Expanded Benchmark

## Queue Status

- Paused on 2026-05-06 at operator request after implementation review
  returned `REVISE`.
- Do not select this item in backlog-drain runs unless it is explicitly moved
  back to `docs/backlog/active/`.
- Last known state: recovered artifacts were non-authoritative and the
  benchmark launch did not complete cleanly.

## Objective

- Run the first expanded-object CDI benchmark on the locked
  `natural_patches128_fixedprobe_v1` dataset.

## Scope

- Consume the completed natural-patch dataset summary and manifests.
- Run the current headline CDI row family where supported:
  - SRU-Net / `pinn_hybrid_resnet`;
  - paired CDI `cnn` U-Net-class supervised and PINN rows;
  - `pinn_fno_vanilla`;
  - `pinn_ffno`;
  - `pinn_neuralop_uno`.
- Keep dataset split, fixed probe, training budget, metric schema, and visual
  policy locked before launch.
- Produce standalone natural-patch table and visual artifacts.

## Required Interpretation

- This item expands the CDI object distribution beyond `lines128`; it does not
  replace the `lines128` authority.
- Report the natural-image source and object-count cap with the result.
- Do not claim broader generalization than the locked dataset source supports.

## Outputs

- Fresh row artifacts under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-natural-patch-expanded-benchmark/`
- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_natural_patch_expanded_benchmark_summary.md`
- JSON, CSV, TeX, and visual comparison payloads.
- Updates to the evidence matrix, model variant index, paper evidence index,
  and study index.

## Notes For Reviewer

- Reject plans that regenerate the dataset or alter the split during benchmark
  execution.
- Reject plans that rerun existing `lines128` rows as part of this item.
- Reject plans that present this as multi-seed evidence unless a later checked
  item authorizes seed aggregation.
