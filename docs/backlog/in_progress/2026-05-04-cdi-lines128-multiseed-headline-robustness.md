---
priority: 36
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-multiseed-headline-robustness/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-29-cdi-lines128-paper-benchmark-execution.md"),
        Path("docs/backlog/done/2026-04-30-cdi-lines128-uno-table-extension.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json"),
        Path("scripts/studies/grid_lines_compare_wrapper.py"),
        Path("scripts/studies/grid_lines_torch_runner.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing CDI multiseed robustness inputs: {missing}")
    print("CDI multiseed robustness inputs present")
    PY
  - pytest -q tests/studies/test_paper_results_refresh.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
  - python -m compileall -q scripts/studies ptycho_torch
prerequisites:
  - 2026-04-29-cdi-lines128-paper-benchmark-execution
  - 2026-04-30-cdi-lines128-uno-table-extension
related_roadmap_phases:
  - phase-3-cdi-anchor-regeneration
signals_for_selection:
  - Reviewer-style feedback identified single-seed CDI evidence as a main acceptance risk.
  - This item strengthens the existing headline Lines128 table without changing its dataset, split, or row contract.
  - It should run before optional mechanism ablations when compute is available, but it must not rerun completed seed roots whose contract audit passes.
---

# Backlog Item: Lines128 Multi-Seed Headline Robustness

## Objective

- Add a Phase 3 CDI evidence-strengthening pass that reports training-seed
  variability for the current headline `lines128` rows.

## Scope

- Reuse the completed `seed=3` rows by lineage after verifying their contract
  and provenance.
- Add only the missing seeds needed for a three-seed aggregate, unless a
  row-level blocker is recorded.
- Include the current headline CDI table rows:
  - `pinn_hybrid_resnet` / SRU-Net;
  - paired CDI `cnn` U-Net-class supervised and PINN rows;
  - `pinn_fno_vanilla`;
  - `pinn_ffno`;
  - `pinn_neuralop_uno`.
- Keep the dataset, split, probe preprocessing, epoch budget, scheduler, loss,
  output mode, metric schema, fixed visual sample IDs, and shared visual scales
  fixed across seeds.

## Required Interpretation

- Report this as training-seed robustness on the existing `lines128` split.
- Do not claim broader object-distribution robustness from this item.
- Preserve the existing single-seed row lineage in any manuscript or table
  refresh.

## Outputs

- Fresh missing-seed row artifacts under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-multiseed-headline-robustness/`
- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_multiseed_headline_robustness_summary.md`
- Per-seed and mean/std JSON, CSV, and TeX table payloads.
- Updates to the evidence matrix, model variant index, paper evidence index,
  and study index.

## Notes For Reviewer

- Reject plans that broaden this into expanded object families, noise sweeps,
  architecture ablations, or new baselines.
- Reject implementations that rerun completed seed roots without first auditing
  whether the existing artifact already satisfies the current contract.
- If a row cannot produce the requested seeds, require a row-level blocker and
  keep incomplete rows out of the aggregate mean/std table.
