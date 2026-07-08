# Seed Plan: Lines128 Multi-Seed Headline Robustness

## Objective

Add a Phase 3 evidence-strengthening pass for the headline `lines128` CDI
rows by measuring seed-to-seed variability under the already locked benchmark
contract.

## Scope

- Start from the completed `lines128` CDI paper benchmark and U-NO table
  extension authorities.
- Reuse the completed `seed=3` rows by lineage after auditing their config,
  metric schema, split, and artifact provenance.
- Add missing training-seed rows for the headline table, targeting three total
  seeds per row unless a row-level blocker is recorded:
  - `pinn_hybrid_resnet` / SRU-Net;
  - paired CDI `cnn` U-Net-class supervised and PINN rows;
  - `pinn_fno_vanilla`;
  - `pinn_ffno`;
  - `pinn_neuralop_uno`.
- Keep fixed across seeds: dataset, object/probe split, probe preprocessing,
  epoch budget, scheduler, loss, output mode, metric schema, fixed visual
  sample IDs, and shared visual scales.
- Report seed aggregation as training-seed robustness on the existing
  `lines128` object split. Do not present it as expanded object-distribution
  evidence.

## Non-Goals

- Do not expand the object family, number of train/test objects, noise regime,
  probe model, resolution, or CDI task in this item.
- Do not rerun completed seed roots whose contract and provenance audit pass.
- Do not add new architecture variants, objective controls, or ablations.
- Do not rewrite the authoritative single-seed table root; publish the
  multi-seed result as an append-only robustness table that references the
  existing row lineage.

## Expected Artifacts

- Fresh row roots under:
  `.artifacts/work/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-04-cdi-lines128-multiseed-headline-robustness/`
- A durable summary:
  `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_multiseed_headline_robustness_summary.md`
- Machine-readable metrics with per-seed values plus mean and standard
  deviation:
  - JSON;
  - CSV;
  - TeX table fragment.
- Updated discoverability surfaces:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`
  - `docs/studies/index.md`
- Optional manuscript refresh only after every included row has either a
  complete seed set or an explicit row-level blocker.

## Verification Commands

```bash
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
pytest -q tests/studies/test_paper_results_refresh.py tests/test_grid_lines_compare_wrapper.py tests/torch/test_grid_lines_torch_runner.py
python -m compileall -q scripts/studies ptycho_torch
```
