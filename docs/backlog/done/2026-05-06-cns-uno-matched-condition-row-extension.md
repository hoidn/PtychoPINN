---
priority: 34
plan_path: docs/plans/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/execution_plan.md
check_commands:
  - |
    python - <<'PY'
    from pathlib import Path
    required = [
        Path("docs/backlog/done/2026-04-30-cdi-lines128-uno-generator-integration.md"),
        Path("docs/backlog/done/2026-05-04-cns-matched-condition-table-refresh.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_matched_condition_table_refresh_summary.md"),
        Path("docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics.json"),
        Path("ptycho_torch/generators/neuralop_uno.py"),
        Path("scripts/studies/pdebench_image128/run_config.py"),
        Path("scripts/studies/pdebench_image128/models.py"),
        Path("scripts/studies/run_pdebench_image128_suite.py"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(f"missing CNS U-NO row-extension inputs: {missing}")
    print("CNS U-NO row-extension inputs present")
    PY
  - pytest -q tests/studies/test_pdebench_image128_models.py -k "uno or profile or cns"
  - pytest -q tests/studies/test_pdebench_image128_runner.py -k "matched_condition or pdebench_cns or uno"
  - python -m compileall -q ptycho_torch scripts/studies
prerequisites:
  - 2026-04-30-cdi-lines128-uno-generator-integration
  - 2026-05-04-cns-matched-condition-table-refresh
related_roadmap_phases:
  - phase-2-pdebench-128x128-image-suite
signals_for_selection:
  - The current CNS matched-condition table has FFNO, SRU-Net*, FNO, and U-Net, but no U-NO row.
  - U-NO has already been integrated as a real NeuralOperator-backed generator for CDI; this item checks whether the same architecture family can be evaluated under the CNS image-suite runner.
  - The row should be append-only under the existing CNS matched-condition contract, not a replacement for the current four-row headline table.
---

# Backlog Item: Append U-NO Row To PDEBench CNS Matched-Condition Results

## Objective

- Add a single U-NO comparator row to the existing PDEBench CNS
  matched-condition result set, then republish the CNS table assets with the
  U-NO row appended by lineage.

## Scope

- Consume the current CNS matched-condition authority:
  - task: PDEBench `2d_cfd_cns`;
  - spatial shape: `128x128`;
  - `history_len=5`;
  - train/validation/test trajectories: `512 / 64 / 64`;
  - epochs: `40`;
  - batch size: `4`;
  - optimizer/loss/normalization: the existing CNS matched-condition training
    recipe, including MSE loss.
- Add one CNS profile for NeuralOperator U-NO, for example
  `neuralop_uno_cns_base`, only if the current image-suite runner cannot
  already instantiate an equivalent row.
- The CNS U-NO profile must use the existing NeuralOperator U-NO adapter or a
  narrowly scoped task-local wrapper around the same U-NO body. Record every
  shape adapter, channel mapping, padding/cropping policy, mode count, hidden
  width, block/depth setting, and dependency version in the row-local profile
  manifest.
- Run only the new U-NO row. Do not rerun FFNO, SRU-Net*, FNO, or U-Net just to
  append this row.
- Publish a derived CNS bundle that includes:
  - the existing four matched-condition rows by lineage reference;
  - the new U-NO row metrics and provenance;
  - JSON/CSV/TeX table assets with U-NO appended;
  - a short summary describing whether the U-NO row is eligible for the
    manuscript CNS table or should remain adjacent append-only context.

## Required Interpretation

- This is an append-only comparator row, not a new CNS headline-table
  selection policy.
- The current four-row CNS matched-condition table remains the baseline
  authority until this item completes and explicitly republishes a derived
  table.
- Keep the claim boundary bounded to the existing capped CNS contract. Do not
  promote this to full-training PDEBench evidence.
- Do not compare U-NO against CDI U-NO numbers; CDI and CNS have different
  data, objectives, and claim boundaries.
- If U-NO cannot be made shape-compatible with the CNS runner without changing
  the benchmark contract, stop with a clear blocker and do not substitute U-Net
  or another local-convolution model.

## Outputs

- Row-local artifacts under
  `.artifacts/NEURIPS-HYBRID-RESNET-2026/backlog/2026-05-06-cns-uno-matched-condition-row-extension/`.
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/pdebench_cns_uno_row_extension_summary.md`
- Derived table assets, with exact filenames chosen by the implementation plan,
  such as:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.json`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.csv`
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/pdebench_cns_matched_condition_metrics_plus_uno.tex`
- Updates to:
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/evidence_matrix.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/paper_evidence_index.md`;
  - `docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`;
  - `docs/studies/index.md`.

## Completion Gate

- The U-NO row must have the same CNS matched-condition contract as the current
  four rows, or the item must close as blocked with a precise shape/dependency
  reason.
- The derived table must preserve source paths and claim boundaries for all
  existing rows and identify U-NO as an appended row.
- Any manuscript-facing recommendation must say whether to replace
  `tables/pdebench_cns_matched_condition_metrics.tex` with the derived
  plus-U-NO table or keep U-NO as adjacent context.

## Notes For Reviewer

- Reject implementations that rerun completed CNS baseline rows unnecessarily.
- Reject implementations that mix history lengths, caps, epoch budgets, losses,
  or normalization protocols in the same row-ranking table.
- Reject implementations that label U-NO as U-Net, CNN, FNO, or SRU-Net for
  convenience.
- Reject implementations that hide shape adapters or NeuralOperator dependency
  versions from the row-local profile manifest.
