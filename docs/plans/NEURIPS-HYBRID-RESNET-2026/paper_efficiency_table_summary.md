# Paper Efficiency Table Summary

This summary records the repo-local efficiency table generated for the NeurIPS Hybrid ResNet evidence package.
The table groups rows by benchmark and keeps runtime fields as provenance context unless an explicit throughput field exists.

## Outputs

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`

## Row Counts

- Synthetic CDI: 8 rows
- PDEBench CNS: 4 rows
- BRDT: 2 historical secondary-context rows

## Runtime And Throughput Policy

- Parameter counts use `unique_trainable_params` from the model-configuration table when available; otherwise they fall back to existing row artifacts.
- Training/runtime fields keep their original source field names.
- Missing inference throughput is labeled `missing`; training runtime is not converted into throughput.
- Rows from different benchmarks are not ranked against each other.

## Superseded Context

- BRDT: the current 40-epoch FFNO row is a historical local-refiner proxy, not
  a pure FFNO-paper-stack result. Regenerate the BRDT efficiency rows after
  `2026-05-06-brdt-corrected-ffno-40ep-rerun` if the manuscript keeps a BRDT
  FFNO comparison.
