# Paper Efficiency Table Summary

This summary records the repo-local efficiency table generated for the NeurIPS SRU-Net evidence package.
The table groups rows by benchmark and keeps runtime fields as provenance context unless an explicit throughput field exists.

## Outputs

- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.json`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.csv`
- `docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/paper_efficiency_table.tex`

## Row Counts

- CDI: 8 rows
- PDEBench CNS: 4 rows
- BRDT: 2 rows

## Runtime And Throughput Policy

- Parameter counts use `unique_trainable_params` from the model-configuration table when available; otherwise they fall back to existing row artifacts.
- Training/runtime fields keep their original source field names.
- Missing inference throughput is labeled `missing`; training runtime is not converted into throughput.
- Rows from different benchmarks are not ranked against each other.

## Superseded Context

- None.
