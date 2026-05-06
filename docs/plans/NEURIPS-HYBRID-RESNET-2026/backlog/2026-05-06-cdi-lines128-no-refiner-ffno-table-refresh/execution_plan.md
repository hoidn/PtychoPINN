# CDI Lines128 Four-Block No-Refiner FFNO Table/Figure Refresh Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create worktrees.

**Goal:** Replace near-term manuscript-facing CDI FFNO proxy rows with corrected four-block no-refiner FFNO rows after the cheap reruns complete.

**Architecture:** Treat the four-block no-refiner PINN and supervised FFNO reruns as row-level replacements by lineage for the immediate paper metrics/figure/image refresh. Rebuild tables and indexes without changing non-FFNO metrics, keep the historical `fno_cnn_blocks=2` evidence visible only as proxy context, and leave depth-24 rows to the later full-results wave.

**Tech Stack:** Existing paper result refresh scripts, JSON/CSV/TeX assets, Markdown evidence indexes.

---

## Tasks

- [ ] Audit corrected no-refiner FFNO row roots and verify `fno_blocks=4` and `fno_cnn_blocks=0` in config/invocation.
- [ ] Regenerate CDI metrics and objective-control tables with corrected four-block FFNO rows.
- [ ] Regenerate CDI figure/image source manifests or panels that currently consume the historical FFNO-local-refiner proxy row.
- [ ] Regenerate model-config and efficiency assets with corrected parameter/runtime/provenance fields.
- [ ] Update `model_variant_index.json` and `ablation_index.json` so current paper-facing variants point to corrected roots.
- [ ] Update `evidence_matrix.md`, `paper_evidence_index.md`, `docs/studies/index.md`, and manuscript references that consume the old FFNO table rows.
- [ ] Preserve old rows as `FFNO-local proxy` historical context with artifact roots and metrics intact.

## Verification

- `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
- CSV parse smoke check for regenerated CSV files.
- Targeted grep that fails if a manuscript-facing old artifact root is still labeled canonical `FFNO + PINN` or `FFNO + supervised`.
- Targeted audit that fails if this interim refresh requires or substitutes any `fno_blocks=24` row.
