# CDI Lines128 No-Refiner FFNO Table Refresh Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create worktrees.

**Goal:** Replace manuscript-facing CDI FFNO proxy rows with corrected no-refiner FFNO rows after the reruns complete.

**Architecture:** Treat the no-refiner PINN and supervised FFNO reruns as row-level replacements by lineage. Rebuild tables and indexes without changing non-FFNO metrics, and keep the historical `fno_cnn_blocks=2` evidence visible only as proxy context.

**Tech Stack:** Existing paper result refresh scripts, JSON/CSV/TeX assets, Markdown evidence indexes.

---

## Tasks

- [ ] Audit corrected no-refiner FFNO row roots and verify `fno_cnn_blocks=0` in config/invocation.
- [ ] Regenerate CDI metrics and objective-control tables with corrected FFNO rows.
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
