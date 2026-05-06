# CDI Lines128 FFNO Depth-24 Final Paper Refresh Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create worktrees.

**Goal:** Refresh final CDI Lines128 paper assets after the depth-24 no-refiner FFNO PINN and supervised rows complete.

**Architecture:** Treat the four-block no-refiner refresh as the interim paper-asset baseline, then audit depth-24 PINN/supervised rows for final promotion. Regenerate only CDI FFNO-consuming paper assets and indexes; reuse all non-FFNO rows by lineage.

**Tech Stack:** Existing paper result refresh scripts, JSON/CSV/TeX assets, figure/image manifests, Markdown evidence indexes.

---

## Tasks

- [ ] Audit depth-24 PINN and supervised row roots for `fno_blocks=24` and `fno_cnn_blocks=0`.
- [ ] Compare depth-24 metrics against four-block no-refiner rows and record the promotion decision.
- [ ] Regenerate final CDI metrics, objective-control, model-config, efficiency, and figure/image assets with the chosen final FFNO rows.
- [ ] Preserve interim four-block no-refiner and historical local-refiner proxy rows in evidence indexes with clear labels.
- [ ] Update `evidence_matrix.md`, `paper_evidence_index.md`, `model_variant_index.json`, `ablation_index.json`, and `docs/studies/index.md`.
- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_ffno_depth24_final_paper_refresh_summary.md`.

## Verification

- `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/model_variant_index.json`
- `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/ablation_index.json`
- `python -m json.tool docs/plans/NEURIPS-HYBRID-RESNET-2026/tables/cdi_lines128_metrics_extended.json`
- CSV parse smoke check for regenerated CSV files.
- Figure/image manifest audit proving canonical FFNO panels point to the selected final no-refiner roots.
