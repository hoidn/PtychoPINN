# CDI FFNO No-Refiner Contract Reactivation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. Do not create worktrees.

**Goal:** Reactivate CDI FFNO paper rows under a no-refiner `fno_cnn_blocks=0` contract and keep older CDI FFNO rows discoverable only as local-refiner proxy evidence.

**Architecture:** Preserve completed backlog items and artifact roots as provenance. Add new active backlog items for corrected Lines128 PINN and supervised FFNO reruns, then a table-refresh item that promotes only rows with the corrected no-refiner contract. Update generated paper-facing tables, indexes, and plans so future readers cannot mistake the historical `fno_cnn_blocks=2` rows for canonical FFNO evidence.

**Tech Stack:** Markdown backlog/plans, JSON/CSV/TeX paper artifacts, existing PyTorch CDI runner config, `FfnoGeneratorModule(cnn_blocks=0)`.

---

## Source Of Truth

- CDI FFNO implementation: `ptycho_torch/generators/ffno.py`
- CDI runner config knob: `scripts/studies/grid_lines_torch_runner.py` / `fno_cnn_blocks`
- Current historical CDI FFNO roots:
  - `2026-04-27-cdi-ffno-generator-lines-best-config`
  - `2026-04-29-cdi-lines128-paper-benchmark-execution`
  - `2026-04-29-cdi-lines128-supervised-equivalent-rows`
  - `2026-04-30-cdi-lines128-uno-table-extension`

## Tasks

- [ ] Add active no-refiner CDI FFNO rerun items:
  - `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`
  - `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`
  - `2026-05-06-cdi-lines128-no-refiner-ffno-table-refresh`
- [ ] Update completed CDI FFNO backlog items with post-hoc caveats instead of changing their completion state.
- [ ] Rework dependent active/paused plans so CDI FFNO rows that make pure-FFNO claims use `fno_cnn_blocks=0`.
- [ ] Relabel current generated paper-facing CDI FFNO rows as `FFNO-local proxy` and add replacement-item pointers.
- [ ] Validate Markdown frontmatter paths, JSON/CSV parseability, and local instantiation of `FfnoGeneratorModule(cnn_blocks=0)`.

## Verification

- `python -m json.tool` on edited JSON artifacts.
- `python - <<'PY'` CSV parse smoke check for edited CSV artifacts.
- `python - <<'PY'` frontmatter path smoke check for new active backlog items.
- `python - <<'PY'` instantiate `FfnoGeneratorModule(cnn_blocks=0)` and assert `len(model.refiners) == 0`.
