# CDI Lines128 Supervised FFNO Depth-24 No-Refiner Rerun Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create worktrees.

**Goal:** Add a supervised Lines128 CDI FFNO row with `fno_blocks=24` and `fno_cnn_blocks=0`.

**Architecture:** Reuse the supervised four-block no-refiner row as the comparator by lineage, but create a distinct `supervised_ffno_depth24` row for the depth-24 architecture. Keep all dataset, probe, training, scheduler, metric, and visual contracts fixed except for `fno_blocks`.

**Tech Stack:** Grid-lines compare wrapper, PyTorch FFNO generator, Lightning metrics/history, Markdown/JSON evidence indexes.

---

## Tasks

- [ ] Locate the completed four-block no-refiner supervised source root from `2026-05-06-cdi-lines128-supervised-ffno-no-refiner-rerun`.
- [ ] Add or use a wrapper path that can emit a distinct `supervised_ffno_depth24` row with `fno_blocks=24`.
- [ ] Add focused tests or wrapper assertions proving `supervised_ffno` remains `fno_blocks=4` and `supervised_ffno_depth24` records `fno_blocks=24`.
- [ ] Run only the new supervised depth-24 row into an item-local root.
- [ ] Audit invocation/config artifacts for `fno_blocks=24`, `fno_cnn_blocks=0`, and unchanged fixed contract fields.
- [ ] Write `docs/plans/NEURIPS-HYBRID-RESNET-2026/cdi_lines128_supervised_ffno_depth24_no_refiner_summary.md`.
- [ ] Update `evidence_matrix.md`, `model_variant_index.json`, `ablation_index.json`, and `docs/studies/index.md`.

## Verification

- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
- `python -m compileall -q ptycho_torch scripts/studies`
- Row-local artifact audit for `fno_blocks=24`, `fno_cnn_blocks=0`, and unchanged fixed contract fields.
