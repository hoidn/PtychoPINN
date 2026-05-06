# CDI Lines128 FFNO No-Refiner Row Rerun Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create worktrees.

**Goal:** Produce a corrected Lines128 CDI `pinn_ffno` row with `fno_cnn_blocks=0`.

**Architecture:** Launch a single append-only FFNO row under the completed Lines128 CDI contract, overriding only the local-refiner count. Keep the previous `fno_cnn_blocks=2` root as historical proxy evidence and write a summary that makes the architecture correction explicit.

**Tech Stack:** `scripts/studies/grid_lines_torch_runner.py`, `scripts/studies/grid_lines_compare_wrapper.py`, PyTorch FFNO generator, Markdown/JSON/CSV summaries.

---

## Tasks

- [ ] Confirm the completed Lines128 contract and source root from `docs/plans/NEURIPS-HYBRID-RESNET-2026/lines128_paper_benchmark_summary.md`.
- [ ] Add or use a row-local runner path that launches `pinn_ffno` with `fno_cnn_blocks=0` without mutating completed roots.
- [ ] Before launch, instantiate the configured model and assert `len(model.refiners) == 0`.
- [ ] Run only the corrected FFNO row into the item-local artifact root.
- [ ] Write a row audit proving all fixed contract fields match the completed table except `fno_cnn_blocks`.
- [ ] Publish `cdi_lines128_ffno_no_refiner_row_rerun_summary.md` with corrected metrics, historical-proxy comparison, and artifact lineage.

## Verification

- `python - <<'PY'` instantiate `FfnoGeneratorModule(cnn_blocks=0)` and assert zero refiners.
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "ffno"`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "ffno"`
- `python -m compileall -q ptycho_torch scripts/studies`
- Artifact audit fails if `fno_cnn_blocks` is missing, nonzero, or if any non-allowed Lines128 contract field changed.
