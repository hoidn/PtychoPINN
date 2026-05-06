# CDI Lines128 Supervised FFNO No-Refiner Rerun Execution Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Do not create worktrees.

**Goal:** Produce a corrected supervised Lines128 CDI FFNO row with `fno_cnn_blocks=0`.

**Architecture:** Reuse the supervised-extension harness and locked Lines128 contract, but launch only `supervised_ffno` with no post-FFNO local refiners. Compare it to the corrected no-refiner `pinn_ffno` row by lineage.

**Tech Stack:** Existing grid-lines Torch runner/wrapper, PyTorch FFNO generator, JSON/CSV/TeX summary assets.

---

## Tasks

- [ ] Confirm the corrected no-refiner `pinn_ffno` source root from `2026-05-06-cdi-lines128-ffno-no-refiner-row-rerun`.
- [ ] Launch only `supervised_ffno` with `fno_cnn_blocks=0` into an item-local root.
- [ ] Write a contract audit comparing the new supervised row to the completed supervised extension and proving only `fno_cnn_blocks` changed.
- [ ] Write an objective-control table using corrected no-refiner `pinn_ffno` and corrected no-refiner `supervised_ffno`.
- [ ] Publish `cdi_lines128_supervised_ffno_no_refiner_rerun_summary.md`.

## Verification

- `python - <<'PY'` instantiate `FfnoGeneratorModule(cnn_blocks=0)` and assert zero refiners.
- `pytest -q tests/torch/test_grid_lines_torch_runner.py -k "supervised_ffno or ffno"`
- `pytest -q tests/test_grid_lines_compare_wrapper.py -k "supervised_ffno or ffno"`
- `python -m compileall -q ptycho_torch scripts/studies`
- Artifact audit fails if either corrected FFNO objective-control row has `fno_cnn_blocks != 0`.
