# Initiative: GRID-LINES-WORKFLOW-001

## Problem Statement
The deprecated `notebooks/ptycho_lines.ipynb` and the legacy grid-based simulation pipeline are no longer modular or reproducible. We need a self-contained, callable workflow under `ptycho/workflows/` that can:
1) generate grid-sampled simulated datasets using the lines object, 2) use the probe extracted from `datasets/Run1084_recon3_postPC_shrunk_3.npz` (upscaled to 128x128 and also kept at 64x64), 3) train both PtychoPINN and the baseline model, and 4) run inference, stitch outputs, and compute SSIM metrics against ground truth.

## Success Criteria
- [ ] A new workflow module exists under `ptycho/workflows/` with a clear API to run grid-based simulation + training + inference + stitching + SSIM.
- [ ] The workflow supports **separate runs** for N=64 and N=128, using the lines object and the extracted probe (64 native + 128 upscaled).
- [ ] Outputs include stitched reconstructions, ground truth, and SSIM metrics in a reproducible output directory layout.
- [ ] Documentation and plan records capture the legacy notebook mapping and parameter choices.

## Deliverables
1. Workflow module (e.g., `ptycho/workflows/grid_lines_workflow.py`) implementing the end-to-end pipeline.
2. Minimal driver or example usage (module-level `main()` or short usage snippet).
3. SSIM report + stitched reconstruction outputs saved to the run directory.

## Constraints
- Must respect CONFIG-001 (call `update_legacy_dict` before legacy modules) and avoid import-time side effects (ANTIPATTERN-001).
- Treat core physics/model code as stable (no edits to `ptycho/model.py`, `ptycho/diffsim.py`, `ptycho/tf_helper.py`).
- Grid-based simulation must use legacy `diffsim.mk_simdata()` consistent with notebook workflow.
- Use `datasets/Run1084_recon3_postPC_shrunk_3.npz` as probe source (`probeGuess`), producing 64 and 128 versions.
- Runs are heavy/quality-focused and executed **separately** for N=64 and N=128.

## Notes
- This plan begins in discovery/brainstorming: the first step is mapping the deprecated notebook to a clean modular workflow and resolving probe scaling strategy.
