### Turn Summary (2026-01-26)
Created a new initiative for the grid-based lines workflow and inspected the deprecated notebook + probe source.
Findings:
- `notebooks/ptycho_lines.ipynb` uses legacy params.cfg with `data_source='lines'`, `size=392`, `gridsize=1`, `offset=4`, `outer_offset_train=8`, `outer_offset_test=20`, `nimgs_train=2`, `nimgs_test=2`, `nphotons=1e4`, and relies on `ptycho.generate_data` + `train_pinn`.
- Probe source `datasets/Run1084_recon3_postPC_shrunk_3.npz` contains `probeGuess` (64x64 complex128), `objectGuess` (227x226 complex128), and `diffraction` in legacy (H,W,N) order (64x64x1087).
Next: continue requirement capture (probe upscaling strategy + output layout + metrics scope) and draft modular workflow design.
Artifacts: notebooks/ptycho_lines.ipynb; datasets/Run1084_recon3_postPC_shrunk_3.npz
### Turn Summary (2026-01-26)
Captured workflow decisions: persist simulated NPZs under `output_dir/datasets/N{N}/gs{gridsize}/{train,test}.npz`; probe upscaling via `prepare_data_tool.interpolate_array` with probe smoothing sigma=0.5; support gridsize=1 and gridsize=2 in separate invocations; separate runs for N=64 and N=128; nphotons=1e9; nimgs_train/nimgs_test=2; epochs=60; SSIM via `ptycho.evaluation.eval_reconstruction` saved as JSON; stitched outputs saved for amplitude + phase; PINN default loss=MAE with configurable weights; save model checkpoints; keep notebook object size and outer offsets for both N=64 and N=128 (size=392, outer_offset_train=8, outer_offset_test=20); baseline uses **channel 0 only** for gridsize>1 (no flatten); save comparison PNGs (GT vs PINN vs Baseline); emit run parameters as JSON metadata.
Next: finalize remaining output layout (run artifact dirs, naming) and draft modular workflow design.
Artifacts: plans/active/GRID-LINES-WORKFLOW-001/summary.md
### Turn Summary (2026-01-27)
Drafted the design/implementation plan for the grid-lines workflow, including module/CLI layout, probe prep, grid simulation, baseline channel handling, stitching workaround, and output artifacts.
Artifacts: docs/plans/2026-01-27-grid-lines-workflow.md
### Turn Summary (2026-01-27)
Created Phase 0 test strategy for GRID-LINES-WORKFLOW-001 (pytest-only unit tests, artifacts under `.artifacts/`).
Artifacts: plans/active/GRID-LINES-WORKFLOW-001/test_strategy.md
### Turn Summary (2026-01-27)
Implemented Task 1 skeleton: new workflow module and CLI wrapper for grid lines; CLI help validated.
Artifacts: ptycho/workflows/grid_lines_workflow.py; scripts/studies/grid_lines_workflow.py
