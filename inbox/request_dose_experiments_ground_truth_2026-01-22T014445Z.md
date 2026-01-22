Title: Request — legacy dose_experiments ground-truth artifacts

Context
- Initiative: DEBUG-SIM-LINES-DOSE-001 (Phase D amplitude-bias investigation)
- Need: Ground-truth outputs from the legacy `dose_experiments` pipeline for direct comparison.
- Blocker: Local runs fail under Keras 3.x (`KerasTensor cannot be used as input to a TensorFlow function`).

Request
Please run the legacy `dose_experiments` simulate→train→infer flow in a compatible legacy TF/Keras environment and provide the artifacts below.

Minimum artifacts requested
1) Simulation outputs (NPZ) and any generated probes/objects.
2) Training logs (loss curves, config snapshot, seed info).
3) Inference outputs:
   - Reconstructed amplitude/phase arrays (NPY/NPZ acceptable).
   - Any stitched image PNGs or intermediate outputs.
4) Config snapshot(s) used (loss mode, gridsize, N, nphotons, probe settings).

Preferred delivery
- Drop under: `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth/`
- Include a brief `README.md` with exact commands, environment details, and any deviations.

Why this matters
We need a faithful legacy baseline to compare sim_lines_4x amplitude bias against dose_experiments outputs. This is required to close A1b and proceed with Phase D.
