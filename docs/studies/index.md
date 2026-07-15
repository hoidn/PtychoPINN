# Studies Index

## CDI Datasets / Resources

### `grid-lines-ffno-ci-integration-calibration` (5 epochs; half training population)

- Purpose: provide a main-native FFNO example and regression gate for the
  corrected CI probe/scale chain at N=128, grid size 1, and seed 3. It uses one
  synthetic training object group (4,489 diffraction patterns) and one test
  group (729 patterns).
- Architecture boundary: FFNO is the executable `main` substitute for the
  `fno-stable` Hybrid ResNet integration example. `main` deliberately excludes
  the ResNet family, so the Hybrid command and metric fixture will not run on
  this branch as written. The FFNO calibration must not be cited as Hybrid
  ResNet evidence or as a manuscript benchmark.
- Selection: under this exact five-epoch contract, FFNO exceeded FNO SSIM for
  both amplitude (`0.52034` versus `0.51027`) and phase (`0.88549` versus
  `0.86314`), so the main-native gate uses FFNO. This is an SSIM-based choice;
  FNO retained lower MAE and loss, so it does not establish universal FFNO
  superiority.
- Prerequisites: a CUDA device, the `ptycho311` environment, and the Run1084
  probe fixture at `datasets/Run1084_recon3_postPC_shrunk_3.npz`.
- Run the causal CPU gate and the calibrated five-epoch FFNO outcome gate:

  ```bash
  source /home/ollie/miniconda3/etc/profile.d/conda.sh
  conda activate ptycho311
  python -m pytest -q \
    tests/torch/test_grid_lines_ci_probe_roundtrip_integration.py::test_grid_lines_ci_roundtrip_uses_realized_probe_and_preserves_count_rate \
    tests/torch/test_grid_lines_ci_ffno_quality_integration.py::test_grid_lines_ffno_ci_nll_five_epoch_quality_and_visual
  ```

- The slow gate recreates
  `.artifacts/integration/grid_lines_ffno_ci_nll/`; do not run another writer
  against that root concurrently. Its calibration is stored in
  `tests/fixtures/grid_lines_ffno_ci_nll_5ep_metrics.json`, and its full
  amplitude/phase truth-reconstruction-error grid is written to
  `.artifacts/integration/grid_lines_ffno_ci_nll/ci_nll_5ep/visuals/ci_nll_full_comparison.png`.
- Boundary: this is deterministic integration-test calibration and regression
  evidence only.
