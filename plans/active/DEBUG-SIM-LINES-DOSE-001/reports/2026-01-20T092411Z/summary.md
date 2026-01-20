### Turn Summary — 2026-01-20T014500Z
Extended `run_dose_stage.py` with parameter clamping to fix the KD-tree IndexError: neighbor_count is now clamped to `min(current, nimages - 1)` so the tree never requests more neighbors than exist, nimages is capped at 512 to avoid GPU OOM, and gridsize+N are forced to 1 and 64 respectively for simulation stage since `RawData.from_simulation` requires gridsize=1 and the NPZ probe is 64x64.
Simulation stage now completes successfully, producing `simulated_data_visualization.png` and `random_groups_*.png` artifacts with 512 diffraction patterns.
Training stage fails with `KerasTensor cannot be used as input to a TensorFlow function` — this is a known Keras 3.x compatibility issue in the legacy model code, not related to the IndexError fix.
Next: decide whether to provision a legacy TF/Keras environment for full training or document the simulation-only capability.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/ (simulation_clamped4.log, .artifacts/simulation/)

---

### Turn Summary — Prior
Patched a non-production compatibility runner plus shimmed `tensorflow_addons`/`components` so the legacy `/home/ollie/Documents/PtychoPINN` `dose_experiments` scripts could import under the frozen env, then attempted the A1b simulate→train→infer flow.
Missing-module errors are gone, but simulation still fails — full-size runs OOM the RTX 3090 even after reducing `--nimages`, and smoke runs with tiny `nimages` crash in `RawData.group_coords` because the legacy `neighbor_count=5` exceeds the available scan positions; logs recorded under this hub (e.g., simulation_attempt16.log, simulation_smoke.log).
Next: finish the compatibility runner so it clamps neighbor_count/group_count, keeps `nimages` ≤512, and rerun the pipeline to produce archived ground-truth artifacts.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/ (import_path.log, simulation_attempt*.log, simulation_smoke.log)
