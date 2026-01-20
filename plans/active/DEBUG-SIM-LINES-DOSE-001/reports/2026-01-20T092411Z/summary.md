### Turn Summary
Patched a non-production compatibility runner plus shimmed `tensorflow_addons`/`components` so the legacy `/home/ollie/Documents/PtychoPINN` `dose_experiments` scripts could import under the frozen env, then attempted the A1b simulate→train→infer flow.
Missing-module errors are gone, but simulation still fails — full-size runs OOM the RTX 3090 even after reducing `--nimages`, and smoke runs with tiny `nimages` crash in `RawData.group_coords` because the legacy `neighbor_count=5` exceeds the available scan positions; logs recorded under this hub (e.g., simulation_attempt16.log, simulation_smoke.log).
Next: finish the compatibility runner so it clamps neighbor_count/group_count, keeps `nimages` ≤512, and rerun the pipeline to produce archived ground-truth artifacts.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-20T092411Z/ (import_path.log, simulation_attempt*.log, simulation_smoke.log)
