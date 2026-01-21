### Turn Summary
Scoped Phase D4d to keep dataset-derived intensity scaling CPU-only so lazy containers stop materializing `.X` on GPU.
Updated `plans/active/DEBUG-SIM-LINES-DOSE-001/{implementation.md,summary.md}` and docs/fix_plan.md with the new checklist plus reviewer context, then rewrote input.md with concrete code/test/analyzer commands and artifacts hub `2026-01-21T004455Z/`.
Next: Ralph updates `ptycho/train_pinn.py::calculate_intensity_scale()`, adds the `_tensor_cache` regression test, reruns gs2_ideal/analyzer, and archives pytest logs under the new hub.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-21T004455Z/
