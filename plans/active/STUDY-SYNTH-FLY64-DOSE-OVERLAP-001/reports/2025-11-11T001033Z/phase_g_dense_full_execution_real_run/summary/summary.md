### Turn Summary
Mapped the real state of the highlight metadata guard, updated fix_plan/input, and staged the 2025-11-11 hub so Ralph can focus on metadata-aware TDD plus the dense rerun.
Explained that the verifier already inspects metrics_delta_summary.json + preview files, so the new Do Now targets structured metadata assertions and consistent result shapes before a full pipeline run.
Next: Ralph executes the updated tests and verifier patch, then runs `run_phase_g_dense.py --clobber` and archives verifier/highlight outputs with MS-SSIM/MAE deltas.
Artifacts: plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-11T001033Z/phase_g_dense_full_execution_real_run/ (plan/plan.md, summary/summary.md)
