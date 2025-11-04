# Phase G Dense Execution — Do Now Scaffold (2025-11-07T07:05:00Z)

## Objective
Generate real evidence for dose=1000 dense/train and dense/test comparisons by orchestrating Phase C→G pipelines inside this hub using the manifest-aware executor.

## Checklist
- [ ] Create initiative script `plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py` that sequentially executes Phase C→G commands for a provided hub directory, propagating `AUTHORITATIVE_CMDS_DOC` and capturing stdout/stderr to per-phase log files.
- [ ] Re-run `pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv` (RED already green, guard for regressions) and archive log outputs under this hub.
- [ ] Execute Phase C dataset regeneration into `{HUB}/data/phase_c` using fly64 base NPZ, logging to `cli/phase_c_generation.log`.
- [ ] Generate dense overlap Phase D artifacts into `{HUB}/data/phase_d` with analysis artifacts under `{HUB}/analysis`.
- [ ] Train baseline (gs1) and dense (gs2) TensorFlow models for dose=1000, writing manifests under `{HUB}/data/training` and preserving CLI logs.
- [ ] Produce Phase F LSQML reconstructions for dense/train and dense/test splits; confirm manifests include `output_dir` and `ptychi_reconstruction.npz` under `{HUB}/data/reconstruction`.
- [ ] Run Phase G comparisons (`python -m studies.fly64_dose_overlap.comparison`) for dense/train and dense/test, ensuring logs land in `cli/` and metrics/plots in `analysis/`.
- [ ] Summarize key metrics (MS-SSIM phase/amplitude, MAE) and artifact locations in `{HUB}/summary/summary.md`; update ledger + galph memory with outcome and references.

## Command Map (set `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before running)
1. `pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv`
2. `python plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/bin/run_phase_g_dense.py --hub {HUB} --dose 1000 --view dense --splits train test`
3. Archive resulting logs/metrics already written by the script (no extra commands expected beyond verifying outputs).

_HUB = plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T070500Z/phase_g_execution_real_runs_

## Guardrails & Notes
- Script must bail on non-zero return codes and record the failing command in `{HUB}/analysis/blocker.log` before raising.
- Preserve TYPE-PATH-001 by normalizing all filesystem inputs/outputs inside the script.
- Training CLI already bridges CONFIG-001; capture CLI stdout to prove the manifest annotations.
- Deterministic execution: keep CPU, default seeds; do not enable GPUs or change hyperparameters.
- If reconstruction or comparison fails, stop and mark attempt blocked; do not fabricate metrics.
- After completion, ensure `docs/fix_plan.md` Attempts History includes this attempt with artifact path and findings referenced.
