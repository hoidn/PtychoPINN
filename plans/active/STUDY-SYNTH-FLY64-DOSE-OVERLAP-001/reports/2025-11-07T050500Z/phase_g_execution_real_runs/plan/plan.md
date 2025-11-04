# Phase G Comparison Execution — Do Now Scaffold (2025-11-07T05:05:00Z)

## Objective
Restore Phase G three-way comparisons (dose=1000 dense/train,test) by wiring Phase F manifests into the comparison CLI and regenerating prerequisite artifacts inside this hub.

## Checklist
- [ ] Update `studies/fly64_dose_overlap/comparison.py::execute_comparison_jobs` to parse each job's manifest JSON, validate `ptychi_reconstruction.npz`, and append `--tike_recon_path=<path>` while keeping all paths as `Path` objects.
- [ ] Add helper coverage in `tests/study/test_dose_overlap_comparison.py` (new test `test_execute_comparison_jobs_appends_tike_recon_path`) exercising both missing-file failure and happy path.
- [ ] Run targeted selector with AUTHORITATIVE_CMDS_DOC guard: `pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv` (capture RED/GREEN logs).
- [ ] Regenerate Phase C data: `python -m studies.fly64_dose_overlap.generation --base-npz <fly64 base> --output-root .../data/phase_c`.
- [ ] Phase D overlap for dose=1000 view=dense: `python -m studies.fly64_dose_overlap.overlap --doses 1000 --views dense --artifact-root .../analysis`.
- [ ] Phase E training runs (baseline gs1 TensorFlow, dense gs2 TensorFlow) writing to `.../data/training`.
- [ ] Phase F LSQML reconstructions (dense/train and dense/test) into `.../data/reconstruction`.
- [ ] Execute Phase G comparisons (dense/train and dense/test) capturing logs in `cli/dose1000_dense_{split}/comparison.log` and ensuring manifests record `tike_recon_path`.
- [ ] Summarize outcomes + checksums in `analysis/summary.md`; update ledger + memory after Ralph completes.

## Command Map (set `AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md` before running)
1. `pytest tests/study/test_dose_overlap_comparison.py -k tike_recon_path -vv`
2. `python -m studies.fly64_dose_overlap.generation --base-npz tike_outputs/fly001_reconstructed_final_prepared/fly001_reconstructed_interp_smooth_both.npz --output-root {HUB}/data/phase_c`
3. `python -m studies.fly64_dose_overlap.overlap --phase-c-root {HUB}/data/phase_c --output-root {HUB}/data/phase_d --artifact-root {HUB}/analysis --doses 1000 --views dense`
4. `python -m studies.fly64_dose_overlap.training --phase-c-root {HUB}/data/phase_c --phase-d-root {HUB}/data/phase_d --artifact-root {HUB}/data/training --dose 1000 --view baseline --gridsize 1 --backend tensorflow`
5. `python -m studies.fly64_dose_overlap.training --phase-c-root {HUB}/data/phase_c --phase-d-root {HUB}/data/phase_d --artifact-root {HUB}/data/training --dose 1000 --view dense --gridsize 2 --backend tensorflow`
6. `python -m studies.fly64_dose_overlap.reconstruction --phase-c-root {HUB}/data/phase_c --phase-d-root {HUB}/data/phase_d --artifact-root {HUB}/data/reconstruction --dose 1000 --view dense --split train`
7. `python -m studies.fly64_dose_overlap.reconstruction --phase-c-root {HUB}/data/phase_c --phase-d-root {HUB}/data/phase_d --artifact-root {HUB}/data/reconstruction --dose 1000 --view dense --split test`
8. `python -m studies.fly64_dose_overlap.comparison --phase-c-root {HUB}/data/phase_c --phase-e-root {HUB}/data/training --phase-f-root {HUB}/data/reconstruction --artifact-root {HUB}/analysis --dose 1000 --view dense --split train`
9. `python -m studies.fly64_dose_overlap.comparison --phase-c-root {HUB}/data/phase_c --phase-e-root {HUB}/data/training --phase-f-root {HUB}/data/reconstruction --artifact-root {HUB}/analysis --dose 1000 --view dense --split test`

_HUB = plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-07T050500Z/phase_g_execution_real_runs_

## Guardrails & Notes
- Maintain TYPE-PATH-001 by normalizing all filesystem inputs with `Path()` inside the implementation and tests.
- Ensure CONFIG-001 bridge is automatically satisfied by the training CLI; capture logs to prove it executes.
- If reconstruction outputs are missing, capture the failure log and stop before comparisons; mark attempt blocked.
- Deterministic policy: stick to CPU execution flags/seed defaults already codified in the CLIs (see `docs/TESTING_GUIDE.md`).
- Update `docs/fix_plan.md` and `galph_memory.md` after execution with artifact paths and findings references.
