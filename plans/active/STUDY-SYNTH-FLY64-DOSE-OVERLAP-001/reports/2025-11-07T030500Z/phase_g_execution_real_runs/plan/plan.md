# Phase G Comparison Execution — Loop Plan (2025-11-07T03:05:00Z)

## Objective
Restore three-way comparison execution (dose=1000 dense/train,test) by wiring Phase F manifests into the comparison CLI and regenerating prerequisite datasets/checkpoints under this hub.

## Checklist
- [ ] Update `studies/fly64_dose_overlap/comparison.py::execute_comparison_jobs` to read Phase F manifest JSON and pass `--tike_recon_path` pointing at `ptychi_reconstruction.npz`.
- [ ] Extend `tests/study/test_dose_overlap_comparison.py` fixtures/tests to cover manifest-driven `--tike_recon_path` wiring (RED→GREEN).
- [ ] Run targeted selector: `pytest tests/study/test_dose_overlap_comparison.py -k "manifest" -vv` (archive RED/GREEN logs).
- [ ] Regenerate dose=1000 dense Phase C data (`phase_c/`), Phase D overlap (`phase_d/`), Phase E training artifacts (`data/training/`), and Phase F recon (`data/reconstruction/`) under this hub.
- [ ] Execute Phase G comparisons for dose=1000 dense/train and dense/test, logging outputs to `cli/dose1000_dense_{train,test}/comparison.log`.
- [ ] Update `comparison_manifest.json` and summary files with execution results; capture metrics/plots under `analysis/`.
- [ ] Log outcomes in `docs/fix_plan.md` and `galph_memory.md`.

## Key Commands (AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md)
1. `pytest tests/study/test_dose_overlap_comparison.py -k "manifest" -vv`
2. `python -m studies.fly64_dose_overlap.generation ...` (Phase C, dose=1000 only)
3. `python -m studies.fly64_dose_overlap.overlap ... --doses 1000 --views dense`
4. `python -m studies.fly64_dose_overlap.training ... --dose 1000 --view baseline --gridsize 1`
5. `python -m studies.fly64_dose_overlap.training ... --dose 1000 --view dense --gridsize 2`
6. `python -m studies.fly64_dose_overlap.reconstruction ... --dose 1000 --view dense --split train`
7. `python -m studies.fly64_dose_overlap.reconstruction ... --dose 1000 --view dense --split test`
8. `python -m studies.fly64_dose_overlap.comparison ... --dose 1000 --view dense --split train`
9. `python -m studies.fly64_dose_overlap.comparison ... --dose 1000 --view dense --split test`

## Dependencies & Guardrails
- Ensure PyTorch backend remains available (POLICY-001) for Phase F LSQML.
- Apply CONFIG-001 bridge via existing Phase E CLI; confirm logs show `update_legacy_dict`.
- Respect DATA-001 when regenerating NPZs; rely on existing CLI validation.
- Enforce TYPE-PATH-001 by keeping all new path operations on `Path` objects.
- Abort comparisons if manifest lacks `ptychi_reconstruction.npz`; document in `analysis/blocker.log`.
