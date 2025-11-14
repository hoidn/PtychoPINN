# Blocker: Missing HUB Data Directories

## Timestamp
2025-11-13T21:00:00Z

## Issue
The input.md Brief requests execution of compare_models commands with paths under `$HUB/data/`, but these directories do not exist:

- `$HUB/data/phase_e/dose_1000/dense/gs2` (PINN model dir)
- `$HUB/data/phase_e/dose_1000/baseline/gs1` (Baseline model dir)
- `$HUB/data/phase_c/dose_1000/patched_train.npz` (train data)
- `$HUB/data/phase_c/dose_1000/patched_test.npz` (test data)
- `$HUB/data/phase_f/dose_1000/dense/train/ptychi_reconstruction.npz` (PtyChi recon)

## Evidence
Translation guard tests GREEN (2/2 passed, 6.18s, see green/pytest_compare_models_translation_fix_v18.log).
Chunked Baseline code present (scripts/compare_models.py:1150-1250).

Available data found at:
- data/phase_c/dose_1000/patched_{train,test}.npz (project root)
- archive/phase_c_20251113T102533Z/dose_1000/ (archived)

## Status
Cannot execute compare_models commands from input.md Brief without resolving data paths.

## Next Action
Supervisor must either populate $HUB/data/ or update Brief with correct paths.
