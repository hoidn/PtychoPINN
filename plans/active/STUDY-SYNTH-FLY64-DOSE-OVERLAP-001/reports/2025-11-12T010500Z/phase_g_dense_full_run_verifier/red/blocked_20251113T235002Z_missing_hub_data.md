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
```bash
$ ls -d "$HUB"/data/phase_e/dose_1000/dense/gs2
ls: cannot access... No such file or directory
```

Data exists elsewhere:
- `data/phase_c/dose_1000/patched_{train,test}.npz` exists in project root
- `archive/phase_c_*/*.npz` contains archived versions

## Status
Translation guard tests are GREEN (2/2 passed, 6.18s). Chunked Baseline code is merged and present in scripts/compare_models.py:1150-1250.

## Next Action Required
Before running compare_models commands, either:
1. Populate `$HUB/data/` with symlinks/copies of the required models and data, OR
2. Update the Brief to use actual paths from project root (`data/phase_c/dose_1000/`, `analysis/dose_1000/*/...`)

## Mitigation  
Cannot proceed with compare_models execution until data paths are resolved.
