# Ground Truth Manifest

**Scenario ID:** PGRID-20250826-P1E5-T1024

**Generated:** 2026-01-23T00:15:06.250570+00:00

**Dataset Root:** `photon_grid_study_20250826_152459`

**Spec Reference:** `specs/data_contracts.md`

## Datasets

| File | Size | SHA256 |
|------|------|--------|
| `data_p1e3.npz` | 4.85 MB | `f9fa3f9f3f1cf8fc...` |
| `data_p1e4.npz` | 11.07 MB | `1cce1fe9596a8229...` |
| `data_p1e5.npz` | 16.58 MB | `01007daf8afc67aa...` |
| `data_p1e6.npz` | 24.64 MB | `95cfd6aee6b2c061...` |
| `data_p1e7.npz` | 35.55 MB | `9902ae24e90d2fa6...` |
| `data_p1e8.npz` | 44.45 MB | `56b4f66a92aa28b2...` |
| `data_p1e9.npz` | 55.29 MB | `3e1f229af34525a7...` |

### NPZ Key Validation

Required keys (per `specs/data_contracts.md`): `['diff3d', 'probeGuess', 'scan_index']`

**Sample (data_p1e3.npz):**
- Present optional keys: `['ground_truth_patches', 'objectGuess', 'xcoords', 'xcoords_start', 'ycoords', 'ycoords_start']`
- Extra keys: `[]`

## Baseline Outputs

**params.dill:** `params.dill`
- SHA256: `92c27229e2edca3a...`
- N: 64
- gridsize: 1
- nepochs: 50

| File | Type | Size | SHA256 |
|------|------|------|--------|
| `baseline_model.h5` | baseline_output | 54203.1 KB | `46b88686b95ce4e4...` |
| `recon.dill` | baseline_output | 820.0 KB | `2501b93db2fea8e3...` |

## PINN Weights

| `wts.h5.zip` | 31.98 MB | `56a26314a6c6db4f...` |
