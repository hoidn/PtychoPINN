# Dose Experiments Ground-Truth Bundle

**Scenario ID:** PGRID-20250826-P1E5-T1024
**Generated:** 2026-01-23T00:35:37.797655+00:00
**Dataset Root:** `photon_grid_study_20250826_152459`

## 1. Overview

This bundle contains the artifacts from a legacy `dose_experiments` 
simulate->train->infer run. It is provided so Maintainer <2> can 
compare outputs against new implementations without re-running the 
legacy TF/Keras 2.x pipeline.

**Pipeline stages:**
1. **Simulation:** Generate diffraction patterns at various photon doses
2. **Training:** Train baseline model on simulated data
3. **Inference:** Reconstruct amplitude/phase from trained model

**Key Parameters:**
- N (patch size): 64
- gridsize: 1
- nepochs: 50
- batch_size: 16
- loss: NLL-only (nll_weight=1.0, mae_weight=0.0)
- probe.trainable: False
- intensity_scale.trainable: True
- intensity_scale_value (learned): 988.21

**Baseline Metrics (train / test):**
- MS-SSIM: 0.9248 / 0.9206
- PSNR: 71.32 dB / 158.06 dB

## 2. Environment Requirements

**IMPORTANT:** This pipeline was generated under TensorFlow/Keras 2.x. 
Running under Keras 3.x will produce errors such as:

```
KerasTensor cannot be used as input to a TensorFlow function
```

**Recommended environment:**
- Python 3.9 or 3.10
- TensorFlow 2.12 - 2.15
- Keras 2.x (bundled with TF 2.x)
- NumPy < 2.0

If you cannot set up this environment, use the pre-computed artifacts 
in this bundle instead of re-running the pipeline.

## 3. Simulation Commands

The canonical entry point is `notebooks/dose_dependence.ipynb`, which 
invokes `notebooks/dose.py`:

```python
# In dose_dependence.ipynb cell:
from notebooks import dose
dose.init(nphotons=1e5, loss_fn='nll')
# Simulation runs through dose.run_experiment_with_photons() internally
```

The `dose.init()` function configures:
```python
cfg['data_source'] = 'lines'
cfg['gridsize'] = 2  # Note: dose.py default; baseline used gs=1
cfg['intensity_scale.trainable'] = True
cfg['probe.trainable'] = False
cfg['nepochs'] = 60
```

**Note:** The baseline run in this bundle used `gridsize=1` and 
`nepochs=50`, overriding the dose.py defaults.

## 4. Training Commands

From repo root:

```bash
cd ~/Documents/PtychoPINN

python -m ptycho.train \
    --train_data_file photon_grid_study_20250826_152459/data_p1e5.npz \
    --output_dir photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run \
    --batch_size 16 \
    --nepochs 50 \
    --gridsize 1 \
    --intensity_scale_trainable True \
    --probe_trainable False
```

**Overrides from dose.py defaults:**
- `gridsize=1` (baseline; dose.py defaults to 2)
- `batch_size=16`
- `nepochs=50` (baseline; dose.py defaults to 60)

## 5. Inference Commands

```bash
python -m ptycho.inference \
    --model_path photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/pinn_run/wts.h5.zip \
    --test_data photon_grid_study_20250826_152459/data_p1e5.npz
```

## 6. Artifact Provenance Table

All sizes and SHA256 checksums are sourced from the Phase-A manifest 
(`ground_truth_manifest.json`). NPZ files conform to 
`specs/data_contracts.md` RawData NPZ requirements.

### Datasets

| File | Photon Dose | Size | SHA256 |
|------|-------------|------|--------|
| `data_p1e3.npz` | 1e3 | 4.85 MB | `f9fa3f9f3f1c...` |
| `data_p1e4.npz` | 1e4 | 11.07 MB | `1cce1fe9596a...` |
| `data_p1e5.npz` | 1e5 | 16.58 MB | `01007daf8afc...` |
| `data_p1e6.npz` | 1e6 | 24.64 MB | `95cfd6aee6b2...` |
| `data_p1e7.npz` | 1e7 | 35.55 MB | `9902ae24e90d...` |
| `data_p1e8.npz` | 1e8 | 44.45 MB | `56b4f66a92aa...` |
| `data_p1e9.npz` | 1e9 | 55.29 MB | `3e1f229af345...` |

### Baseline Artifacts

| File | Type | Size | SHA256 |
|------|------|------|--------|
| `params.dill` | params | 34.79 KB | `92c27229e2ed...` |
| `baseline_model.h5` | baseline_output | 52.93 MB | `46b88686b95c...` |
| `recon.dill` | baseline_output | 820.04 KB | `2501b93db2fe...` |
| `wts.h5.zip` | pinn_weights | 31.98 MB | `56a26314a6c6...` |

## 7. NPZ Key Requirements

Per `specs/data_contracts.md` RawData NPZ, each dataset NPZ must contain:

**Required keys:**
- `xcoords`, `ycoords` - scan positions (float64)
- `xcoords_start`, `ycoords_start` - starting positions (float64, deprecated but present)
- `diff3d` - diffraction patterns, shape (N_patterns, N, N), dtype float32
- `probeGuess` - embedded probe, shape (N, N), dtype complex128
- `scan_index` - scan indices (int64)

**Optional keys:**
- `objectGuess` - initial object guess
- `ground_truth_patches` - ground truth for training/evaluation

**Array shapes (from first dataset):**
- `diff3d`: [5000, 64, 64] (float32)
- `ground_truth_patches`: [5000, 64, 64, 1] (complex64)
- `objectGuess`: [232, 232] (complex128)
- `probeGuess`: [64, 64] (complex128)
- `scan_index`: [5000] (int64)
- `xcoords`: [5000] (float64)
- `xcoords_start`: [5000] (float64)
- `ycoords`: [5000] (float64)
- `ycoords_start`: [5000] (float64)

## 8. Delivery Artifacts

**Bundle root:** `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-22T014445Z/dose_experiments_ground_truth`
**Tarball:** `dose_experiments_ground_truth.tar.gz`
**Tarball size:** 270.70 MB
**Tarball SHA256:** `7fe5e14ed9909f056807b77d5de56e729b8b79c8e5b8098ba50507f13780dd72`

**Bundle structure:**
```
dose_experiments_ground_truth/
  simulation/     # 7 datasets (data_p1e3.npz ... data_p1e9.npz)
  training/       # params.dill, baseline_model.h5, recon.dill
  inference/      # wts.h5.zip (PINN weights)
  docs/           # README.md, manifests, summaries
```

**Verification summary:** 15/15 files verified, 278.18 MB total

---

**Manifest source:** `ground_truth_manifest.json`
**Baseline summary:** `dose_baseline_summary.json`

For questions, contact Maintainer <1> (dose_experiments branch).
