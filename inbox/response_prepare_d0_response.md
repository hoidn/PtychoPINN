# Response: D0 Parity Logging + Maintainer Coordination

**In reply to:** `README_prepare_d0_response.md`
**Date:** 2026-01-22
**Scenario ID:** PGRID-20250826-P1E5-T1024

---

## 1. Baseline Selection

**Selected baseline:**
`photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/`

**Rationale:**
- Complete artifact set: `params.dill` present with full metrics
- Stable metrics: ms_ssim = 0.925 (train) / 0.921 (test), psnr = 71.3 dB (amplitude)
- NLL-only loss (nll_weight=1.0, mae_weight=0.0), which is the canonical dose experiment loss
- gridsize=1 run with frozen probe, trainable intensity_scale
- Representative mid-dose scenario (1e5 photons) from the 7-level photon study

**Note:** While `params.dill` records `nphotons=1e9`, this is a legacy parameter. The actual photon dose is encoded in the dataset filename (e.g., `data_p1e5.npz` → 1e5 photons). The diffraction patterns themselves are scaled to the correct photon dose during simulation.

---

## 2. Dataset Parity Details

**Dataset root:** `photon_grid_study_20250826_152459/`

| File | Photon Dose | Patterns | Diff Shape | SHA256 |
|------|-------------|----------|------------|--------|
| data_p1e3.npz | 1e3 | 5000 | (5000, 64, 64) | `f9fa3f9f3f1cf8fc181f9c391b5acf15247ba3f25ed54fe6aaf9809388a860b2` |
| data_p1e4.npz | 1e4 | 5000 | (5000, 64, 64) | `1cce1fe9596a82290bffc3cd6116f43cfed17abe7339503dd849ec5826378402` |
| data_p1e5.npz | 1e5 | 5000 | (5000, 64, 64) | `01007daf8afc67aad3ad037e93077ba8bfb28b58d07e17a1e539bd202ffa0d95` |
| data_p1e6.npz | 1e6 | 5000 | (5000, 64, 64) | `95cfd6aee6b2c061e2a8fbe62a824274165e39e0c4514e4537ee1fecf7a79f64` |
| data_p1e7.npz | 1e7 | 5000 | (5000, 64, 64) | `9902ae24e90d2fa63bebf7830a538868cea524fab9cb15a512509803a9896251` |
| data_p1e8.npz | 1e8 | 5000 | (5000, 64, 64) | `56b4f66a92aa28b2983757417cad008ef14c45c796b2d61d9e366aae3a3d55cf` |
| data_p1e9.npz | 1e9 | 5000 | (5000, 64, 64) | `3e1f229af34525a7a912c9c62fa8df6ab87c69528572686a34de4d2640c57c4a` |

**Total size:** ~200 MB (all 7 NPZ files combined)

**NPZ array structure (all files share the same schema):**
- `diff3d`: (5000, 64, 64) float32 — diffraction patterns
- `probeGuess`: (64, 64) complex128 — embedded probe
- `objectGuess`: (232, 232) complex128 — initial object guess
- `xcoords`, `ycoords`: (5000,) float64 — scan positions
- `xcoords_start`, `ycoords_start`: (5000,) float64 — starting positions
- `scan_index`: (5000,) int64 — scan indices
- `ground_truth_patches`: (5000, 64, 64, 1) complex64 — ground truth for training

---

## 3. Probe Provenance

**Source:** Dataset-embedded `probeGuess` (64×64 complex128)

**Generation:**
- Produced by the lines-based simulator (see `notebooks/dose.py` lines 1–35)
- Configuration: `cfg['data_source'] = 'lines'`, `cfg['probe.mask'] = False` (though params.dill shows this as False for the baseline)
- `default_probe_scale = 0.7`

**Training settings:**
- `probe.trainable = False` (frozen during training)
- `intensity_scale.trainable = True` (learned scalar multiplier)
- Final learned `intensity_scale_value = 988.21`

---

## 4. Config Snapshot

Key parameters from `params.dill`:

| Parameter | Value |
|-----------|-------|
| N | 64 |
| gridsize | 1 |
| nimgs_train | 9 |
| nimgs_test | 3 |
| batch_size | 16 |
| nepochs | 50 |
| mae_weight | 0.0 |
| nll_weight | 1.0 |
| default_probe_scale | 0.7 |
| probe.trainable | False |
| intensity_scale.trainable | True |
| intensity_scale_value (learned) | 988.21 |
| label | baseline_gs1 |
| timestamp | 08/26/2025, 16:38:17 |

**Metrics (train, test):**

| Metric | Train | Test |
|--------|-------|------|
| mae | 0.00956 | 5.38e-07 |
| ms_ssim | 0.9248 | 0.9206 |
| psnr | 71.32 | 158.06 |
| mse | 0.00480 | 1.02e-11 |

---

## 5. Commands Executed

**Simulation (dose_experiments notebook):**

The canonical entry point is `notebooks/dose_dependence.ipynb`, which invokes `notebooks/dose.py`:

```python
# In dose_dependence.ipynb cell:
from notebooks import dose
dose.init(nphotons=1e5, loss_fn='nll')
# Simulation runs through dose.run_experiment_with_photons() internally
```

The `dose.init()` function sets:
```python
cfg['data_source'] = 'lines'
cfg['gridsize'] = 2  # Note: dose.py default differs from baseline (gs=1)
cfg['intensity_scale.trainable'] = True
cfg['probe.trainable'] = False
cfg['nepochs'] = 60
```

**Training (baseline run):**

```bash
cd ~/Documents/PtychoPINN

python -m ptycho.train \
    --train_data_file photon_grid_study_20250826_152459/data_p1e5.npz \
    --output_dir photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1 \
    --batch_size 16 \
    --nepochs 50 \
    --gridsize 1 \
    --intensity_scale_trainable True \
    --probe_trainable False
```

**Inference:**

```bash
python -m ptycho.inference \
    --model_path photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/wts.h5.zip \
    --test_data photon_grid_study_20250826_152459/data_p1e5.npz
```

**Overrides from defaults:**
- `gridsize=1` (baseline; dose.py defaults to 2)
- `batch_size=16`
- `nepochs=50` (baseline; dose.py defaults to 60)
- `intensity_scale.trainable=True`
- `probe.trainable=False` (probe frozen)

---

## 6. Artifacts Available

**Dataset root:**
`photon_grid_study_20250826_152459/` (7 NPZ files, ~200 MB total)

**Baseline run directory:**
`photon_grid_study_20250826_152459/results_p1e5/train_1024/trial_1/baseline_run/08-26-2025-16.38.17_baseline_gs1/08-26-2025-16.38.17_baseline_gs1/`

Contents:
- `params.dill` — full parameter snapshot including metrics
- (Checkpoints may include `baseline_model.h5`, `recon.dill` if present)

**Summary artifacts (this response):**
- `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json` — machine-readable snapshot
- `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.md` — human-readable summary

---

## 7. Preferred Handoff

**Artifact placement:**
`plans/active/seed/reports/` or external storage (link in this directory)

**Size estimate:**
~200 MB for all NPZ datasets; ~36 KB for params.dill; negligible for JSON/MD summaries

**Recommendations:**
1. For D0 parity logging, focus on `data_p1e5.npz` as the primary scenario (matches the baseline).
2. The full 7-dose sweep (1e3 through 1e9) is available for extended studies.
3. If external storage is preferred for large files, a manifest with SHA256 checksums is provided above.
4. The `dose_baseline_snapshot.py` script can regenerate summaries for any scenario/dataset combination.

---

## Attachments

- Full JSON snapshot: `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.json`
- Markdown summary: `plans/active/seed/reports/2026-01-22T024002Z/dose_baseline_summary.md`
- Snapshot script: `plans/active/seed/bin/dose_baseline_snapshot.py`
