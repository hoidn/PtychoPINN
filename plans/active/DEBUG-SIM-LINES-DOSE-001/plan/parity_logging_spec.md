# Parity Logging Specification v1.0

**Initiative:** DEBUG-SIM-LINES-DOSE-001
**Purpose:** Define implementation-agnostic telemetry schema for comparing sim_lines_4x and dose_experiments pipelines
**Spec Owner:** specs/spec-ptycho-core.md (Normalization Invariants)
**Last Updated:** 2026-01-21

---

## 1. Overview

This specification defines a unified logging schema to capture telemetry at key pipeline stages for amplitude-bias root cause analysis. The schema is designed to be:

- **Implementation-agnostic:** Both sim_lines_4x and legacy dose_experiments pipelines emit identical field names and structures
- **Stage-traceable:** Each processing stage captures stats before/after transformations
- **Probe-aware:** Explicit probe provenance, normalization, and statistics
- **Reproducible:** Checksums, seeds, and provenance chains enable exact replay

---

## 2. JSON Schema Definition

### 2.1 Root Structure

```json
{
  "schema_version": "1.0",
  "timestamp": "<ISO8601Z>",
  "pipeline": "<sim_lines_4x | dose_experiments>",
  "scenario": "<scenario_name>",
  "provenance": { ... },
  "configuration": { ... },
  "probe": { ... },
  "intensity_stages": [ ... ],
  "training": { ... },
  "inference": { ... },
  "comparison": { ... }
}
```

### 2.2 Provenance Block

```json
{
  "provenance": {
    "commit_hash": "<git SHA or 'unknown'>",
    "branch": "<branch name>",
    "dirty": "<true | false>",
    "environment": {
      "python_version": "<3.x.y>",
      "tensorflow_version": "<x.y.z>",
      "keras_version": "<x.y.z>",
      "numpy_version": "<x.y.z>"
    },
    "seeds": {
      "object_seed": "<int>",
      "sim_seed": "<int>",
      "training_seed": "<int | null>"
    },
    "dataset_source": {
      "type": "<simulated | npz_file>",
      "path": "<path or null>",
      "checksum": "<SHA256 of input file or null>",
      "provenance_chain": "<description of data origin>"
    }
  }
}
```

### 2.3 Configuration Block

```json
{
  "configuration": {
    "N": "<int - patch size>",
    "gridsize": "<int - grouping factor>",
    "nphotons": "<float - photon count>",
    "neighbor_count": "<int - KDTree neighbors>",
    "group_count": "<int - requested groups>",
    "batch_size": "<int>",
    "nepochs": "<int>",
    "split_fraction": "<float - train/test ratio>",
    "loss_fn": "<nll | mae>",
    "loss_weights": {
      "nll_weight": "<float>",
      "mae_weight": "<float>",
      "realspace_weight": "<float>",
      "tv_weight": "<float>"
    },
    "intensity_scale_trainable": "<true | false>",
    "legacy_params_cfg_snapshot": { ... }
  }
}
```

### 2.4 Probe Block (REQUIRED)

**Critical for amplitude parity.** This block captures complete probe provenance and statistics.

```json
{
  "probe": {
    "source": {
      "type": "<idealized | custom | npz_loaded>",
      "path": "<file path or null for idealized>",
      "checksum": "<SHA256 of probe file or null>"
    },
    "shape": [N, N],
    "dtype": "<complex64 | complex128>",
    "pre_normalization": {
      "amplitude": {
        "min": "<float>",
        "max": "<float>",
        "mean": "<float>",
        "std": "<float>",
        "l2_norm": "<float>"
      },
      "phase": {
        "min": "<float>",
        "max": "<float>",
        "mean": "<float>",
        "std": "<float>"
      },
      "energy": "<float - sum of |probe|^2>"
    },
    "normalization": {
      "method": "<set_default_probe | normalize_probe_guess | none>",
      "probe_scale": "<float>",
      "probe_big": "<true | false>",
      "probe_mask": "<true | false>",
      "mask_stats": {
        "nonzero_count": "<int>",
        "total_count": "<int>",
        "coverage_fraction": "<float>"
      },
      "computed_norm_factor": "<float - scale * mean(mask * |probe|)>"
    },
    "post_normalization": {
      "amplitude": {
        "min": "<float>",
        "max": "<float>",
        "mean": "<float>",
        "std": "<float>",
        "l2_norm": "<float>"
      },
      "phase": {
        "min": "<float>",
        "max": "<float>",
        "mean": "<float>",
        "std": "<float>"
      },
      "energy": "<float - sum of |probe|^2>"
    }
  }
}
```

### 2.5 Intensity Stages Block

Each stage records statistics at a specific pipeline point per `specs/spec-ptycho-core.md §Normalization Invariants`.

```json
{
  "intensity_stages": [
    {
      "name": "<stage_name>",
      "source": "<module::function or description>",
      "stats": {
        "shape": [B, N, N, C],
        "dtype": "<float32 | float64>",
        "min": "<float>",
        "max": "<float>",
        "mean": "<float>",
        "std": "<float>",
        "nan_count": "<int>",
        "finite_count": "<int>",
        "total_count": "<int>"
      },
      "metadata": {
        "count": "<int - batch/sample count>",
        "gridsize": "<int | null>",
        "group_limit": "<int | null>"
      }
    }
  ]
}
```

**Required stages (in order):**
1. `raw_diffraction` — Direct from RawData/simulation
2. `grouped_diffraction` — After `generate_grouped_data()`
3. `grouped_X_full` — After `normalize_data()`
4. `container_X` — Inside `PtychoDataContainer`
5. `training_labels_Y_amp` — Y_amp fed to model during training
6. `inference_prediction` — Model output amplitude

**Optional stages:**
- `pre_normalization_diffraction` — Before any normalize_data call
- `training_labels_Y_I` — Y_I (intensity) fed to model
- `scaler_output` — After IntensityScaler layer

### 2.6 Stage Ratios Block

Computed from intensity_stages to detect normalization breaks.

```json
{
  "stage_ratios": {
    "raw_to_grouped": "<float - grouped_mean / raw_mean>",
    "grouped_to_normalized": "<float - normalized_mean / grouped_mean>",
    "normalized_to_container": "<float>",
    "prediction_to_truth": "<float - pred_mean / truth_mean>"
  },
  "full_chain_product": "<float - product of all ratios>",
  "symmetry_check": {
    "expected": 1.0,
    "actual": "<float>",
    "tolerance": 0.05,
    "violated": "<true | false>",
    "primary_deviation_source": "<stage name with largest deviation>"
  }
}
```

### 2.7 Intensity Scale Block

```json
{
  "intensity_scale": {
    "dataset_derived": {
      "formula": "sqrt(nphotons / E_batch[sum_xy |X|^2])",
      "batch_mean_sum_intensity": "<float>",
      "computed_scale": "<float>",
      "spec_reference": "specs/spec-ptycho-core.md §Normalization Invariants"
    },
    "fallback": {
      "formula": "sqrt(nphotons) / (N/2)",
      "computed_scale": "<float>"
    },
    "used_mode": "<dataset_derived | fallback>",
    "recorded_in_bundle": "<float>",
    "model_exp_log_scale": "<float - exp(log_scale) from IntensityScaler>",
    "delta_bundle_vs_model": "<float>",
    "ratio_bundle_vs_model": "<float>"
  }
}
```

### 2.8 Training Block

```json
{
  "training": {
    "epochs_completed": "<int>",
    "early_stopped": "<true | false>",
    "early_stop_epoch": "<int | null>",
    "final_loss": "<float>",
    "final_val_loss": "<float>",
    "nan_detected": "<true | false>",
    "first_nan_epoch": "<int | null>",
    "history_artifact": "<relative path to history.json>",
    "checkpoint_artifact": "<relative path to model checkpoint>"
  }
}
```

### 2.9 Inference Block

```json
{
  "inference": {
    "group_limit": "<int>",
    "stitched_shape": [H, W],
    "amplitude": {
      "min": "<float>",
      "max": "<float>",
      "mean": "<float>",
      "std": "<float>",
      "nan_count": "<int>"
    },
    "phase": {
      "min": "<float>",
      "max": "<float>",
      "mean": "<float>",
      "std": "<float>",
      "nan_count": "<int>"
    },
    "artifacts": {
      "amplitude_npy": "<path>",
      "phase_npy": "<path>",
      "amplitude_png": "<path>",
      "phase_png": "<path>"
    }
  }
}
```

### 2.10 Comparison Block (when ground truth available)

```json
{
  "comparison": {
    "ground_truth_source": "<simulated_object | external_npz | none>",
    "crop_metadata": {
      "target_size": "<int>",
      "start": [y, x],
      "end": [y, x]
    },
    "amplitude": {
      "mae": "<float>",
      "rmse": "<float>",
      "max_abs": "<float>",
      "pearson_r": "<float | null>",
      "bias_summary": {
        "mean": "<float - truth - pred>",
        "median": "<float>",
        "p05": "<float>",
        "p95": "<float>"
      },
      "scaling_analysis": {
        "truth_to_pred_ratio_mean": "<float>",
        "least_squares_scalar": "<float>",
        "rescaled_mae": "<float>"
      }
    },
    "phase": { ... }
  }
}
```

---

## 3. Capture Points

| Stage | Module/Function | When to Capture |
|-------|----------------|-----------------|
| raw_diffraction | `RawData.from_simulation()` or `RawData.from_file()` | After simulation/load, before grouping |
| probe (pre-norm) | `make_probe()` or NPZ load | Before any normalization |
| probe (post-norm) | `normalize_probe_guess()` or `set_default_probe()` | After normalization applied |
| grouped_diffraction | `RawData.generate_grouped_data()` | After KDTree grouping |
| grouped_X_full | `normalize_data()` | After L2 normalization |
| container_X | `PtychoDataContainer` init | When loading into container |
| intensity_scale | `calculate_intensity_scale()` | Before training start |
| training_labels | `Trainer` or manual loop | First batch Y_amp/Y_I |
| inference_prediction | `run_inference_and_reassemble()` | After stitching |

---

## 4. Maintainer Coordination Protocol

### 4.1 Required Artifacts from Legacy Pipeline

To achieve dataset parity, the maintainer must provide:

1. **Simulation outputs:**
   - `raw_data.npz` — Complete RawData with diffraction, coords, probe
   - Checksum: `sha256sum raw_data.npz`

2. **Training artifacts:**
   - `parity_telemetry.json` — Following this spec's schema
   - `history.json` — Per-epoch loss/metrics
   - `model_checkpoint/` — Saved weights directory

3. **Inference outputs:**
   - `amplitude.npy`, `phase.npy` — Stitched reconstructions
   - `ground_truth_amp.npy`, `ground_truth_phase.npy` — If simulated

4. **Configuration snapshot:**
   - `config.json` — Complete params.cfg + dataclass config dump

### 4.2 Commands Template for Maintainer

```bash
# Environment requirements
python --version  # Record exact version
pip list | grep -E "tensorflow|keras|numpy"  # Record package versions

# Generate artifacts with telemetry
python scripts/dose_experiments_with_telemetry.py \
  --output-dir /path/to/artifacts/ \
  --parity-log parity_telemetry.json \
  --scenario gs2_ideal \
  --nphotons 1e9 \
  --gridsize 2 \
  --nepochs 60

# Checksums
sha256sum /path/to/artifacts/*.npz > checksums.txt
sha256sum /path/to/artifacts/*.npy >> checksums.txt
```

### 4.3 Delivery Location

```
plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<timestamp>/dose_experiments_ground_truth/
├── README.md              # Commands used, environment notes, deviations
├── checksums.txt          # SHA256 of all data files
├── parity_telemetry.json  # Following this spec
├── config.json            # Full configuration snapshot
├── raw_data.npz           # Simulation output
├── history.json           # Training history
├── model_checkpoint/      # Saved model
├── amplitude.npy          # Inference output
├── phase.npy              # Inference output
├── ground_truth_amp.npy   # Ground truth (if simulated)
└── ground_truth_phase.npy # Ground truth (if simulated)
```

---

## 5. Dataset Parity Guidance

### 5.1 Ideal Case: Same Input Data

When exact dataset parity is achievable:

1. **Same NPZ file:** Use identical `raw_data.npz` for both pipelines
2. **Checksum verification:** `sha256sum` must match before/after transfer
3. **Seed alignment:** Use identical object_seed and sim_seed
4. **Provenance chain:** Document full path from simulation to comparison

### 5.2 Non-Ideal Case: Two-Track Comparison

When exact parity is impossible (e.g., legacy environment cannot run sim_lines simulator):

**Track A — Legacy Pipeline on Legacy Dataset:**
```
[Legacy Sim] → [Legacy Train] → [Legacy Infer] → artifacts_legacy/
```
- Run entirely in legacy TF/Keras environment
- Document exact parameters and seeds
- Emit parity_telemetry.json per this spec

**Track B — Local Pipeline on Same Dataset (or Equivalent):**
```
[Same NPZ or Equivalent Params] → [Local Train] → [Local Infer] → artifacts_local/
```

**Comparison Analysis:**
1. If using same NPZ: Compare at every stage, identify first divergence point
2. If using equivalent params: Compare only final metrics + probe stats, flag dataset generation as potential source

### 5.3 Parity Checklist

| Item | Same NPZ | Equivalent Params |
|------|----------|-------------------|
| raw_diffraction stats within 1% | REQUIRED | N/A |
| probe stats within 1% | REQUIRED | COMPARE |
| grouped_X_full stats within 5% | REQUIRED | COMPARE |
| intensity_scale within 1% | REQUIRED | COMPARE |
| Final amplitude MAE | COMPARE | COMPARE |
| pearson_r | COMPARE | COMPARE |

---

## 6. Versioning

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-21 | Initial spec with probe logging, maintainer coordination, two-track plan |

---

## 7. References

- `specs/spec-ptycho-core.md` §Normalization Invariants — Intensity scale formulas
- `docs/findings.md` CONFIG-001 — Legacy params.cfg bridging requirement
- `plans/active/DEBUG-SIM-LINES-DOSE-001/implementation.md` — Phase D checklist
- `inbox/request_dose_experiments_ground_truth_2026-01-22T014445Z.md` — Prior maintainer request
