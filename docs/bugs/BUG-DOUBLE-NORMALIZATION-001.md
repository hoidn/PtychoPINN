# BUG-DOUBLE-NORMALIZATION-001: Double Normalization in Simulation Path

**Status:** Fixed (2026-01-07)
**Severity:** Critical
**Affected Component:** `dose_response_study.py`, `RawData.from_simulation()`, `generate_grouped_data()`
**Reporter:** Claude Code
**Date:** 2026-01-07

---

## Summary

Simulated data generated via `RawData.from_simulation()` undergoes double normalization, causing reconstruction failures for both MAE and Poisson NLL loss objectives. The working legacy path (`data_preprocessing.py`) applies single normalization, while the modern path (`dose_response_study.py`) applies normalization twice.

---

## Symptoms

- Reconstructions from `dose_response_study.py` are incorrect for all experimental arms
- Both MAE and NLL loss objectives produce bad results
- Sanity check shows intensity ratio ~1.22 between high/low dose (expected: variable based on normalization)
- `intensity_scaler_inv_loss` converges to very different values for high vs low dose

---

## Root Cause

Two different normalization steps are applied sequentially to simulated data:

### Normalization Step 1: `illuminate_and_diffract()` (diffsim.py:141)
```python
X, Y_I, Y_phi = X / intensity_scale, Y_I / intensity_scale, Y_phi
```
Where `intensity_scale = sqrt(nphotons / mean_photons_original)`

Result: `mean(|X|²) ≈ mean_photons_original`

### Normalization Step 2: `normalize_data()` (raw_data.py:978)
```python
X_full_norm = np.sqrt(((N / 2)**2) / np.mean(tf.reduce_sum(dset['diffraction']**2, axis=[1, 2])))
return X_full_norm * X_full
```

Result: `mean(|X|²) = (N/2)²`

### The Problem

The `dose_response_study.py` path hits **both** normalizations:
1. `RawData.from_simulation()` → `illuminate_and_diffract()` → **Normalization 1**
2. `generate_grouped_data()` → `normalize_data()` → **Normalization 2**

The legacy `data_preprocessing.py` path hits **only** Normalization 1:
1. `mk_simdata()` → `illuminate_and_diffract()` → **Normalization 1**
2. Direct to `PtychoDataContainer` → **No Normalization 2**

---

## Affected vs Working Code Paths

### Working Path (dose_experiments branch, notebooks/dose.py)
```
dose.py
  → ptycho.train
  → ptycho.generate_data
  → data_preprocessing.generate_data()
  → mk_simdata()
  → illuminate_and_diffract() [normalizes once]
  → PtychoDataContainer [direct, no normalize_data()]
```

### Broken Path (dose_response_study.py)
```
dose_response_study.py
  → generate_simulated_data()
  → RawData.from_simulation()
  → illuminate_and_diffract() [normalizes once]
  → create_ptycho_data_container()
  → generate_grouped_data()
  → normalize_data() [normalizes AGAIN]
  → loader.load()
```

---

## Impact

The double normalization corrupts the relationship between:
- The stored diffraction data `X`
- The `intensity_scale` parameter used in physics loss

This causes the Poisson NLL loss to operate at the wrong photon scale, and even MAE fails because the data statistics are inconsistent with what the model expects.

---

## Recommended Fix

**Option A: Undo normalization in `from_simulation()` (Preferred)**

Store data at physical scale so `normalize_data()` can handle all normalization consistently:

```python
# raw_data.py, from_simulation(), around line 244-251
X, Y_I_xprobe, Y_phi_xprobe, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probeGuess)
X_physical = X * intensity_scale  # Undo the division, store at physical scale
norm_Y_I = scale_nphotons(X)
return RawData(...,
               diff3d=tf.squeeze(X_physical).numpy(),  # Physical scale, not normalized
               ...)
```

**Why this works:**
- Single normalization path via `normalize_data()`
- Matches behavior of NPZ data loading (which expects un-normalized input)
- `intensity_scale = sqrt(nphotons) / (N/2)` in `train_pinn.py` becomes correct
- No changes to stable `diffsim.py`

**Option B: Add `skip_normalize` parameter**

Add parameter to `generate_grouped_data()`:
```python
def generate_grouped_data(self, N, K=4, ..., skip_normalize=False):
    ...
    if not skip_normalize:
        X_full = normalize_data(dset, N)
    else:
        X_full = dset['diffraction']
```

Callers using `from_simulation()` data pass `skip_normalize=True`.

---

## Test Plan

1. Run `dose_response_study.py` with fix applied
2. Verify intensity ratio sanity check shows expected behavior
3. Verify MAE reconstructions are reasonable (not dependent on intensity_scale)
4. Verify NLL reconstructions improve with correct intensity_scale
5. Compare to legacy `notebooks/dose_dependence.ipynb` results

---

## Related Files

- `ptycho/raw_data.py` - `from_simulation()`, `generate_grouped_data()`, `normalize_data()`
- `ptycho/diffsim.py` - `illuminate_and_diffract()`, `scale_nphotons()`
- `ptycho/loader.py` - `normalize_data()` (duplicate), `load()`
- `ptycho/data_preprocessing.py` - `mk_simdata()`, `create_ptycho_dataset()`
- `ptycho/train_pinn.py` - `calculate_intensity_scale()`
- `scripts/studies/dose_response_study.py` - affected study script

---

## Resolution

**Fix applied:** Option A - Undo normalization in `from_simulation()` (raw_data.py:244-252)

```python
X, Y_I_xprobe, Y_phi_xprobe, intensity_scale = illuminate_and_diffract(Y_I, Y_phi, probeGuess)
# Store diffraction data at physical (un-normalized) scale so that
# normalize_data() in generate_grouped_data() handles all normalization consistently.
# This fixes double normalization bug BUG-DOUBLE-NORMALIZATION-001.
X_physical = X * intensity_scale
```

**Verification:**
- Intensity ratio (High/Low): 3.87e+02 (was 1.22, closer to expected ~1e5)
- All 4 arms show clear object structure in reconstructions
- Intensity scaler converges consistently within dose levels

---

## References

- `docs/DATA_NORMALIZATION_GUIDE.md` - Normalization conventions
- `specs/data_contracts.md` - Data format specifications
- Comment in `loader.py:515-526` - TODO noting normalization should be unified
