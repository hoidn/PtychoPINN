# Troubleshooting Guide

## Shape Mismatch Errors

### Problem: Getting (batch, 64, 64, 1) instead of (batch, 64, 64, 4) with gridsize=2

**Symptom:**
```python
# Expected with gridsize=2:
X.shape = (1024, 64, 64, 4)  # 4 channels

# Actually getting:
X.shape = (1024, 64, 64, 1)  # Only 1 channel!
```

**Root Cause:**
The `generate_grouped_data()` method depends on `params.cfg['gridsize']` being set, but this isn't always initialized properly.

**Quick Diagnosis:**
```python
# Add this debug code before the error:
print(f"Config gridsize: {config.model.gridsize}")
print(f"Params gridsize: {params.cfg.get('gridsize', 'NOT SET')}")
# If these don't match, you've found the problem!
```

**Solutions by Context:**

#### In Training Scripts
```python
# Ensure this is called BEFORE create_ptycho_data_container:
from ptycho.config.config import update_legacy_dict
update_legacy_dict(params.cfg, config)
```

#### In Inference Scripts  
```python
# After loading the model, ensure params are updated:
model_manager.load_model(model_path)
params.cfg['gridsize'] = loaded_config.model.gridsize
```

#### In Test Code
```python
# Explicitly set before calling generate_grouped_data:
params.cfg['gridsize'] = 2  # or whatever gridsize you're testing
```

#### In Workflow Scripts (like run_complete_generalization_study.sh)
```bash
# Verify the config file has the correct gridsize:
grep gridsize <your_config>.yaml
# Should show: gridsize: 2
```

**Prevention Checklist:**
- [ ] Is `update_legacy_dict()` called before data loading?
- [ ] Does your config file specify the correct gridsize?
- [ ] Are you mixing models trained with different gridsizes?
- [ ] Is params.cfg being cleared/reset between runs?

**Debug Logging:**
Add this to trace the issue:
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# In your script
logger = logging.getLogger(__name__)
logger.debug(f"Before data load - params.cfg: {params.cfg}")
logger.debug(f"Config object: gridsize={config.model.gridsize}")
```

---

## Oversampling Not Working

### Problem: A request for more groups than selected rows is rejected

**Symptom:**
```
Requesting 1024 groups but only 128 points available ...
K choose C oversampling is required but not enabled.
```

**Root Cause:**
The K-choose-C branch is entered only when `training_groups > n_points` and
`gridsize > 1`, where `n_points` is the selected raw-row count. That branch
requires explicit `enable_oversampling: true` and `K >= C`, with
`C = gridsize²` and K taken from `neighbor_pool_size` (or `neighbor_count`
when the pool size is omitted).

**Quick Check:**
```python
# These conditions must be met for the oversampling branch:
C = gridsize ** 2  # e.g., 4 for gridsize=2
K = neighbor_pool_size or neighbor_count  # e.g., 7
needs_oversampling = training_groups > n_points and C > 1
assert needs_oversampling, "Oversampling is triggered only above the selected-row count"
assert enable_oversampling, "Oversampling requires explicit opt-in"
assert C > 1, "Need gridsize > 1 for oversampling"
assert K >= C, "Need K >= C for valid groups"

# Candidate diversity, not an enforced output cap:
candidate_upper_bound = n_points * math.comb(K, C)
print(f"At most {candidate_upper_bound} per-anchor combinations before overlap")
```

If the requested group count exceeds the generated unique combination pool,
the current sampler logs a warning and samples combinations with replacement.
The value above therefore describes potential unique diversity, not a maximum
allowed `training_groups`.

**Solution:**
```yaml
# Put numeric and Boolean values in the nested training YAML.
model:
  gridsize: 2                 # C=4
sampling:
  neighbor_count: 7           # Neighbor query size
  enable_oversampling: true   # Explicit opt-in
  neighbor_pool_size: 7       # K=7, C(7,4)=35 combinations
  train_raw_selection: 128     # Raw images selected
  training_groups: 1024        # Grouped samples requested
```

---

## Global Params Not Updated

### Problem: Changes to config don't affect legacy modules

**Symptom:**
```python
from dataclasses import replace
config = config.model_copy(
    update={"model": replace(config.model, gridsize=2)}
)
# But legacy module still uses gridsize=1
```

**Root Cause:**
The one-way bridge from the resolved configuration to `params.cfg` was not
called.

**Solution:**
```python
# After ANY config changes, update legacy params:
from ptycho.config.config import update_legacy_dict
update_legacy_dict(params.cfg, config)

# Verify it worked:
assert params.cfg['gridsize'] == config.model.gridsize
```

**Common Mistake:**
```python
# WRONG: deriving a replacement after update_legacy_dict without bridging it
update_legacy_dict(params.cfg, config)
config = config.model_copy(
    update={"model": replace(config.model, gridsize=2)}
)

# RIGHT: derive the complete record first, then bridge it
config = config.model_copy(
    update={"model": replace(config.model, gridsize=2)}
)
update_legacy_dict(params.cfg, config)
```

---

## Configuration Precedence Issues

### Problem: CLI arguments ignored when using YAML config

**Symptom:**
```bash
# This doesn't work as expected:
ptycho_train --config config.yaml --backend pytorch
# Still uses backend from config.yaml!
```

**Root Cause:**
Configuration precedence may be incorrect in some scripts.

**Expected Precedence (highest to lowest):**
1. CLI arguments
2. YAML config file  
3. Default values

**Debug:**
```python
# Check what values are being used:
print(f"Args: {args.backend}")
print(f"YAML: {yaml_config.get('backend')}")
print(f"Final: {config.backend}")
```

---

## Quick Debugging Commands

### Check Current Params State
```python
python -c "from ptycho import params; print(params.cfg)"
```

### Verify Config File
```bash
# Check gridsize in config
python -c "import yaml; print(yaml.safe_load(open('configs/comparison_config.yaml')))"
```

### Test Data Generation
```python
from ptycho.raw_data import RawData
from ptycho import params

# Test with explicit gridsize
params.cfg['gridsize'] = 2
data = RawData.from_file('your_data.npz')
result = data.generate_grouped_data(N=64, K=7, nsamples=100, gridsize=2)
print(f"Shape: {result['diffraction'].shape}")  # Should be (100, 64, 64, 4)
```

---

## When to File a Bug Report

File an issue if you encounter:
1. Shape mismatches even after following this guide
2. `update_legacy_dict()` not syncing values correctly  
3. Inconsistent behavior between training and inference
4. Parameters silently reverting to defaults

Include in your report:
- Output of the debug commands above
- Your config file
- The exact command you ran
- Full error traceback
