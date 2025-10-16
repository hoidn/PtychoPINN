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
grep gridsize configs/gridsize2_minimal.yaml
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

### Problem: Can't create more groups than images (e.g., 1024 groups from 128 images)

**Symptom:**
```
Requested 1024 groups but only getting 128
```

**Root Cause:**
Oversampling requires `gridsize > 1` AND `K > C` (where C = gridsize²).

**Quick Check:**
```python
# These conditions must be met for oversampling:
C = gridsize ** 2  # e.g., 4 for gridsize=2
K = neighbor_count  # e.g., 7
assert C > 1, "Need gridsize > 1 for oversampling"
assert K >= C, "Need K >= C for valid groups"
assert K > C, "Need K > C for oversampling combinations"

# Maximum possible groups:
max_groups = n_images * math.comb(K, C)
print(f"Can create up to {max_groups} groups from {n_images} images")
```

**Solution:**
```bash
# Use these parameters for oversampling:
--gridsize 2       # Sets C=4
--neighbor-count 7  # Sets K=7, giving C(7,4)=35 combinations per seed
--n-subsample 128   # Number of images to load
--n-groups 1024     # Number of groups to create (can be > n_subsample!)
```

---

## Global Params Not Updated

### Problem: Changes to config don't affect legacy modules

**Symptom:**
```python
config.model.gridsize = 2  # Set in config
# But legacy module still uses gridsize=1
```

**Root Cause:**
The one-way bridge from dataclass to params.cfg wasn't called.

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
# WRONG: Updating config after update_legacy_dict
update_legacy_dict(params.cfg, config)
config.model.gridsize = 2  # This change won't propagate!

# RIGHT: Update config first, then sync
config.model.gridsize = 2
update_legacy_dict(params.cfg, config)
```

---

## Configuration Precedence Issues

### Problem: CLI arguments ignored when using YAML config

**Symptom:**
```bash
# This doesn't work as expected:
ptycho_train --config config.yaml --gridsize 4
# Still uses gridsize from config.yaml!
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
print(f"Args: {args.gridsize}")
print(f"YAML: {yaml_config.get('gridsize')}")
print(f"Final: {config.model.gridsize}")
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
python -c "import yaml; print(yaml.safe_load(open('configs/gridsize2_minimal.yaml')))"
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