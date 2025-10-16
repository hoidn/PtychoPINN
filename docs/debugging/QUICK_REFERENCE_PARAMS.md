# Quick Reference: params.cfg Initialization

## ⚠️ The Golden Rule
**Always call `update_legacy_dict()` BEFORE any data operations!**

## Common Scenarios

### Training Script
```python
from ptycho.config.config import update_legacy_dict
config = setup_configuration(args, yaml_path)
update_legacy_dict(params.cfg, config)  # ← DO THIS FIRST!
data = load_data(...)  # Now safe to load data
```

### Inference Script
```python
model_manager.load_model(model_path)
config = model_manager.config
update_legacy_dict(params.cfg, config)  # ← Sync params with loaded model
data = load_data(...)
```

### Test Code
```python
def setUp(self):
    params.cfg.clear()  # Clean slate
    config = TrainingConfig(model=ModelConfig(gridsize=2))
    update_legacy_dict(params.cfg, config)  # ← Initialize for test
```

### Direct RawData Usage (NEW - with explicit gridsize)
```python
# After refactoring - no params.cfg needed!
data.generate_grouped_data(N=64, gridsize=2)  # ← Pass explicitly
```

## Red Flags in Code Review 🚩

```python
# 🚩 BAD: Reading params without initialization
gridsize = params.get('gridsize')  # Where was this set?

# 🚩 BAD: Assuming params are set
def my_function():
    C = params.get('gridsize') ** 2  # Hidden dependency!

# 🚩 BAD: Setting params in random places  
params.cfg['gridsize'] = 2  # Should use update_legacy_dict()
```

## Green Flags in Code Review ✅

```python
# ✅ GOOD: Explicit parameter
def my_function(gridsize: int):
    C = gridsize ** 2  # No hidden state!

# ✅ GOOD: Document dependencies
def legacy_function():
    """
    ⚠️ Requires params.cfg['gridsize'] to be set.
    Call update_legacy_dict() before this function.
    """
    gridsize = params.get('gridsize')

# ✅ GOOD: Fail fast with helpful message
if params.get('gridsize') is None:
    raise ValueError(
        "params.cfg['gridsize'] not set. "
        "Call update_legacy_dict(params.cfg, config) first."
    )
```

## Debugging Shape Mismatches

### Quick Check
```python
print(f"Config: {config.model.gridsize}")
print(f"Params: {params.cfg.get('gridsize', 'NOT SET')}")
# These should match!
```

### Expected Shapes
| gridsize | C | Shape |
|----------|---|-------|
| 1 | 1 | `(batch, 64, 64, 1)` |
| 2 | 4 | `(batch, 64, 64, 4)` |
| 3 | 9 | `(batch, 64, 64, 9)` |

### Common Fix
```python
# If getting (*, *, *, 1) instead of (*, *, *, 4):
update_legacy_dict(params.cfg, config)  # You forgot this!
```

## The 66-File Problem

**Why we still have params.cfg:**
- 66+ files depend on it
- Can't refactor all at once
- Gradual migration in progress

**The Migration Path:**
1. New code: Use explicit parameters
2. Legacy code: Document params dependencies  
3. Eventually: Remove params.cfg entirely

**Your Part:**
- Don't add new params.get() calls
- Document any you can't remove
- Prefer explicit parameters

## Links
- Full details: `docs/TROUBLESHOOTING.md#shape-mismatch-errors`
- Migration guide: `docs/DEVELOPER_GUIDE.md#configuration-migration`
- Test template: `tests/test_template_gridsize.py`