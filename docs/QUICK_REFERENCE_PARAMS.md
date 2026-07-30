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

### Direct RawData Usage (NEW - with explicit gridsize)
```python
# After refactoring - no params.cfg needed!
data.generate_grouped_data(N=64, gridsize=2)  # ← Pass explicitly
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

## Links
- Full details: `docs/TROUBLESHOOTING.md#shape-mismatch-errors`
- Configuration guide: `docs/CONFIGURATION.md`
