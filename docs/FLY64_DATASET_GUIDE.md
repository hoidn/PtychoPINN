# FLY64 Experimental Dataset Guide

FLY64 is an experimental ptychography dataset (64×64 pixel resolution) that requires preprocessing for PtychoPINN compatibility.

## Quick Start

**Problem:** Raw FLY64 datasets cause PtychoPINN to fail with zero amplitude predictions.

**Solution:** Always preprocess with format conversion:

```bash
python scripts/tools/transpose_rename_convert_tool.py \
    datasets/fly64/fly001_64_train.npz \
    datasets/fly64/fly001_64_train_converted.npz
```

## Dataset Files

| File | Status | Description | Use Case |
|------|--------|-------------|----------|
| `fly001_64_train.npz` | Raw | Original experimental data (uint16) | **Do not use directly** |
| `fly001_64_train_converted.npz` | Ready | Format-converted for PtychoPINN | **Recommended** |
| `fly001_64_prepared_final_*.npz` | Processed | Subsampled + Y patches added | Use if subsampling acceptable |

## Format Issues & Solutions

### Raw Format Problems
- **Data type:** `uint16` intensity → PtychoPINN expects `float32` amplitude
- **Key naming:** `diff3d` → PtychoPINN expects `diffraction`
- **Missing Y patches:** No ground truth patches for supervised learning

### Preprocessing Pipeline
```bash
# Required: Format conversion
python scripts/tools/transpose_rename_convert_tool.py raw.npz converted.npz

# Optional: Add Y patches for supervised learning
python scripts/tools/generate_patches_tool.py converted.npz final.npz
```

## Usage Example

```bash
# Train with preprocessed FLY64
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly001_64_train_converted.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --train-sizes "512 1024" \
    --output-dir "fly64_study" \
    --skip-data-prep
```

## Validation

Verify your dataset is properly formatted:
```python
import numpy as np
data = np.load('datasets/fly64/fly001_64_train_converted.npz')
assert 'diffraction' in data.keys(), "Missing 'diffraction' key"
assert data['diffraction'].dtype == np.float32, "Wrong data type"
print("✓ FLY64 dataset ready for PtychoPINN")
```

## Troubleshooting

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'diffraction'` | Using raw dataset | Use `transpose_rename_convert_tool.py` |
| Zero amplitude predictions | uint16 data type | Use `transpose_rename_convert_tool.py` |
| Shape mismatch errors | Missing preprocessing | Use converted dataset |

## See Also

- <doc-ref type="workflow-guide">scripts/tools/README.md</doc-ref> - Preprocessing tools
- <doc-ref type="workflow-guide">docs/studies/GENERALIZATION_STUDY_GUIDE.md</doc-ref> - Other datasets
- <doc-ref type="contract">docs/data_contracts.md</doc-ref> - Data format specifications