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
| `fly64_shuffled.npz` | Ready | Complete dataset (10304 images), shuffled | **Recommended for full spatial coverage** |
| `fly64_top_half_shuffled.npz` | Ready | Upper spatial region (5172 points), shuffled | **Spatial subset studies** |
| `fly64_bottom_half_shuffled.npz` | Ready | Lower spatial region (5050 points), shuffled | **Spatial subset studies** |
| **Validation Subsets** | | | |
| `fly64_sequential_train_800.npz` | Ready | Sequential subset (800 images) for training | **GridSize 2 validation** |
| `fly64_sequential_test_200.npz` | Ready | Sequential subset (200 images) for testing | **GridSize 2 validation** |
| `fly64_random_train_800.npz` | Ready | Random subset (800 images) for training | **GridSize 1 validation** |
| `fly64_random_test_200.npz` | Ready | Random subset (200 images) for testing | **GridSize 1 validation** |

## Specialized Datasets

### fly64_shuffled.npz
**Purpose:** The complete fly64 dataset with randomized scan order to eliminate spatial bias in subsampling.

**Created:** All 10,304 scan points from `fly001_64_train_converted.npz`, randomized with seed 42.

**Creation Process:**
1. **Source:** `fly001_64_train_converted.npz` (complete 10304 scan points)
2. **Shuffle:** Randomize order using `shuffle_dataset_tool.py --seed 42`
3. **Result:** 10,304 scan points in random order with full spatial coverage

**Use Case:** Pre-shuffled dataset useful for creating reproducible benchmarks and comparisons. No longer required for gridsize=1 training (as of unified sampling update), but remains valuable for canonical benchmark datasets.

**Key Properties:**
- **Full spatial coverage:** x=[33.5, 198.5], y=[33.1, 198.9] (165×166 range)
- **All relationships preserved:** Each diffraction pattern correctly corresponds to its coordinates
- **Verified shuffling:** Contains `_shuffle_applied` and `_shuffle_seed` metadata

**Validation:**
```python
import numpy as np
data = np.load('datasets/fly64/fly64_shuffled.npz')
assert len(data['xcoords']) == 10304, "Must contain exactly 10304 scan points"
assert data['_shuffle_applied'][0] == True, "Must be shuffled"
print("✓ fly64_shuffled.npz ready for unbiased training")
```

### fly64_bottom_half_shuffled.npz
**Purpose:** Complementary dataset to top_half for spatial subset studies, containing the lower spatial region.

**Created:** All 5,050 scan points from `fly001_64_train_converted.npz` where Y < 114.3, randomized with seed 42.

**Creation Process:**
1. **Source:** `fly001_64_train_converted.npz` (complete 10304 scan points)
2. **Extract:** Select points where Y < 114.3 (5050 points)
3. **Shuffle:** Randomize order using `shuffle_dataset_tool.py --seed 42`
4. **Result:** 5,050 scan points in random order

**Use Case:** Spatial subset studies, complementary to top_half dataset. Enables controlled experiments comparing models trained on different spatial regions.

**Key Properties:**
- **Lower spatial region:** Y-range [33.1, 114.3] (when visualized, appears at TOP of image)
- **No overlap with top_half:** Clear spatial separation at Y=114.3
- **Full X coverage:** x=[33.5, 198.5] (165 unit range)
- **Consistent shuffling:** Same seed (42) as top_half for reproducibility

**Validation:**
```python
import numpy as np
data = np.load('datasets/fly64/fly64_bottom_half_shuffled.npz')
assert len(data['xcoords']) == 5050, "Must contain exactly 5050 scan points"
assert data['ycoords'].max() < 114.3, "All Y values must be below threshold"
assert data['_shuffle_seed'][0] == 42, "Must use consistent seed"
print("✓ fly64_bottom_half_shuffled.npz ready for spatial subset studies")
```

### fly64_top_half_shuffled.npz
**Purpose:** A specialized dataset for studying spatial sampling bias effects in training.

**Created:** From scan points where Y ≥ 114.3 in `fly001_64_train_converted.npz`, randomized with seed 42.

**Creation Process:**
1. **Source:** `fly001_64_train_converted.npz` (10304 total scan points)
2. **Extract:** Select points where Y ≥ 114.3 (5172 points)
3. **Shuffle:** Randomize order using `shuffle_dataset_tool.py --seed 42`
4. **Result:** 5,172 scan points in random order

**Use Case:** Spatial subset studies, complementary to bottom_half dataset. Originally created for studying spatial sampling bias.

**Key Properties:**
- **Upper spatial region:** Y-range [114.3, 198.9] (when visualized, appears at BOTTOM of image)
- **No overlap with bottom_half:** Clear spatial separation at Y=114.3
- **Partial X coverage:** x=[34.1, 198.5] (164.4 unit range)
- **Consistent shuffling:** Same seed (42) as bottom_half for reproducibility

**Creation Commands:**
```bash
# 1. Extract top half (first 5172 scan points)
python -c "
import numpy as np
data = np.load('datasets/fly64/fly001_64_train_converted.npz')
subset = {k: v[:5172] if v.shape and v.shape[0] == 10304 else v for k, v in data.items()}
np.savez_compressed('datasets/fly64/fly64_top_half.npz', **subset)
"

# 2. Shuffle the subset
python scripts/tools/shuffle_dataset_tool.py \
    --input-file datasets/fly64/fly64_top_half.npz \
    --output-file datasets/fly64/fly64_top_half_shuffled.npz \
    --seed 42

# 3. Clean up intermediate file
rm datasets/fly64/fly64_top_half.npz
```

**Validation:**
```python
import numpy as np
data = np.load('datasets/fly64/fly64_top_half_shuffled.npz')
assert len(data['xcoords']) == 5172, "Must contain exactly 5172 scan points"
print("✓ fly64_top_half_shuffled.npz ready for spatial bias studies")
```

## Validation Subsets

### Quick Validation Datasets

For rapid prototyping and validation experiments, smaller subsets (1000 images) with train/test splits are available:

#### Sequential Subsets
**Purpose:** Preserve scan order for experiments requiring specific spatial patterns or debugging.

**Files:**
- `fly64_sequential_train_800.npz`: 800 images from a localized region
- `fly64_sequential_test_200.npz`: 200 images from adjacent region

**Key Properties:**
- **Spatial locality preserved:** Small Y-range ensures neighbors are truly adjacent
- **Non-overlapping regions:** Train and test sets are spatially separated
- **GridSize 2 compatible:** ~200 valid neighbor groups in training set

**Usage:**
```bash
ptycho_train --train_data datasets/fly64/fly64_sequential_train_800.npz \
             --test_data datasets/fly64/fly64_sequential_test_200.npz \
             --gridsize 2 --n_images 200 --nepochs 10 \
             --output_dir gs2_validation
```

#### Random Subsets
**Purpose:** Provide spatially representative samples for standard training.

**Files:**
- `fly64_random_train_800.npz`: 800 randomly sampled images
- `fly64_random_test_200.npz`: 200 randomly sampled images

**Key Properties:**
- **Full spatial coverage:** Both subsets span the entire object area
- **Unbiased sampling:** Pre-shuffled to ensure representativeness
- **General purpose:** Suitable for any gridsize value

**Usage:**
```bash
ptycho_train --train_data datasets/fly64/fly64_random_train_800.npz \
             --test_data datasets/fly64/fly64_random_test_200.npz \
             --gridsize 1 --n_images 800 --nepochs 10 \
             --output_dir gs1_validation
```

**Creation Process:**
These subsets were created from properly converted datasets:
- Sequential: First 1000 images from `fly001_64_train_converted.npz` (format-converted, sequential order), split 80/20
- Random: First 1000 images from `fly64_shuffled.npz` (format-converted, randomized order), split 80/20

Both source datasets have the correct float32 amplitude format required by PtychoPINN.

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
# Train with preprocessed FLY64 (shuffled dataset optional but useful for benchmarks)
./scripts/studies/run_complete_generalization_study.sh \
    --train-data "datasets/fly64/fly64_shuffled.npz" \
    --test-data "datasets/fly64/fly001_64_train_converted.npz" \
    --train-sizes "512 1024" \
    --output-dir "fly64_study" \
    --skip-data-prep

# Alternative: train with unshuffled dataset (may have spatial bias)
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