# Migration Guide: Efficient Coordinate Grouping

## Overview

The coordinate grouping implementation in PtychoPINN has been completely rewritten for efficiency. This guide helps you migrate from the old caching-based system to the new high-performance implementation.

## What Changed?

### Old Implementation (v1.x)
- **Strategy:** "Group-then-sample" - computed ALL possible groups first
- **Performance:** Slow first run (up to minutes for large datasets)
- **Memory:** High usage (stored all groups in memory and cache)
- **Caching:** Created `*.groups_cache.npz` files
- **Reproducibility:** Not supported

### New Implementation (v2.0)
- **Strategy:** "Sample-then-group" - samples seed points first
- **Performance:** Fast every run (seconds for any dataset size)
- **Memory:** Minimal usage (only stores sampled groups)
- **Caching:** No cache files needed
- **Reproducibility:** Supported via `seed` parameter

## Performance Improvements

Based on benchmarks with real datasets:

| Dataset Size | Old Time | New Time | Improvement |
|-------------|----------|----------|-------------|
| 1K points   | ~2s      | 0.01s    | 200x faster |
| 10K points  | ~30s     | 0.05s    | 600x faster |
| 100K points | ~5min    | 0.3s     | 1000x faster |

Memory usage reduced by 10-100x depending on dataset size.

## Migration Steps

### 1. Clean Up Old Cache Files

Old cache files are no longer needed and can be safely deleted:

```bash
# Find all cache files in your project
find . -name "*.groups_cache.npz" -type f

# Remove them (after verifying the list)
find . -name "*.groups_cache.npz" -type f -delete
```

### 2. Update Your Code

#### Basic Usage (No Changes Required)

If you're using the standard interface, your code continues to work:

```python
# This still works exactly the same
grouped_data = raw_data.generate_grouped_data(
    N=64,
    K=7,
    nsamples=1000
)
```

#### Adding Reproducibility

To get reproducible results across runs, use the new `seed` parameter:

```python
# Old way (no reproducibility)
grouped_data = raw_data.generate_grouped_data(
    N=64,
    K=7,
    nsamples=1000
)

# New way (reproducible)
grouped_data = raw_data.generate_grouped_data(
    N=64,
    K=7,
    nsamples=1000,
    seed=42  # Any integer for reproducible sampling
)
```

#### Dataset Path Parameter

The `dataset_path` parameter is kept for backward compatibility but is no longer used:

```python
# Still works but dataset_path is ignored
grouped_data = raw_data.generate_grouped_data(
    N=64,
    K=7,
    nsamples=1000,
    dataset_path="/path/to/data.npz"  # Ignored, kept for compatibility
)
```

### 3. Update Configuration Files

If you have YAML configuration files, you can add the seed parameter:

```yaml
# Old configuration
training:
  n_images: 1000
  K: 7
  gridsize: 2

# New configuration (with reproducibility)
training:
  n_images: 1000
  K: 7
  gridsize: 2
  seed: 42  # Optional: for reproducible training
```

### 4. Verify Performance

Run the benchmark script to verify improvements:

```bash
python scripts/benchmark_grouping.py
```

Expected output:
- First-run performance should be fast (no cache generation)
- Subsequent runs should have identical performance
- No cache files should be created

## Troubleshooting

### Q: I'm getting different results between runs

**A:** This is expected behavior without a seed. To get reproducible results:
```python
grouped_data = raw_data.generate_grouped_data(..., seed=42)
```

### Q: Can I still use my old cache files?

**A:** No, the new implementation doesn't use cache files at all. Delete them to save disk space.

### Q: Is the output format the same?

**A:** Yes, the output dictionary structure and array shapes are identical. Only the internal implementation changed.

### Q: Will this affect my trained models?

**A:** No, existing trained models are unaffected. The data format remains the same.

### Q: What if I need the exact same groups as before?

**A:** The old caching approach didn't guarantee specific groups either (it randomly sampled from all groups). Use the seed parameter for reproducibility going forward.

## Benefits Summary

✅ **Faster Development:** No more waiting for cache generation  
✅ **Less Disk Usage:** No cache files cluttering your filesystem  
✅ **Better Reproducibility:** Seed parameter for deterministic results  
✅ **Lower Memory:** 10-100x reduction in RAM usage  
✅ **Simpler Codebase:** 300 lines of code removed  

## Support

If you encounter any issues during migration:

1. Check that you're using the latest version
2. Verify all cache files have been removed
3. Try running with a specific seed for debugging
4. Report issues at the project repository

## Version Compatibility

- **PtychoPINN >= 2.0:** New efficient implementation
- **PtychoPINN < 2.0:** Old caching implementation

The new implementation maintains full backward compatibility with existing code while providing dramatic performance improvements.