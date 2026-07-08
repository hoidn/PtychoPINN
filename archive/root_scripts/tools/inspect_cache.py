#!/usr/bin/env python3
"""
Cache file inspector for probe generalization study debugging.
Based on Gemini's recommendation to analyze .groups_cache.npz files.
"""
import numpy as np
import sys

if len(sys.argv) != 2:
    print(f"Usage: python {sys.argv[0]} <path_to_cache_file.npz>")
    sys.exit(1)

filepath = sys.argv[1]
print(f"Inspecting cache file: {filepath}")

try:
    with np.load(filepath, allow_pickle=True) as data:
        total_memory = 0
        for key in data.files:
            item = data[key]
            print(f"  - Key: '{key}'")
            if isinstance(item, np.ndarray):
                memory_mb = item.nbytes / 1024**2
                total_memory += memory_mb
                print(f"    - Shape: {item.shape}")
                print(f"    - Dtype: {item.dtype}")
                print(f"    - Size in memory (MB): {memory_mb:.2f}")
            else:
                print(f"    - Type: {type(item)}")
        print(f"\n  Total memory usage: {total_memory:.2f} MB")
except Exception as e:
    print(f"Error reading file: {e}")