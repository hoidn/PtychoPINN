#!/usr/bin/env python
"""
Benchmark to compare translation performance between ImageProjectiveTransformV3 and pure TF XLA version.
"""

import tensorflow as tf
import numpy as np
import time
import sys
sys.path.insert(0, '.')

from ptycho.tf_helper import translate_core
from ptycho.projective_warp_xla import translate_xla

def benchmark_translation(batch_sizes=[1, 8, 32, 128], image_sizes=[64, 128, 256], num_warmup=5, num_runs=20):
    """Compare performance of different translation implementations."""
    
    print("=" * 80)
    print("Translation Performance Benchmark")
    print("=" * 80)
    print("\nComparing:")
    print("1. ImageProjectiveTransformV3 (native TF operation)")
    print("2. Pure TF XLA implementation (projective_warp_xla)")
    print()
    
    results = {}
    
    for img_size in image_sizes:
        for batch_size in batch_sizes:
            print(f"\n{'='*60}")
            print(f"Testing: Batch={batch_size}, Image Size={img_size}x{img_size}")
            print('='*60)
            
            # Create test data
            images = tf.random.normal((batch_size, img_size, img_size, 1), dtype=tf.float32)
            translations = tf.random.normal((batch_size, 2), dtype=tf.float32) * 5.0  # Random translations up to 5 pixels
            
            # Test ImageProjectiveTransformV3 (if available)
            try:
                print("\n1. ImageProjectiveTransformV3:")
                # Warmup
                for _ in range(num_warmup):
                    _ = translate_core(images, translations, use_xla_workaround=False)
                
                # Benchmark
                times = []
                for _ in range(num_runs):
                    start = time.perf_counter()
                    _ = translate_core(images, translations, use_xla_workaround=False)
                    elapsed = (time.perf_counter() - start) * 1000  # Convert to ms
                    times.append(elapsed)
                
                mean_time = np.mean(times)
                std_time = np.std(times)
                print(f"   Mean: {mean_time:.3f} ms ± {std_time:.3f} ms")
                print(f"   Min: {np.min(times):.3f} ms, Max: {np.max(times):.3f} ms")
                
                results[f"ImageProjectiveTransformV3_B{batch_size}_S{img_size}"] = {
                    'mean': mean_time,
                    'std': std_time
                }
                
            except Exception as e:
                print(f"   Failed: {str(e)[:100]}")
                results[f"ImageProjectiveTransformV3_B{batch_size}_S{img_size}"] = {
                    'mean': float('inf'),
                    'std': 0
                }
            
            # Test Pure TF XLA implementation
            print("\n2. Pure TF XLA (projective_warp_xla):")
            
            # First without JIT compilation
            print("   a) Without JIT:")
            # Warmup
            for _ in range(num_warmup):
                _ = translate_xla(images, translations, use_jit=False)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = translate_xla(images, translations, use_jit=False)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"      Mean: {mean_time:.3f} ms ± {std_time:.3f} ms")
            print(f"      Min: {np.min(times):.3f} ms, Max: {np.max(times):.3f} ms")
            
            results[f"XLA_NoJIT_B{batch_size}_S{img_size}"] = {
                'mean': mean_time,
                'std': std_time
            }
            
            # With JIT compilation
            print("   b) With JIT compilation:")
            # Warmup (includes compilation)
            for _ in range(num_warmup):
                _ = translate_xla(images, translations, use_jit=True)
            
            # Benchmark
            times = []
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = translate_xla(images, translations, use_jit=True)
                elapsed = (time.perf_counter() - start) * 1000
                times.append(elapsed)
            
            mean_time = np.mean(times)
            std_time = np.std(times)
            print(f"      Mean: {mean_time:.3f} ms ± {std_time:.3f} ms")
            print(f"      Min: {np.min(times):.3f} ms, Max: {np.max(times):.3f} ms")
            
            results[f"XLA_JIT_B{batch_size}_S{img_size}"] = {
                'mean': mean_time,
                'std': std_time
            }
            
            # Calculate speedup
            if f"ImageProjectiveTransformV3_B{batch_size}_S{img_size}" in results:
                v3_time = results[f"ImageProjectiveTransformV3_B{batch_size}_S{img_size}"]['mean']
                xla_nojit_time = results[f"XLA_NoJIT_B{batch_size}_S{img_size}"]['mean']
                xla_jit_time = results[f"XLA_JIT_B{batch_size}_S{img_size}"]['mean']
                
                if v3_time != float('inf'):
                    print(f"\n   Speedup vs ImageProjectiveTransformV3:")
                    print(f"      XLA (no JIT): {v3_time/xla_nojit_time:.2f}x {'faster' if v3_time > xla_nojit_time else 'slower'}")
                    print(f"      XLA (with JIT): {v3_time/xla_jit_time:.2f}x {'faster' if v3_time > xla_jit_time else 'slower'}")
    
    return results


def print_summary(results):
    """Print a summary table of results."""
    print("\n" + "=" * 80)
    print("SUMMARY TABLE")
    print("=" * 80)
    print("\nAll times in milliseconds (lower is better):")
    print("\n{:<20} {:<10} {:<10} {:>12} {:>12} {:>12}".format(
        "Configuration", "Batch", "Size", "ImageProjV3", "XLA_NoJIT", "XLA_JIT"))
    print("-" * 80)
    
    # Parse and organize results
    configs = set()
    for key in results.keys():
        parts = key.split('_')
        if len(parts) >= 2:
            batch = parts[-2].replace('B', '')
            size = parts[-1].replace('S', '')
            configs.add((int(batch), int(size)))
    
    for batch, size in sorted(configs):
        v3_key = f"ImageProjectiveTransformV3_B{batch}_S{size}"
        xla_nojit_key = f"XLA_NoJIT_B{batch}_S{size}"
        xla_jit_key = f"XLA_JIT_B{batch}_S{size}"
        
        v3_time = results.get(v3_key, {}).get('mean', float('inf'))
        xla_nojit_time = results.get(xla_nojit_key, {}).get('mean', 0)
        xla_jit_time = results.get(xla_jit_key, {}).get('mean', 0)
        
        v3_str = f"{v3_time:.3f}" if v3_time != float('inf') else "N/A"
        
        print("{:<20} {:<10} {:<10} {:>12} {:>12} {:>12}".format(
            f"B{batch}_S{size}",
            batch,
            f"{size}x{size}",
            v3_str,
            f"{xla_nojit_time:.3f}",
            f"{xla_jit_time:.3f}"
        ))
    
    print("\n" + "=" * 80)
    print("KEY INSIGHTS:")
    print("=" * 80)
    
    # Calculate average speedups
    speedups_nojit = []
    speedups_jit = []
    
    for batch, size in configs:
        v3_key = f"ImageProjectiveTransformV3_B{batch}_S{size}"
        xla_nojit_key = f"XLA_NoJIT_B{batch}_S{size}"
        xla_jit_key = f"XLA_JIT_B{batch}_S{size}"
        
        v3_time = results.get(v3_key, {}).get('mean', float('inf'))
        xla_nojit_time = results.get(xla_nojit_key, {}).get('mean', 0)
        xla_jit_time = results.get(xla_jit_key, {}).get('mean', 0)
        
        if v3_time != float('inf') and xla_nojit_time > 0:
            speedups_nojit.append(v3_time / xla_nojit_time)
        if v3_time != float('inf') and xla_jit_time > 0:
            speedups_jit.append(v3_time / xla_jit_time)
    
    if speedups_nojit:
        avg_speedup_nojit = np.mean(speedups_nojit)
        print(f"\n1. ImageProjectiveTransformV3 vs XLA (no JIT):")
        print(f"   Average: {avg_speedup_nojit:.2f}x {'faster' if avg_speedup_nojit > 1 else 'slower'}")
        
    if speedups_jit:
        avg_speedup_jit = np.mean(speedups_jit)
        print(f"\n2. ImageProjectiveTransformV3 vs XLA (with JIT):")
        print(f"   Average: {avg_speedup_jit:.2f}x {'faster' if avg_speedup_jit > 1 else 'slower'}")
    
    print("\nNOTE: ImageProjectiveTransformV3 is not XLA-compatible, so it cannot be")
    print("      used in XLA-compiled graphs. The pure TF version enables full graph")
    print("      optimization and fusion, which can provide additional benefits beyond")
    print("      just the translation operation itself.")


if __name__ == "__main__":
    # Run benchmark with different configurations
    results = benchmark_translation(
        batch_sizes=[1, 8, 32],
        image_sizes=[64, 128],
        num_warmup=3,
        num_runs=10
    )
    
    print_summary(results)