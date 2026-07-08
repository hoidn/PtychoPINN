#!/usr/bin/env python
"""
Simple demo of the benchmarking infrastructure without full model inference.
This demonstrates the profiling capabilities of the tool.
"""

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, '.')

from scripts.benchmark_inference_throughput import (
    BenchmarkConfig, TimingProfiler, MemoryProfiler, 
    ThroughputAnalyzer
)

def simulate_inference(batch_size: int, n_images: int = 100):
    """Simulate inference with artificial delays based on batch size."""
    # Simulate processing time - larger batches are more efficient
    base_time = 0.001  # 1ms per image
    batch_overhead = 0.01  # 10ms per batch
    
    n_batches = (n_images + batch_size - 1) // batch_size
    total_time = n_images * base_time + n_batches * batch_overhead
    
    # Add some variance
    total_time *= np.random.uniform(0.9, 1.1)
    
    time.sleep(total_time)
    return np.random.randn(n_images, 64, 64)  # Mock output

def main():
    print("=" * 60)
    print("PtychoPINN Inference Throughput Benchmarking Demo")
    print("=" * 60)
    
    # Initialize profilers
    timer = TimingProfiler()
    memory_profiler = MemoryProfiler()
    analyzer = ThroughputAnalyzer()
    
    # Configuration
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128]
    n_images = 1000
    num_runs = 3
    
    print(f"\nConfiguration:")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Number of images: {n_images}")
    print(f"  Runs per batch: {num_runs}")
    
    # Benchmark each batch size
    results = {}
    
    print("\n" + "=" * 60)
    print("Running Benchmarks")
    print("=" * 60)
    
    for batch_size in batch_sizes:
        print(f"\nBatch size {batch_size}:")
        
        # Take initial memory snapshot
        memory_profiler.snapshot(f"before_batch_{batch_size}")
        
        throughputs = []
        for run in range(num_runs):
            # Measure inference time
            output, elapsed = timer.measure(
                f"batch_{batch_size}_run_{run}",
                simulate_inference,
                batch_size,
                n_images
            )
            
            throughput = analyzer.calculate_throughput(n_images, elapsed)
            throughputs.append(throughput)
            print(f"  Run {run+1}: {throughput:6.1f} images/sec ({elapsed:.3f}s)")
        
        # Take final memory snapshot
        final_snapshot = memory_profiler.snapshot(f"after_batch_{batch_size}")
        
        # Store results
        results[batch_size] = {
            'status': 'success' if batch_size <= 64 else 'OOM',  # Simulate OOM for large batches
            'throughput': np.mean(throughputs),
            'throughput_std': np.std(throughputs),
            'memory_mb': final_snapshot['cpu_mb']
        }
    
    # Find optimal batch size
    optimal_batch, optimal_result = analyzer.find_optimal_batch_size(results)
    
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    print("\n| Batch Size | Throughput (img/s) | Std Dev | Memory (MB) | Status |")
    print("|------------|-------------------|---------|-------------|--------|")
    
    for batch_size in batch_sizes:
        result = results[batch_size]
        if result['status'] == 'success':
            print(f"| {batch_size:10} | {result['throughput']:17.1f} | {result['throughput_std']:7.2f} | "
                  f"{result['memory_mb']:11.1f} | ✅ {result['status']:6} |")
        else:
            print(f"| {batch_size:10} | {'--':>17} | {'--':>7} | {'--':>11} | ❌ {result['status']:6} |")
    
    print("\n" + "=" * 60)
    print("Optimization Results")
    print("=" * 60)
    
    print(f"\n✨ Optimal Batch Size: {optimal_batch}")
    print(f"   Maximum Throughput: {optimal_result.get('throughput', 0):.1f} images/sec")
    
    # Calculate speedup
    baseline = results.get(1, {}).get('throughput', 1)
    speedup = optimal_result.get('throughput', 0) / baseline if baseline > 0 else 1
    print(f"   Speedup vs Batch=1: {speedup:.2f}x")
    
    # Efficiency analysis
    print("\n📊 Efficiency Analysis:")
    for batch_size in [1, 8, 16, 32, optimal_batch]:
        if batch_size in results and results[batch_size]['status'] == 'success':
            efficiency = analyzer.calculate_efficiency(
                results[batch_size]['throughput'],
                results[batch_size]['memory_mb']
            )
            print(f"   Batch {batch_size:3}: {efficiency:.3f} images/sec/MB")
    
    print("\n" + "=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print("\nThis demonstrates the core profiling capabilities of the")
    print("inference throughput benchmarking tool. In production, this")
    print("would run actual model inference instead of simulated delays.")
    print("\nFor full benchmarking, use:")
    print("  python scripts/benchmark_inference_throughput.py --help")

if __name__ == '__main__':
    main()