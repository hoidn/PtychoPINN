#!/usr/bin/env python3
"""Benchmark translation performance: Original vs XLA implementation."""

import os
import time
import numpy as np
import tensorflow as tf

# Test both implementations
os.environ['USE_XLA_TRANSLATE'] = '0'

from ptycho.tf_helper import translate_core
from ptycho.projective_warp_xla import translate_xla

def benchmark_implementation(name, func, images, translations, num_iterations=100, warmup=10):
    """Benchmark a translation implementation."""
    
    # Warmup
    for _ in range(warmup):
        _ = func(images, translations)
    
    # Force GPU sync
    if tf.config.list_physical_devices('GPU'):
        tf.test.gpu_device_name()  # Forces sync
    
    # Benchmark
    start_time = time.time()
    for _ in range(num_iterations):
        result = func(images, translations)
    
    # Force GPU sync again
    if tf.config.list_physical_devices('GPU'):
        _ = result.numpy()  # Forces computation to complete
    
    elapsed_time = time.time() - start_time
    
    # Calculate metrics
    total_images = images.shape[0] * num_iterations
    images_per_sec = total_images / elapsed_time
    ms_per_batch = (elapsed_time / num_iterations) * 1000
    
    return {
        'name': name,
        'total_time': elapsed_time,
        'images_per_sec': images_per_sec,
        'ms_per_batch': ms_per_batch,
        'iterations': num_iterations
    }

def run_benchmarks():
    """Run comprehensive benchmarks."""
    print("Translation Performance Benchmark")
    print("=" * 80)
    print(f"TensorFlow version: {tf.__version__}")
    print(f"GPU available: {tf.config.list_physical_devices('GPU')}")
    print()
    
    # Test configurations
    configs = [
        {'batch': 1, 'size': 64, 'channels': 1, 'name': 'Small (1×64×64×1)'},
        {'batch': 16, 'size': 64, 'channels': 1, 'name': 'Training batch (16×64×64×1)'},
        {'batch': 32, 'size': 64, 'channels': 1, 'name': 'Large batch (32×64×64×1)'},
        {'batch': 8, 'size': 128, 'channels': 1, 'name': 'Higher res (8×128×128×1)'},
        {'batch': 4, 'size': 256, 'channels': 1, 'name': 'High res (4×256×256×1)'},
        {'batch': 16, 'size': 64, 'channels': 2, 'name': 'Multi-channel (16×64×64×2)'},
    ]
    
    # Test with different translation patterns
    translation_types = [
        {'name': 'Integer', 'gen': lambda b: tf.random.uniform((b, 2), -10, 10, dtype=tf.int32)},
        {'name': 'Fractional', 'gen': lambda b: tf.random.uniform((b, 2), -10, 10, dtype=tf.float32)},
    ]
    
    results = []
    
    for config in configs:
        batch = config['batch']
        size = config['size']
        channels = config['channels']
        
        # Create test data
        images = tf.random.uniform((batch, size, size, channels), dtype=tf.float32)
        
        print(f"\nTesting {config['name']}:")
        print("-" * 60)
        
        for trans_type in translation_types:
            translations = tf.cast(trans_type['gen'](batch), tf.float32)
            
            print(f"\n{trans_type['name']} translations:")
            
            # Benchmark original implementation
            orig_result = benchmark_implementation(
                'Original (ImageProjectiveTransformV3)',
                lambda img, trans: translate_core(img, trans, interpolation='bilinear', use_xla_workaround=False),
                images, translations
            )
            
            # Benchmark XLA implementation (no JIT)
            xla_result = benchmark_implementation(
                'XLA (no JIT)',
                lambda img, trans: translate_xla(img, trans, interpolation='bilinear', use_jit=False),
                images, translations
            )
            
            # Benchmark XLA implementation (with JIT)
            # Create JIT function
            @tf.function(jit_compile=True)
            def translate_xla_jit(img, trans):
                return translate_xla(img, trans, interpolation='bilinear', use_jit=True)
            
            # First call to compile
            _ = translate_xla_jit(images, translations)
            
            xla_jit_result = benchmark_implementation(
                'XLA (with JIT)',
                translate_xla_jit,
                images, translations
            )
            
            # Print results
            for result in [orig_result, xla_result, xla_jit_result]:
                print(f"  {result['name']}:")
                print(f"    Images/sec: {result['images_per_sec']:,.0f}")
                print(f"    ms/batch: {result['ms_per_batch']:.2f}")
            
            # Calculate speedups
            xla_speedup = xla_result['images_per_sec'] / orig_result['images_per_sec']
            jit_speedup = xla_jit_result['images_per_sec'] / orig_result['images_per_sec']
            
            print(f"\n  Speedup vs Original:")
            print(f"    XLA (no JIT): {xla_speedup:.2f}x")
            print(f"    XLA (with JIT): {jit_speedup:.2f}x")
            
            results.append({
                'config': config['name'],
                'translation': trans_type['name'],
                'original_imgs_sec': orig_result['images_per_sec'],
                'xla_imgs_sec': xla_result['images_per_sec'],
                'xla_jit_imgs_sec': xla_jit_result['images_per_sec'],
                'xla_speedup': xla_speedup,
                'jit_speedup': jit_speedup
            })
    
    # Summary table
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'Configuration':<30} {'Translation':<12} {'Original':<12} {'XLA':<12} {'XLA+JIT':<12} {'Speedup':<10}")
    print(f"{'':30} {'':12} {'(imgs/sec)':<12} {'(imgs/sec)':<12} {'(imgs/sec)':<12} {'(JIT)':<10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['config']:<30} {r['translation']:<12} "
              f"{r['original_imgs_sec']:<12,.0f} {r['xla_imgs_sec']:<12,.0f} "
              f"{r['xla_jit_imgs_sec']:<12,.0f} {r['jit_speedup']:<10.2f}x")
    
    # Calculate average speedups
    avg_xla_speedup = sum(r['xla_speedup'] for r in results) / len(results)
    avg_jit_speedup = sum(r['jit_speedup'] for r in results) / len(results)
    
    print("\n" + "=" * 80)
    print(f"Average Speedups:")
    print(f"  XLA (no JIT): {avg_xla_speedup:.2f}x")
    print(f"  XLA (with JIT): {avg_jit_speedup:.2f}x")

def test_complex_performance():
    """Test performance with complex numbers."""
    print("\n\n" + "=" * 80)
    print("COMPLEX NUMBER PERFORMANCE")
    print("=" * 80)
    
    batch = 16
    size = 64
    channels = 1
    
    # Create complex test data
    real_part = tf.random.uniform((batch, size, size, channels), dtype=tf.float32)
    imag_part = tf.random.uniform((batch, size, size, channels), dtype=tf.float32)
    complex_images = tf.complex(real_part, imag_part)
    translations = tf.random.uniform((batch, 2), -10, 10, dtype=tf.float32)
    
    print(f"Testing complex images ({batch}×{size}×{size}×{channels}):")
    
    # Import the complexified translate function
    from ptycho.tf_helper import translate
    
    # Benchmark original
    orig_result = benchmark_implementation(
        'Original (complex)',
        lambda img, trans: translate(img, trans, use_xla_workaround=False),
        complex_images, translations,
        num_iterations=50
    )
    
    # Benchmark XLA
    xla_result = benchmark_implementation(
        'XLA (complex)',
        lambda img, trans: translate_xla(img, trans, use_jit=False),
        complex_images, translations,
        num_iterations=50
    )
    
    print(f"\nOriginal: {orig_result['images_per_sec']:,.0f} images/sec")
    print(f"XLA: {xla_result['images_per_sec']:,.0f} images/sec")
    print(f"Speedup: {xla_result['images_per_sec'] / orig_result['images_per_sec']:.2f}x")

if __name__ == "__main__":
    run_benchmarks()
    test_complex_performance()