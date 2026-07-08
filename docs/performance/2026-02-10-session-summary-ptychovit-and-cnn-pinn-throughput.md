# Session Summary: CNN-PINN Throughput Benchmarks

**Date:** 2026-02-10  
**Scope:** Benchmark warm-start inference throughput for `ptychopinn-torch` `cnn-pinn` at `64x64` and `128x128`.

## 1) Throughput Benchmarks (Warm Start)

**Hardware:** NVIDIA GeForce RTX 3090 (`cuda`)  
**Model family:** `ptychopinn-torch`  
**Model mode/arch:** `pinn` + `cnn`  
**Protocol:** 3 warmup passes, then 20 timed passes, repeated 5 trials.

### 1.1 64x64
- Mean forward-only throughput: **6058.95 images/s**
- Std dev: **8.36 images/s**
- Single-pass end-to-end helper throughput (forward + reassembly path): **5016.57 images/s**

### 1.2 128x128
- Mean forward-only throughput: **2598.70 images/s**
- Std dev: **5.83 images/s**
- Single-pass end-to-end helper throughput (forward + reassembly path): **2193.60 images/s**

### 1.3 Relative change
- Forward-only throughput dropped by ~**2.33x** from 64 to 128.

## 2) Important Caveats

- No local checkpoint matching `cnn+pinn+N64` or `cnn+pinn+N128` was found in this workspace; throughput runs used architecture-correct **random-init** models.
- 64x64 and 128x128 benchmarks used different NPZ sources, so absolute throughput is valid, but exact cross-size comparability is bounded by dataset differences.
- These are warm-start throughput measurements, not reconstruction-quality benchmarks.

## 3) Artifacts

### Benchmark JSON outputs
- `tmp/benchmarks/ptychopinn_torch_cnn_pinn_n64_warm_start_throughput.json`
- `tmp/benchmarks/ptychopinn_torch_cnn_pinn_n64_warm_start_throughput_repeats.json`
- `tmp/benchmarks/ptychopinn_torch_cnn_pinn_n128_warm_start_throughput.json`
- `tmp/benchmarks/ptychopinn_torch_cnn_pinn_n128_warm_start_throughput_repeats.json`
