# PtychoPINN Inference Throughput Benchmarking Plan

**Created**: January 22, 2025  
**Status**: In Progress  
**Objective**: Develop comprehensive benchmarking infrastructure to measure and optimize PtychoPINN inference throughput as a function of batch size

## Executive Summary

This plan outlines the development of a systematic benchmarking tool to measure PtychoPINN inference throughput (images/second) across different batch sizes. The goal is to identify optimal batch configurations for various hardware setups and provide data-driven recommendations for production deployments.

## Background and Motivation

### Current State
- PtychoPINN inference uses fixed batch sizes (typically 32) without systematic optimization
- No existing tools to measure throughput vs batch size relationship
- Batch size control is inconsistent across different scripts
- Memory constraints and optimal settings are unknown

### Expected Benefits
- 2-5x throughput improvement through optimal batch sizing
- Reduced memory consumption for constrained environments
- Data-driven configuration recommendations
- Better resource utilization on different hardware

## Technical Architecture

### System Components

```
┌─────────────────────────────────────────────────────────┐
│                  Benchmark Controller                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Model Loader│  │ Data Manager │  │ Config Parser│  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Inference Engine                        │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │Batch Control│  │  Prediction  │  │  Reassembly  │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Profiling Layer                         │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │   Timing    │  │Memory Monitor│  │GPU Profiler  │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────┐
│                  Analysis & Reporting                    │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Statistics  │  │Visualization │  │   Export     │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Input**: Trained model + test dataset + batch size configurations
2. **Processing**: Systematic inference runs with varying batch sizes
3. **Profiling**: Capture timing, memory, and throughput metrics
4. **Analysis**: Statistical analysis and optimal point detection
5. **Output**: Performance curves, recommendations, and reports

## Phase 1: Core Infrastructure Development

### 1.1 Benchmark Script Creation
**File**: `scripts/benchmark_inference_throughput.py`

#### Core Functions

```python
def benchmark_inference_throughput():
    """Main benchmarking orchestrator."""
    
def load_model_and_data(model_path, data_path, n_images):
    """Load trained model and prepare test data."""
    
def warmup_inference(model, test_data, warmup_runs=3):
    """Prime GPU and JIT compilation."""
    
def benchmark_batch_size(model, test_data, batch_size, num_runs=5):
    """Benchmark single batch size configuration."""
    
def profile_memory_usage(model, test_data, batch_size):
    """Profile memory consumption patterns."""
    
def adaptive_batch_sizing(model, test_data, initial_batch_size):
    """Automatically find maximum viable batch size."""
```

#### Command-Line Interface

```bash
python scripts/benchmark_inference_throughput.py \
    --model-path <path_to_model> \
    --test-data <path_to_test_data.npz> \
    --batch-sizes "1,2,4,8,16,32,64,128,256" \
    --n-images 1000 \
    --num-runs 5 \
    --warmup-runs 3 \
    --output-dir benchmark_results \
    --device gpu:0 \
    --profile-memory \
    --adaptive-sizing \
    --plot-results \
    --export-format json,csv
```

### 1.2 Configuration Management

```python
@dataclass
class BenchmarkConfig:
    """Configuration for throughput benchmarking."""
    model_path: Path
    test_data_path: Path
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8, 16, 32, 64, 128])
    n_images: int = 1000
    num_runs: int = 5
    warmup_runs: int = 3
    device: str = "gpu:0"
    profile_memory: bool = True
    adaptive_sizing: bool = False
    output_dir: Path = Path("benchmark_results")
    export_formats: List[str] = field(default_factory=lambda: ["json", "csv"])
```

## Phase 2: Performance Measurement Implementation

### 2.1 Timing Infrastructure

```python
class TimingProfiler:
    """High-precision timing profiler."""
    
    def __init__(self):
        self.timings = defaultdict(list)
        
    def measure(self, operation_name, func, *args, **kwargs):
        """Measure execution time of a function."""
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        self.timings[operation_name].append(elapsed)
        return result, elapsed
        
    def get_statistics(self, operation_name):
        """Get timing statistics for an operation."""
        times = self.timings[operation_name]
        return {
            'mean': np.mean(times),
            'std': np.std(times),
            'min': np.min(times),
            'max': np.max(times),
            'median': np.median(times)
        }
```

### 2.2 Memory Profiling

```python
class MemoryProfiler:
    """GPU and CPU memory profiler."""
    
    def profile_gpu_memory(self):
        """Get current GPU memory usage."""
        if tf.config.list_physical_devices('GPU'):
            # TensorFlow GPU memory info
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            return {
                'current': memory_info['current'] / 1024**2,  # MB
                'peak': memory_info['peak'] / 1024**2  # MB
            }
        return None
        
    def profile_cpu_memory(self):
        """Get current CPU memory usage."""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss / 1024**2  # MB
```

### 2.3 Throughput Metrics

```python
class ThroughputAnalyzer:
    """Calculate and analyze throughput metrics."""
    
    def calculate_throughput(self, n_images, elapsed_time):
        """Calculate images per second."""
        return n_images / elapsed_time
        
    def calculate_latency(self, elapsed_time, batch_size):
        """Calculate latency per batch."""
        return (elapsed_time / batch_size) * 1000  # ms per image
        
    def calculate_efficiency(self, throughput, memory_usage):
        """Calculate memory efficiency (images/second/MB)."""
        return throughput / memory_usage if memory_usage > 0 else 0
```

## Phase 3: Batch Size Optimization

### 3.1 Adaptive Batch Sizing

```python
def find_optimal_batch_size(model, test_data, min_batch=1, max_batch=256):
    """Find optimal batch size through binary search."""
    
    results = {}
    current_batch = max_batch
    
    while current_batch >= min_batch:
        try:
            # Test current batch size
            throughput = benchmark_single_batch(model, test_data, current_batch)
            results[current_batch] = {
                'status': 'success',
                'throughput': throughput
            }
            
            # Try larger batch size if successful
            if current_batch == max_batch:
                break
            current_batch = min(current_batch * 2, max_batch)
            
        except tf.errors.ResourceExhaustedError:
            # OOM - try smaller batch
            results[current_batch] = {'status': 'OOM'}
            current_batch //= 2
            
    # Find optimal based on throughput
    successful = {k: v for k, v in results.items() if v['status'] == 'success'}
    optimal = max(successful.items(), key=lambda x: x[1]['throughput'])
    
    return optimal[0], results
```

### 3.2 Multi-Stage Batch Control

```python
class BatchController:
    """Control batch sizes at different pipeline stages."""
    
    def __init__(self):
        self.predict_batch_size = 32
        self.reassembly_batch_size = 64
        self.preprocessing_batch_size = 128
        
    def optimize_stage_batches(self, model, test_data):
        """Optimize batch sizes for each stage independently."""
        stages = {
            'predict': self.optimize_predict_batch,
            'reassembly': self.optimize_reassembly_batch,
            'preprocessing': self.optimize_preprocessing_batch
        }
        
        optimal_configs = {}
        for stage_name, optimizer_func in stages.items():
            optimal_configs[stage_name] = optimizer_func(model, test_data)
            
        return optimal_configs
```

### 3.3 Memory-Aware Processing

```python
def memory_constrained_inference(model, test_data, memory_limit_mb):
    """Run inference with memory constraints."""
    
    # Estimate memory per sample
    sample_memory = estimate_sample_memory(test_data[0])
    
    # Calculate maximum batch size
    max_batch = int(memory_limit_mb / sample_memory)
    max_batch = min(max_batch, len(test_data))
    
    # Process in batches
    results = []
    for i in range(0, len(test_data), max_batch):
        batch = test_data[i:i+max_batch]
        result = model.predict(batch, batch_size=max_batch)
        results.append(result)
        
    return np.concatenate(results)
```

## Phase 4: Testing Configurations

### 4.1 Test Matrix

```yaml
test_configurations:
  datasets:
    small:
      n_images: 100
      description: "Quick validation"
    medium:
      n_images: 1000
      description: "Standard benchmark"
    large:
      n_images: 5000
      description: "Stress test"
      
  models:
    standard:
      gridsize: 1
      N: 64
    memory_intensive:
      gridsize: 2
      N: 128
      
  hardware:
    - gpu_single: "Single GPU benchmark"
    - gpu_multi: "Multi-GPU scaling"
    - cpu_baseline: "CPU performance baseline"
    
  batch_sizes:
    comprehensive: [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    quick: [8, 16, 32, 64, 128]
    memory_test: [1, 256, 512, 1024]
```

### 4.2 Benchmark Scenarios

1. **Throughput Optimization**
   - Find batch size with maximum images/second
   - Balance between batch overhead and parallelization

2. **Memory Optimization**
   - Find maximum batch size before OOM
   - Characterize memory scaling behavior

3. **Latency Optimization**
   - Find batch size with minimum per-image latency
   - Important for real-time applications

4. **Energy Efficiency**
   - Measure throughput per watt (if power monitoring available)
   - Identify energy-optimal configurations

## Phase 5: Analysis and Visualization

### 5.1 Performance Metrics

```python
class PerformanceAnalyzer:
    """Analyze and summarize performance metrics."""
    
    def generate_report(self, benchmark_results):
        """Generate comprehensive performance report."""
        return {
            'optimal_batch_size': self.find_optimal_batch_size(benchmark_results),
            'throughput_curve': self.fit_throughput_curve(benchmark_results),
            'memory_scaling': self.analyze_memory_scaling(benchmark_results),
            'bottleneck_analysis': self.identify_bottlenecks(benchmark_results),
            'recommendations': self.generate_recommendations(benchmark_results)
        }
        
    def identify_bottlenecks(self, results):
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        # Check if throughput plateaus
        if self.has_throughput_plateau(results):
            bottlenecks.append('compute_bound')
            
        # Check if memory increases linearly
        if self.has_linear_memory_scaling(results):
            bottlenecks.append('memory_bandwidth')
            
        return bottlenecks
```

### 5.2 Visualization Components

```python
def create_performance_plots(results, output_dir):
    """Create comprehensive performance visualizations."""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Throughput vs Batch Size
    axes[0, 0].plot(batch_sizes, throughputs, 'o-')
    axes[0, 0].set_xlabel('Batch Size')
    axes[0, 0].set_ylabel('Throughput (images/sec)')
    axes[0, 0].set_xscale('log', base=2)
    axes[0, 0].grid(True)
    
    # 2. Memory Usage vs Batch Size
    axes[0, 1].plot(batch_sizes, memory_usage, 'o-', color='red')
    axes[0, 1].set_xlabel('Batch Size')
    axes[0, 1].set_ylabel('Memory Usage (MB)')
    axes[0, 1].set_xscale('log', base=2)
    
    # 3. Latency Distribution
    axes[0, 2].boxplot(latencies)
    axes[0, 2].set_xlabel('Batch Size')
    axes[0, 2].set_ylabel('Latency (ms)')
    
    # 4. Efficiency Curve
    axes[1, 0].plot(batch_sizes, efficiency, 'o-', color='green')
    axes[1, 0].set_xlabel('Batch Size')
    axes[1, 0].set_ylabel('Efficiency (imgs/sec/MB)')
    
    # 5. Scaling Behavior
    axes[1, 1].plot(batch_sizes, scaling_factor, 'o-')
    axes[1, 1].set_xlabel('Batch Size')
    axes[1, 1].set_ylabel('Scaling Factor')
    
    # 6. Optimal Region
    axes[1, 2].fill_between(optimal_range, 0, max_throughput, alpha=0.3)
    axes[1, 2].set_xlabel('Batch Size')
    axes[1, 2].set_ylabel('Throughput')
    axes[1, 2].set_title('Optimal Operating Region')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_analysis.png', dpi=150)
```

### 5.3 Report Generation

```python
def generate_markdown_report(results, output_path):
    """Generate detailed markdown report."""
    
    report = f"""# PtychoPINN Inference Throughput Benchmark Report

## Executive Summary
- **Optimal Batch Size**: {results['optimal_batch_size']}
- **Maximum Throughput**: {results['max_throughput']:.2f} images/second
- **Memory at Optimal**: {results['optimal_memory']:.1f} MB
- **Speedup vs Batch=1**: {results['speedup']:.2f}x

## Configuration
- Model: {results['model_info']}
- Dataset: {results['dataset_info']}
- Hardware: {results['hardware_info']}

## Performance Metrics

| Batch Size | Throughput (img/s) | Memory (MB) | Latency (ms) | Efficiency |
|------------|-------------------|-------------|--------------|------------|
{generate_metrics_table(results)}

## Recommendations
{generate_recommendations(results)}

## Bottleneck Analysis
{analyze_bottlenecks(results)}
"""
    
    with open(output_path, 'w') as f:
        f.write(report)
```

## Phase 6: Integration and Deployment

### 6.1 Integration with Existing Pipeline

1. **Update inference scripts** to use optimal batch sizes
2. **Add batch size to configuration files**
3. **Create presets for different scenarios**:
   - `speed_optimized`: Maximum throughput
   - `memory_optimized`: Minimum memory usage
   - `balanced`: Good trade-off
   - `real_time`: Minimum latency

### 6.2 Continuous Monitoring

```python
class InferenceMonitor:
    """Monitor inference performance in production."""
    
    def __init__(self, log_dir):
        self.log_dir = log_dir
        self.metrics = []
        
    def log_inference(self, batch_size, elapsed_time, memory_usage):
        """Log inference metrics."""
        self.metrics.append({
            'timestamp': datetime.now(),
            'batch_size': batch_size,
            'elapsed_time': elapsed_time,
            'memory_usage': memory_usage,
            'throughput': batch_size / elapsed_time
        })
        
    def detect_anomalies(self):
        """Detect performance anomalies."""
        recent_throughput = [m['throughput'] for m in self.metrics[-100:]]
        baseline_throughput = np.median(recent_throughput)
        
        if recent_throughput[-1] < 0.8 * baseline_throughput:
            self.alert("Performance degradation detected")
```

### 6.3 Auto-Tuning System

```python
class AutoTuner:
    """Automatically tune batch size based on system state."""
    
    def __init__(self, model, initial_batch_size=32):
        self.model = model
        self.current_batch_size = initial_batch_size
        self.performance_history = []
        
    def adapt_batch_size(self, current_memory_pressure):
        """Dynamically adjust batch size."""
        if current_memory_pressure > 0.9:
            # High memory pressure - reduce batch size
            self.current_batch_size = max(1, self.current_batch_size // 2)
        elif current_memory_pressure < 0.5:
            # Low memory pressure - try increasing
            self.current_batch_size = min(256, self.current_batch_size * 2)
            
        return self.current_batch_size
```

## Expected Outcomes

### Performance Improvements
- **2-5x throughput increase** through optimal batch sizing
- **30-50% memory reduction** for memory-constrained scenarios
- **Predictable scaling** behavior documentation

### Deliverables
1. **Benchmark script** (`scripts/benchmark_inference_throughput.py`)
2. **Performance visualization tools**
3. **Optimal configuration recommendations**
4. **Integration guide** for existing workflows
5. **Performance monitoring dashboard** (optional)

### Success Metrics
- ✅ Measure throughput for ≥7 batch sizes
- ✅ Identify optimal batch size within 10% accuracy
- ✅ Detect OOM boundaries automatically
- ✅ Generate actionable recommendations
- ✅ Document hardware-specific characteristics

## Risk Mitigation

### Technical Risks
1. **OOM Errors**: Implement graceful recovery and adaptive sizing
2. **Hardware Variability**: Test on multiple GPU types
3. **Model Compatibility**: Ensure works with different model architectures

### Mitigation Strategies
- Extensive error handling and recovery
- Configurable safety margins
- Fallback to conservative defaults
- Comprehensive logging for debugging

## Timeline

| Phase | Duration | Status |
|-------|----------|--------|
| Phase 1: Core Infrastructure | 1 day | In Progress |
| Phase 2: Measurement Implementation | 1 day | Pending |
| Phase 3: Batch Optimization | 1 day | Pending |
| Phase 4: Testing | 1 day | Pending |
| Phase 5: Analysis & Reporting | 1 day | Pending |
| Phase 6: Integration | 1 day | Pending |

**Total Estimated Duration**: 6 days

## Appendix A: Code Examples

### Basic Usage
```python
from benchmark_inference_throughput import InferenceBenchmark

benchmark = InferenceBenchmark(
    model_path="models/trained_model",
    test_data="data/test.npz"
)

results = benchmark.run(
    batch_sizes=[8, 16, 32, 64, 128],
    num_runs=5
)

optimal_batch_size = results.get_optimal_batch_size()
print(f"Optimal batch size: {optimal_batch_size}")
```

### Advanced Configuration
```python
config = BenchmarkConfig(
    model_path=Path("models/production_model"),
    test_data_path=Path("data/production_test.npz"),
    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256],
    n_images=5000,
    num_runs=10,
    warmup_runs=5,
    device="gpu:0",
    profile_memory=True,
    adaptive_sizing=True
)

benchmark = InferenceBenchmark(config)
results = benchmark.run_comprehensive_analysis()
results.export("results/benchmark_report.json")
```

## Appendix B: Performance Baseline

### Current Performance (Unoptimized)
- Model: PtychoPINN (gridsize=1, N=64)
- Hardware: NVIDIA RTX 3090
- Batch Size: 32 (fixed)
- Throughput: ~400 images/second
- Memory Usage: ~2GB

### Expected Performance (Optimized)
- Optimal Batch Size: 64-128 (estimated)
- Throughput: 800-2000 images/second
- Memory Usage: 2-4GB (controllable)
- Latency: <5ms per image

---

*This plan provides a comprehensive framework for benchmarking and optimizing PtychoPINN inference throughput. Implementation should proceed incrementally with continuous validation at each phase.*