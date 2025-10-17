# Grid Studies Implementation Plan for PtychoPINN Model Characterization

## Executive Summary

This document outlines a comprehensive approach for implementing grid studies to characterize PtychoPINN model performance across multiple parameters. The plan leverages existing tools while introducing new capabilities for multi-dimensional parameter sweeps, with a focus on photon budget, training set size, and other key hyperparameters.

## 1. Current State Analysis

### Existing Capabilities
- **Complete Generalization Study**: Robust framework for training size sweeps with statistical analysis
- **Model Comparison Engine**: Comprehensive metrics (PSNR, SSIM, MS-SSIM, FRC) with 3-way comparison
- **Simulation Tools**: Direct control over `nphotons`, `n_images`, `gridsize`
- **Statistical Aggregation**: Multi-trial support with median/percentile analysis
- **Visualization Pipeline**: Automated plot generation for performance curves

### Identified Gaps
1. **Multi-dimensional sweeps**: Current tools focus on 1D parameter sweeps
2. **Parameter coupling**: No native support for exploring parameter interactions
3. **Adaptive sampling**: No intelligent parameter space exploration
4. **Resource optimization**: Limited parallel execution for independent parameter combinations
5. **Interactive visualization**: Static plots only, no interactive exploration tools

## 2. Proposed Grid Study Architecture

### 2.1 Core Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Grid Study Controller                     │
│  - Parameter Grid Definition                                 │
│  - Job Scheduling & Resource Management                      │
│  - Progress Tracking & Recovery                             │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  Parameter Sweep Engine                      │
├─────────────────────┬────────────────┬─────────────────────┤
│  Simulation Module  │  Training Module│  Evaluation Module  │
│  - nphotons sweep   │  - Model configs│  - Metrics compute  │
│  - Dataset gen      │  - Hyperparams  │  - Comparison       │
└─────────────────────┴────────────────┴─────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│              Results Aggregation & Analysis                  │
│  - Statistical summaries                                     │
│  - 2D/3D visualization                                      │
│  - Performance surfaces                                      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Parameter Space Definition

**Primary Parameters (Simulation)**:
- `nphotons`: [1e6, 1e7, 1e8, 1e9, 1e10] - Photon budget
- `n_images`: [256, 512, 1024, 2048, 4096] - Training set size
- `gridsize`: [1, 2] - Overlap mode

**Secondary Parameters (Model)**:
- `learning_rate`: [1e-4, 5e-4, 1e-3, 5e-3]
- `batch_size`: [8, 16, 32]
- `nepochs`: [25, 50, 100]
- `nll_weight`: [0.1, 0.5, 1.0, 2.0]

**Derived Metrics**:
- MS-SSIM (amplitude & phase)
- PSNR (amplitude & phase)
- FRC50 resolution
- Training time
- Inference throughput

## 3. Implementation Strategy

### Phase 1: Enhanced Configuration System

Create a flexible parameter grid configuration format:

```yaml
# grid_study_config.yaml
study_name: "photon_training_characterization"
output_dir: "grid_studies/photon_training_2025"

parameter_grid:
  simulation:
    nphotons: 
      type: "log_range"
      values: [1e6, 1e7, 1e8, 1e9]
    n_images:
      type: "linear"
      values: [512, 1024, 2048, 4096]
      
  training:
    learning_rate:
      type: "fixed"
      value: 0.001
    nepochs:
      type: "adaptive"  # Adjust based on dataset size
      formula: "min(100, 25000 / n_images)"
      
  evaluation:
    test_size: 
      type: "fixed"
      value: 10000
    metrics: ["ms_ssim", "psnr", "frc50", "mae"]
    
execution:
  parallel_jobs: 4
  num_trials: 3
  checkpoint_frequency: 10  # Save progress every 10 combinations
  resume_from_checkpoint: true
```

### Phase 2: Grid Study Orchestrator

**Script**: `scripts/studies/run_grid_study.py`

```python
#!/usr/bin/env python3
"""
Grid Study Orchestrator for Multi-dimensional Parameter Sweeps
"""

import itertools
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import concurrent.futures
from dataclasses import dataclass

@dataclass
class GridPoint:
    """Single point in parameter space"""
    simulation_params: Dict
    training_params: Dict
    point_id: str
    
class GridStudyOrchestrator:
    def __init__(self, config_path: str):
        self.config = self.load_config(config_path)
        self.grid_points = self.generate_grid()
        self.results = {}
        
    def generate_grid(self) -> List[GridPoint]:
        """Generate all parameter combinations"""
        sim_params = self.expand_parameters(self.config['parameter_grid']['simulation'])
        train_params = self.expand_parameters(self.config['parameter_grid']['training'])
        
        grid = []
        for sim_combo in itertools.product(*sim_params.values()):
            for train_combo in itertools.product(*train_params.values()):
                point = GridPoint(
                    simulation_params=dict(zip(sim_params.keys(), sim_combo)),
                    training_params=dict(zip(train_params.keys(), train_combo)),
                    point_id=self.generate_point_id(sim_combo, train_combo)
                )
                grid.append(point)
        return grid
        
    def run_study(self):
        """Execute grid study with parallel processing"""
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.config['execution']['parallel_jobs']) as executor:
            futures = {
                executor.submit(self.run_point, point): point 
                for point in self.grid_points
            }
            
            for future in concurrent.futures.as_completed(futures):
                point = futures[future]
                try:
                    result = future.result()
                    self.results[point.point_id] = result
                    self.save_checkpoint()
                except Exception as e:
                    print(f"Point {point.point_id} failed: {e}")
                    
    def run_point(self, point: GridPoint) -> Dict:
        """Execute single parameter combination"""
        # 1. Generate/load dataset with simulation params
        # 2. Train model with training params
        # 3. Evaluate and collect metrics
        # 4. Return results dictionary
        pass
```

### Phase 3: Multi-dimensional Visualization

**Script**: `scripts/studies/visualize_grid_study.py`

Key visualizations:
1. **2D Heatmaps**: Performance as function of two parameters
2. **3D Surface Plots**: Interactive parameter space exploration
3. **Parallel Coordinates**: Multi-dimensional parameter relationships
4. **Performance Contours**: Iso-performance curves in parameter space

```python
def create_performance_surface(results_df, param1, param2, metric):
    """Create 3D surface plot of metric as function of two parameters"""
    import plotly.graph_objects as go
    
    # Pivot data for surface
    pivot = results_df.pivot_table(
        values=metric, 
        index=param1, 
        columns=param2,
        aggfunc='median'
    )
    
    fig = go.Figure(data=[go.Surface(
        x=pivot.columns,
        y=pivot.index,
        z=pivot.values,
        colorscale='Viridis'
    )])
    
    fig.update_layout(
        title=f'{metric} vs {param1} and {param2}',
        scene=dict(
            xaxis_title=param2,
            yaxis_title=param1,
            zaxis_title=metric
        )
    )
    return fig
```

## 4. Optimal Approach Using Existing Tools

### 4.1 Leveraging Current Infrastructure

The most pragmatic approach builds on the existing `run_complete_generalization_study.sh`:

1. **Extend for 2D sweeps**: Modify to accept multiple parameter arrays
2. **Wrap simulation tools**: Create parameter-controlled dataset generation
3. **Chain existing scripts**: Use bash orchestration for simple grids
4. **Aggregate hierarchically**: Extend `aggregate_and_plot_results.py`

### 4.2 Minimal Implementation Path

```bash
#!/bin/bash
# scripts/studies/run_2d_grid_study.sh

# Parameter arrays
NPHOTONS_VALUES=(1e6 1e7 1e8 1e9)
N_IMAGES_VALUES=(512 1024 2048 4096)
OUTPUT_BASE="grid_studies/nphotons_nimages"

for nphotons in "${NPHOTONS_VALUES[@]}"; do
    for n_images in "${N_IMAGES_VALUES[@]}"; do
        study_dir="${OUTPUT_BASE}/nphotons_${nphotons}_nimages_${n_images}"
        
        # Generate dataset with specific parameters
        python scripts/simulation/simulate_and_save.py \
            --n-photons "$nphotons" \
            --n-images "$n_images" \
            --output-file "${study_dir}/data.npz"
        
        # Run training and evaluation
        ptycho_train \
            --train_data "${study_dir}/data.npz" \
            --n_images "$n_images" \
            --output_dir "${study_dir}/train"
            
        # Compare and collect metrics
        python scripts/compare_models.py \
            --pinn_dir "${study_dir}/train" \
            --test_data "${study_dir}/data.npz" \
            --output_dir "${study_dir}/comparison"
    done
done

# Aggregate all results
python scripts/studies/aggregate_grid_results.py "$OUTPUT_BASE"
```

## 5. Recommended Workflow for Immediate Use

### Step 1: Define Study Parameters

Create a study configuration file:
```yaml
# my_grid_study.yaml
nphotons: [1e7, 1e8, 1e9]
n_images_train: [512, 1024, 2048]
n_images_test: 5000
num_trials: 3
```

### Step 2: Generate Datasets

Use existing simulation tools with parameter variation:
```bash
for nphotons in 1e7 1e8 1e9; do
    python scripts/simulation/simulate_and_save.py \
        --n-photons $nphotons \
        --n-images 10000 \
        --output-file "datasets/study/photons_${nphotons}.npz"
done
```

### Step 3: Run Training Grid

Leverage existing generalization study with modifications:
```bash
for dataset in datasets/study/photons_*.npz; do
    ./scripts/studies/run_complete_generalization_study.sh \
        --train-data "$dataset" \
        --train-sizes "512 1024 2048" \
        --num-trials 3 \
        --output-dir "grid_study/$(basename $dataset .npz)"
done
```

### Step 4: Aggregate and Visualize

Create custom aggregation for 2D/3D visualization:
```python
# Collect all results
results = []
for study_dir in Path("grid_study").glob("photons_*"):
    df = pd.read_csv(study_dir / "results.csv")
    df['nphotons'] = extract_nphotons(study_dir.name)
    results.append(df)

# Create performance matrix
performance_matrix = pd.concat(results)
plot_heatmap(performance_matrix, x='n_images', y='nphotons', z='ms_ssim_phase')
```

## 6. Advanced Features for Future Development

### 6.1 Adaptive Sampling
- Start with coarse grid
- Identify regions of interest (high gradient, optimal performance)
- Refine sampling in promising regions

### 6.2 Bayesian Optimization
- Model performance surface with Gaussian Processes
- Intelligently select next parameter combinations
- Minimize total experiments needed

### 6.3 Resource-Aware Scheduling
- Estimate computation time per point
- Optimize parallel execution schedule
- Support for distributed/cluster execution

### 6.4 Interactive Dashboard
- Real-time monitoring of grid study progress
- Dynamic parameter space visualization
- On-the-fly result analysis

## 7. Example Use Cases

### Use Case 1: Photon Budget vs Training Size Study

**Objective**: Characterize MS-SSIM as function of photon count and dataset size

```bash
# Quick 2x3 grid study
PHOTONS="1e7 1e8 1e9"
SIZES="512 1024 2048"

for p in $PHOTONS; do
    for s in $SIZES; do
        # Simulate dataset
        python scripts/simulation/simulate_and_save.py \
            --n-photons $p --n-images $((s*2)) \
            --output-file temp_dataset.npz
            
        # Split and train
        python scripts/tools/split_dataset_tool.py \
            temp_dataset.npz datasets/grid_${p}_${s}/ \
            --split-fraction 0.5
            
        # Train model
        ptycho_train \
            --train_data datasets/grid_${p}_${s}/train.npz \
            --test_data datasets/grid_${p}_${s}/test.npz \
            --n_images $s \
            --output_dir results/grid_${p}_${s}
    done
done

# Visualize results
python scripts/studies/aggregate_and_plot_results.py results/grid_*
```

### Use Case 2: Hyperparameter Optimization

**Objective**: Find optimal learning rate and batch size combination

```python
# Using existing tools with wrapper script
import subprocess
from itertools import product

learning_rates = [1e-4, 5e-4, 1e-3, 5e-3]
batch_sizes = [8, 16, 32]

results = {}
for lr, bs in product(learning_rates, batch_sizes):
    config = f"""
    learning_rate: {lr}
    batch_size: {bs}
    nepochs: 50
    """
    
    # Save config and run training
    config_path = f"configs/grid_lr{lr}_bs{bs}.yaml"
    with open(config_path, 'w') as f:
        f.write(config)
    
    subprocess.run([
        "ptycho_train",
        "--config", config_path,
        "--train_data", "datasets/standard_train.npz",
        "--output_dir", f"hyperparam_study/lr{lr}_bs{bs}"
    ])
    
    # Collect metrics
    metrics = parse_metrics(f"hyperparam_study/lr{lr}_bs{bs}/metrics.csv")
    results[(lr, bs)] = metrics

# Find optimal combination
best_params = max(results.items(), key=lambda x: x[1]['ms_ssim'])
```

## 8. Implementation Priority

### Immediate (Week 1)
1. Create wrapper scripts for 2D parameter grids using existing tools
2. Extend `aggregate_and_plot_results.py` for multi-dimensional data
3. Document standard grid study workflows

### Short-term (Week 2-3)
1. Implement `GridStudyOrchestrator` class
2. Add parallel execution support
3. Create interactive visualization tools

### Long-term (Month 2+)
1. Adaptive sampling strategies
2. Bayesian optimization integration
3. Distributed execution support
4. Web-based monitoring dashboard

## 9. Success Metrics

- **Coverage**: Ability to sweep 3+ parameters simultaneously
- **Efficiency**: 4x speedup through parallel execution
- **Robustness**: Automatic checkpoint/resume for long studies
- **Usability**: Single command to launch complex studies
- **Insights**: Clear visualization of parameter interactions

## 10. Conclusion

This plan provides both immediate practical solutions using existing tools and a roadmap for sophisticated grid study capabilities. The phased approach ensures quick wins while building toward comprehensive model characterization infrastructure.