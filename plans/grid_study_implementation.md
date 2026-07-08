# Grid Study Implementation Plan for PtychoPINN

## Executive Summary

This plan outlines the implementation of a comprehensive parameter grid study framework for PtychoPINN model characterization. The framework will enable systematic exploration of parameter spaces (e.g., photon dose, training size, regularization weights) with automatic metric calculation, visualization, and publication-quality output generation.

## 1. Objectives

### Primary Goals
- Create a flexible framework for 2D/3D parameter grid studies
- Integrate seamlessly with existing PtychoPINN infrastructure
- Generate publication-quality heatmaps and surface plots of model performance
- Support parallel execution for efficient parameter space exploration
- Enable reproducible and resumable studies

### Key Metrics to Track
- MS-SSIM (Multi-Scale Structural Similarity Index)
- PSNR (Peak Signal-to-Noise Ratio)
- FRC50 (Fourier Ring Correlation at 0.5 threshold)
- MAE/MSE (Mean Absolute/Squared Error)
- Training/inference time
- Memory usage

## 2. Architecture Design

### 2.1 Component Overview

```
┌─────────────────────────────────────────────────────────────┐
│                   Grid Study Orchestrator                    │
├─────────────────────────────────────────────────────────────┤
│  • Parameter Grid Generation                                 │
│  • Experiment Management                                     │
│  • Parallel Execution Engine                                 │
│  • Result Aggregation Pipeline                              │
│  • Visualization Generator                                   │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
    ┌──────────┐        ┌──────────┐        ┌──────────┐
    │ Training │        │ Inference│        │ Metrics  │
    │ Pipeline │        │ Pipeline │        │ Pipeline │
    └──────────┘        └──────────┘        └──────────┘
```

### 2.2 Data Flow

1. **Configuration Generation** → Parameter combinations
2. **Data Simulation** → Synthetic datasets with varied parameters
3. **Model Training** → Parallel execution of training jobs
4. **Inference & Evaluation** → Metric calculation on test sets
5. **Result Aggregation** → Statistical analysis across trials
6. **Visualization** → Heatmaps, surface plots, sensitivity analysis

## 3. Implementation Phases

### Phase 1: Core Infrastructure (Week 1)

#### 3.1.1 Parameter Grid Generator (`scripts/studies/parameter_grid.py`)

```python
class ParameterGrid:
    """
    Generates Cartesian product of parameter combinations.
    
    Features:
    - Nested parameter support (e.g., model.nphotons)
    - Logarithmic/linear spacing options
    - Constraint validation
    - Configuration serialization
    """
    
    def __init__(self, base_config: TrainingConfig):
        self.base_config = base_config
        self.dimensions = {}
        
    def add_dimension(self, name: str, values: List, 
                     scale: str = 'linear'):
        """Add parameter dimension to study."""
        
    def generate_configs(self) -> List[Tuple[str, TrainingConfig]]:
        """Generate all valid configurations."""
```

#### 3.1.2 Parallel Executor (`scripts/studies/parallel_executor.py`)

```python
class ParallelExecutor:
    """
    GPU-aware parallel execution engine.
    
    Features:
    - Dynamic GPU allocation
    - Memory-aware scheduling
    - Failure recovery
    - Progress tracking
    """
    
    def __init__(self, max_workers: int = None, 
                 gpu_per_job: float = 0.5):
        self.max_workers = max_workers
        self.gpu_scheduler = GPUScheduler(gpu_per_job)
        
    def run_batch(self, jobs: List[Job]) -> List[Result]:
        """Execute jobs with resource management."""
```

#### 3.1.3 Data Management (`scripts/studies/data_manager.py`)

```python
class GridStudyDataManager:
    """
    Handles data generation and caching.
    
    Features:
    - Synthetic data generation with varying nphotons
    - Intelligent caching to avoid regeneration
    - Validation dataset consistency
    """
    
    def get_dataset(self, config: TrainingConfig) -> Path:
        """Get or generate dataset for configuration."""
        cache_key = self._compute_cache_key(config)
        if self._is_cached(cache_key):
            return self._get_cached_path(cache_key)
        return self._generate_dataset(config)
```

### Phase 2: Orchestration Layer (Week 1-2)

#### 3.2.1 Main Orchestrator (`scripts/studies/grid_study_orchestrator.py`)

```python
class GridStudyOrchestrator:
    """
    Complete grid study workflow management.
    
    Responsibilities:
    - Experiment setup and tracking
    - Checkpoint management
    - Result collection
    - Report generation
    """
    
    def __init__(self, study_name: str, output_dir: Path):
        self.study_name = study_name
        self.output_dir = output_dir
        self.checkpoint_manager = CheckpointManager(output_dir)
        
    def run_study(self, resume: bool = False) -> StudyResults:
        """Execute complete grid study with checkpointing."""
```

#### 3.2.2 Checkpoint System (`scripts/studies/checkpoint_manager.py`)

```python
class CheckpointManager:
    """
    Enable resumable studies.
    
    Features:
    - Periodic state saving
    - Experiment completion tracking
    - Partial result recovery
    """
    
    def save_checkpoint(self, state: StudyState):
        """Save current study state."""
        
    def load_checkpoint(self) -> Optional[StudyState]:
        """Restore previous study state."""
```

### Phase 3: Analysis & Visualization (Week 2)

#### 3.3.1 Result Aggregator (`scripts/studies/result_aggregator.py`)

```python
class ResultAggregator:
    """
    Statistical analysis of grid study results.
    
    Features:
    - Multi-trial statistics (mean, std, percentiles)
    - Outlier detection
    - Sensitivity analysis
    - Optimal parameter identification
    """
    
    def aggregate_metrics(self, results: List[ExperimentResult]) -> pd.DataFrame:
        """Compute statistics across parameter grid."""
```

#### 3.3.2 Visualization Generator (`scripts/studies/visualization.py`)

```python
class GridStudyVisualizer:
    """
    Generate publication-quality visualizations.
    
    Output types:
    - 2D heatmaps (matplotlib/seaborn)
    - 3D surface plots (plotly)
    - Parameter sensitivity plots
    - Convergence analysis
    """
    
    def create_heatmap(self, data: pd.DataFrame, 
                       x_param: str, y_param: str, 
                       metric: str = 'ms_ssim'):
        """Generate 2D parameter heatmap."""
        
    def create_surface_plot(self, data: pd.DataFrame,
                           params: List[str], metric: str):
        """Generate interactive 3D surface plot."""
```

### Phase 4: Integration & CLI (Week 2-3)

#### 3.4.1 Unified CLI (`scripts/studies/run_grid_study.py`)

```python
#!/usr/bin/env python3
"""
Run parameter grid study for PtychoPINN.

Usage:
    python run_grid_study.py \
        --param nphotons "1e6,1e7,1e8,1e9,1e10" \
        --param n_images "512,1024,2048,4096" \
        --param nll_weight "0.1,1.0,10.0" \
        --base-config configs/grid_study_base.yaml \
        --output-dir grid_studies/photon_dose_study \
        --model-types pinn,baseline \
        --num-trials 3 \
        --max-workers 4
"""
```

#### 3.4.2 Configuration Templates (`configs/grid_study/`)

```yaml
# configs/grid_study/dose_response_template.yaml
base_configuration:
  model:
    N: 64
    gridsize: 1
    model_type: pinn
  training:
    nepochs: 100
    batch_size: 32
    
parameter_sweeps:
  nphotons:
    values: [1e6, 1e7, 1e8, 1e9, 1e10]
    scale: log
  n_images:
    values: [256, 512, 1024, 2048, 4096]
    scale: log2
    
study_settings:
  num_trials: 5
  metrics: [ms_ssim, psnr, frc50]
  comparison_models: [pinn, baseline]
```

## 4. Integration with Existing Infrastructure

### 4.1 Leveraging Current Tools

| Component | Existing Tool | Integration Method |
|-----------|--------------|-------------------|
| Training | `ptycho_train` | Subprocess with config injection |
| Inference | `ptycho_inference` | Direct API call |
| Comparison | `compare_models.py` | Import and extend |
| Metrics | `ptycho.evaluation` | Direct function calls |
| Visualization | `aggregate_and_plot_results.py` | Extend with grid plots |
| Simulation | `scripts/simulation/` | Direct API for data generation |

### 4.2 Configuration Management

```python
# Extend existing TrainingConfig
@dataclass
class GridStudyConfig(TrainingConfig):
    """Extended configuration for grid studies."""
    study_name: str
    parameter_grid: Dict[str, List]
    num_trials: int = 5
    comparison_models: List[str] = field(default_factory=lambda: ['pinn'])
```

### 4.3 Logging Integration

```python
# Use centralized logging
from ptycho.log_config import setup_logging

def setup_study_logging(output_dir: Path):
    """Configure logging for grid study."""
    study_log_dir = output_dir / "logs"
    setup_logging(study_log_dir, console_level='INFO')
```

## 5. Usage Examples

### 5.1 2D Photon Dose vs Training Size Study

```bash
python scripts/studies/run_grid_study.py \
    --study-name "dose_vs_size" \
    --param nphotons "1e6,1e7,1e8,1e9,1e10" \
    --param n_images "256,512,1024,2048,4096" \
    --base-config configs/fly64_config.yaml \
    --output-dir studies/dose_vs_size_$(date +%Y%m%d) \
    --metrics ms_ssim,psnr,frc50 \
    --num-trials 5 \
    --max-workers 4 \
    --generate-heatmap
```

### 5.2 3D Parameter Study

```bash
python scripts/studies/run_grid_study.py \
    --study-name "3d_parameter_exploration" \
    --param nphotons "1e7,1e8,1e9" \
    --param n_images "512,1024,2048" \
    --param nll_weight "0.01,0.1,1.0,10.0" \
    --output-dir studies/3d_exploration \
    --visualization-type surface \
    --interactive-plots
```

### 5.3 Model Architecture Study

```bash
python scripts/studies/run_grid_study.py \
    --study-name "architecture_study" \
    --param n_filters_scale "0.5,1.0,2.0" \
    --param batch_size "16,32,64" \
    --param learning_rate "1e-4,1e-3,1e-2" \
    --fixed nphotons=1e9 \
    --fixed n_images=1024 \
    --output-dir studies/architecture \
    --focus-metric convergence_speed
```

## 6. Output Structure

### 6.1 Directory Organization

```
grid_study_output/
├── experiments/
│   ├── exp_001_nphotons_1e6_nimages_512/
│   │   ├── trial_1/
│   │   │   ├── pinn_run/
│   │   │   │   ├── wts.h5.zip
│   │   │   │   ├── history.dill
│   │   │   │   └── logs/
│   │   │   ├── baseline_run/
│   │   │   └── metrics.json
│   │   ├── trial_2/
│   │   └── trial_3/
│   └── exp_002_nphotons_1e6_nimages_1024/
├── visualizations/
│   ├── heatmap_nphotons_vs_nimages_ms_ssim.png
│   ├── heatmap_nphotons_vs_nimages_psnr.png
│   ├── surface_plot_3d_ms_ssim.html
│   ├── parameter_sensitivity_analysis.png
│   └── optimal_parameters_pareto.png
├── data/
│   ├── aggregated_results.csv
│   ├── raw_results_all_trials.csv
│   └── statistical_summary.csv
├── reports/
│   ├── executive_summary.md
│   ├── detailed_analysis.html
│   └── latex_figures/
└── study_config.yaml
```

### 6.2 Result Files

#### `aggregated_results.csv`
```csv
nphotons,n_images,nll_weight,ms_ssim_mean,ms_ssim_std,psnr_mean,psnr_std,frc50_mean
1e6,512,1.0,0.745,0.023,28.3,1.2,0.42
1e7,512,1.0,0.812,0.018,31.5,0.9,0.51
```

#### `study_config.yaml`
```yaml
study_metadata:
  name: dose_vs_size_study
  date: 2025-01-25
  ptychopinn_version: 1.0.0
  total_experiments: 25
  total_runtime_hours: 18.5
  
parameter_space:
  nphotons: [1e6, 1e7, 1e8, 1e9, 1e10]
  n_images: [256, 512, 1024, 2048, 4096]
  
best_parameters:
  ms_ssim_optimal:
    nphotons: 1e9
    n_images: 2048
    score: 0.923
```

## 7. Advanced Features

### 7.1 Adaptive Sampling

```python
class AdaptiveSampler:
    """
    Intelligently sample parameter space.
    
    Methods:
    - Bayesian optimization for efficient exploration
    - Gradient-based refinement near optima
    - Uncertainty-guided sampling
    """
    
    def suggest_next_point(self, current_results: pd.DataFrame) -> Dict:
        """Use Gaussian Process to identify high-value regions."""
        gp = GaussianProcessRegressor()
        gp.fit(current_results[params], current_results[metric])
        return self._maximize_acquisition(gp)
```

### 7.2 Real-time Monitoring

```python
class StudyMonitor:
    """
    Web-based monitoring dashboard.
    
    Features:
    - Live progress tracking
    - Partial result visualization
    - Resource utilization graphs
    - Estimated completion time
    """
    
    def start_dashboard(self, port: int = 8080):
        """Launch monitoring web interface."""
```

### 7.3 Distributed Execution

```python
class DistributedExecutor(ParallelExecutor):
    """
    Multi-node execution support.
    
    Features:
    - SLURM integration
    - AWS Batch support
    - Automatic job distribution
    - Fault tolerance
    """
```

## 8. Testing Strategy

### 8.1 Unit Tests

```python
# tests/studies/test_parameter_grid.py
class TestParameterGrid(unittest.TestCase):
    def test_cartesian_product(self):
        """Verify correct parameter combinations."""
        
    def test_constraint_validation(self):
        """Test invalid parameter rejection."""
        
    def test_config_generation(self):
        """Validate TrainingConfig creation."""
```

### 8.2 Integration Tests

```python
# tests/studies/test_grid_study_integration.py
class TestGridStudyIntegration(unittest.TestCase):
    def test_end_to_end_small_grid(self):
        """Run minimal grid study (2x2)."""
        
    def test_checkpoint_resume(self):
        """Verify study resumption."""
        
    def test_parallel_execution(self):
        """Test multi-worker execution."""
```

### 8.3 Performance Tests

```python
# tests/studies/test_performance.py
class TestGridStudyPerformance(unittest.TestCase):
    def test_large_grid_memory(self):
        """Monitor memory usage for 10x10 grid."""
        
    def test_execution_time_scaling(self):
        """Verify linear scaling with workers."""
```

## 9. Documentation

### 9.1 User Guide (`docs/GRID_STUDY_USER_GUIDE.md`)

- Quick start tutorial
- Parameter selection best practices
- Computational resource estimation
- Troubleshooting common issues

### 9.2 Developer Guide (`docs/GRID_STUDY_DEVELOPER_GUIDE.md`)

- Architecture overview
- Extension points
- Adding new metrics
- Custom visualizations

### 9.3 API Reference

- Complete docstrings for all classes
- Usage examples for each component
- Configuration schema documentation

## 10. Success Criteria

### Functional Requirements
- ✅ Support 2D and 3D parameter grids
- ✅ Generate publication-quality visualizations
- ✅ Handle 100+ experiment combinations
- ✅ Provide checkpoint/resume capability
- ✅ Integrate with existing PtychoPINN tools

### Performance Requirements
- ✅ < 5 minute setup for new study
- ✅ Linear scaling with worker count
- ✅ < 10% overhead vs manual execution
- ✅ Memory usage < 2GB per worker

### Quality Requirements
- ✅ 90% test coverage
- ✅ Comprehensive documentation
- ✅ Reproducible results
- ✅ Clean error handling

## 11. Timeline

| Week | Phase | Deliverables |
|------|-------|-------------|
| 1 | Core Infrastructure | Parameter grid, executor, data manager |
| 1-2 | Orchestration | Main orchestrator, checkpointing |
| 2 | Analysis & Visualization | Aggregator, visualizer |
| 2-3 | Integration & CLI | CLI interface, config templates |
| 3 | Testing & Documentation | Test suite, user guides |
| 3-4 | Validation & Refinement | Example studies, performance tuning |

## 12. Risk Mitigation

| Risk | Impact | Mitigation |
|------|--------|------------|
| GPU memory issues | High | Implement adaptive batch sizing |
| Long execution times | Medium | Add distributed execution support |
| Data storage growth | Medium | Implement selective result retention |
| Complex dependencies | Low | Use subprocess isolation |

## 13. Future Enhancements

- **Multi-objective optimization**: Pareto front exploration
- **Transfer learning**: Reuse models across parameter values
- **Active learning**: Dynamic parameter selection
- **Cloud integration**: AWS/GCP batch processing
- **Interactive notebook**: Jupyter-based study design

## Appendix A: Example Configuration Files

### A.1 Minimal Grid Study
```yaml
# configs/grid_study/minimal.yaml
parameters:
  nphotons: [1e7, 1e8, 1e9]
  n_images: [512, 1024]
metrics: [ms_ssim]
output_dir: studies/minimal_test
```

### A.2 Comprehensive Study
```yaml
# configs/grid_study/comprehensive.yaml
parameters:
  nphotons: 
    values: [1e6, 1e7, 1e8, 1e9, 1e10]
    scale: log
  n_images:
    values: [256, 512, 1024, 2048, 4096]
    scale: log2
  nll_weight:
    values: [0.01, 0.1, 1.0, 10.0]
    scale: log
    
models:
  - type: pinn
    config_overrides:
      probe_trainable: true
  - type: baseline
    
evaluation:
  metrics: [ms_ssim, psnr, frc50, mae]
  num_trials: 5
  test_set_size: 1000
  
execution:
  max_workers: 8
  gpu_per_job: 0.5
  checkpoint_interval: 10
  
visualization:
  types: [heatmap, surface, sensitivity]
  save_format: [png, html, pdf]
```

## Appendix B: Command Reference

```bash
# List available parameters
python run_grid_study.py --list-params

# Dry run (show what would be executed)
python run_grid_study.py --config study.yaml --dry-run

# Resume interrupted study
python run_grid_study.py --resume studies/interrupted_study/

# Generate report only (from existing results)
python run_grid_study.py --report-only studies/completed_study/

# Interactive parameter selection
python run_grid_study.py --interactive
```

---

*This implementation plan provides a robust, extensible framework for comprehensive parameter grid studies in PtychoPINN, maximizing reuse of existing infrastructure while adding powerful new capabilities for model characterization and optimization.*