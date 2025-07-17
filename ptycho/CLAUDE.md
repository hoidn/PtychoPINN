# PtychoPINN Core Library Agent Guide

## Quick Context
- **Purpose**: Core TensorFlow implementation of physics-informed neural networks
- **Architecture**: Two-system design (legacy grid-based vs. modern coordinate-based)
- **Critical Rule**: Never modify core physics (model.py, diffsim.py, tf_helper.py) unless explicitly asked
- **Philosophy**: Stable physics implementation with flexible data handling

## Module Organization

### Configuration System
- **Modern**: `config/` (dataclass system - `ModelConfig`, `TrainingConfig`)
- **Legacy**: `params.py` (dictionary-based, maintained for compatibility)
- **Source of truth**: <code-ref type="config">ptycho/config/config.py</code-ref>
- **Flow**: Modern config → Updates legacy dict → Global access

### Data Pipeline
- **Input**: `loader.py`, `raw_data.py` (RawData → PtychoDataContainer)
- **Contracts**: Follow data_contracts.md specifications
- **Flow**: Raw NPZ → RawData → PtychoDataContainer → Model

### Core Model Architecture
- **Model**: `model.py` (U-Net + physics layers) - **DO NOT MODIFY**
- **Physics**: `diffsim.py`, `nongrid_simulation.py` - **STABLE IMPLEMENTATION**
- **TensorFlow helpers**: `tf_helper.py` - **CORE FUNCTIONALITY**

### Image Processing
- **Modern location**: `image/` subdirectory
- **Key modules**: 
  - `stitching.py`: Grid-based patch reassembly
  - `cropping.py`: Contains `<code-ref type="function">align_for_evaluation</code-ref>`
  - `registration.py`: Automatic image alignment

### Workflows
- **High-level**: `workflows/` (orchestration functions)
- **Usage**: Called by scripts/, not directly by users
- **Purpose**: Bridge between low-level modules and user scripts

## Critical Anti-Patterns to Avoid

### Never Perform Complex Operations on Import
**Wrong**:
```python
# DON'T DO THIS - triggers data loading on import
from ptycho.generate_data import YY_ground_truth  # Side effect!
```

**Correct**:
```python
# DO THIS - explicit function calls
from ptycho.generate_data import load_ground_truth
ground_truth = load_ground_truth(dataset_path)
```

### Never Rely on Global State in New Code
**Wrong**:
```python
# DON'T DO THIS - hidden dependency
from ptycho.params import params
def my_function():
    value = params.get('hidden_setting')  # Fragile
```

**Correct**:
```python
# DO THIS - explicit arguments
def my_function(config: ModelConfig):
    value = config.hidden_setting  # Clear dependency
```

### Never Modify Core Physics Without Explicit Request
**Protected modules**:
- `model.py` - Neural network architecture
- `diffsim.py` - Forward physics simulation
- `tf_helper.py` - Core TensorFlow operations

## Safe Initialization Pattern

### Correct Startup Sequence
```python
# 1. Set up modern configuration
config = TrainingConfig(
    model=ModelConfig(N=64, model_type='pinn'),
    training=TrainingParams(nepochs=100),
    # ... other parameters
)

# 2. Update legacy dictionary (one-way street)
update_legacy_dict(params.cfg, config)

# 3. Load data and populate remaining params.cfg keys
data = load_data(config.data.train_data_file)
params.cfg.update({'data_specific_key': data.some_property})

# 4. NOW safe to import modules that depend on global state
from ptycho.workflows import run_training_workflow
```

### Configuration System Rules
- **New code**: Always accept dataclass configurations as arguments
- **Legacy compatibility**: Use `update_legacy_dict()` to maintain backward compatibility
- **Global state**: Avoid `params.get()` in new functions
- **One-way flow**: Modern config → Legacy dict (never reverse)

## Module-Specific Guidance

### Data Loading (`loader.py`, `raw_data.py`)
- **Purpose**: Convert NPZ files to model-ready containers
- **Key classes**: `RawData`, `PtychoDataContainer`
- **Critical**: Always verify data contracts before processing

### Evaluation (`evaluation.py`)
- **Metrics**: MAE, MSE, PSNR, SSIM, MS-SSIM, FRC
- **Registration**: Automatic alignment for fair comparison
- **Usage**: Integrated into model comparison workflows

### Baselines (`baselines.py`)
- **Purpose**: Supervised baseline model implementations
- **Usage**: For comparison with PINN models
- **Training**: Uses same data pipeline as PINN models

### Miscellaneous (`misc.py`)
- **Caching**: `@memoize_disk_and_memory`, `@memoize_simulated_data`
- **Purpose**: Speed up repeated computations
- **Usage**: Apply to expensive functions that benefit from caching

## Development Guidelines

### Adding New Functionality
1. **Design**: Use modern dataclass configurations
2. **Interface**: Accept explicit arguments, avoid global state
3. **Testing**: Write unit tests for new functions
4. **Documentation**: Update relevant CLAUDE.md files
5. **Integration**: Update workflow scripts if needed

### Debugging Core Issues
1. **Data first**: Always verify data format using data contracts
2. **Configuration**: Check modern vs legacy config consistency
3. **Imports**: Verify no side effects on module import
4. **Physics**: Only modify if explicitly requested and well-justified

### Performance Optimization
1. **Caching**: Use decorators from misc.py for expensive operations
2. **Batching**: Leverage TensorFlow's batch processing capabilities
3. **Memory**: Monitor GPU memory usage, especially with large datasets
4. **Profiling**: Use TensorFlow profiler for bottleneck identification

## Cross-References

- **Architecture details**: <doc-ref type="guide">docs/DEVELOPER_GUIDE.md</doc-ref>
- **Data structure**: <doc-ref type="technical">ptycho/loader_structure.md</doc-ref>
- **Configuration guide**: <doc-ref type="guide">docs/CONFIGURATION_GUIDE.md</doc-ref>
- **Data contracts**: <doc-ref type="contract">docs/data_contracts.md</doc-ref>
- **FRC implementation**: <doc-ref type="workflow-guide">ptycho/FRC/CLAUDE.md</doc-ref>