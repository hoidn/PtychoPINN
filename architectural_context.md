# PtychoPINN Architectural Context for Documentation Agents

## Project Overview
PtychoPINN is a TensorFlow-based physics-informed neural network for ptychographic reconstruction. It combines deep learning with domain-specific physics constraints to achieve rapid, high-resolution reconstruction from coherent diffraction data.

## Dual-System Architecture (Critical Understanding)

### Legacy System (Grid-Based)
- **Workflow**: `ptycho/train.py` (deprecated)
- **Configuration**: Global `ptycho.params.cfg` dictionary
- **Characteristics**: 
  - Grid-based patch processing
  - Global state dependencies
  - Implicit configuration flow

### Modern System (Coordinate-Based) 
- **Workflow**: `scripts/` directory entry points
- **Configuration**: Dataclass-based (`ptycho.config.config`)
- **Characteristics**:
  - Coordinate-based non-uniform processing
  - Explicit data flow
  - Modern configuration management

**Rule**: New documentation must acknowledge both systems where relevant.

## Core Data Flow Pipeline

```
Raw NPZ File → RawData (raw_data.py) → PtychoDataContainer (loader.py) → Model (model.py) → Reconstruction
```

### Data Contracts
- **NPZ Format**: Defined in `specs/data_contracts.md`
- **Key Arrays**: `probeGuess`, `objectGuess`, `diffraction`, `xcoords`, `ycoords`
- **Critical**: `diffraction` is amplitude (sqrt of intensity), not intensity itself

### Tensor Flow Patterns
- **Shapes**: Document all input/output tensor shapes
- **Formats**: Grid Format `(B,G,G,N,N,1)` vs Channel Format `(B,N,N,G*G)` vs Flat Format `(B*G*G,N,N,1)`
- **Conversions**: Handled by `tf_helper.py`

## Module Categories by Function

### Tier 1: Core Infrastructure
- `model.py` - Neural network architecture (U-Net + physics layers)
- `loader.py` - Data transformation pipeline  
- `tf_helper.py` - Low-level tensor operations
- `raw_data.py` - Data ingestion and grouping
- `evaluation.py` - Metrics and quality assessment

### Tier 2: Physics & Simulation  
- `diffsim.py` - Forward physics model
- `physics.py` - Domain-specific computations
- `fourier.py` - Frequency domain operations
- `probe.py` - Scanning probe utilities

### Tier 3: Data Processing
- `image/` - Modern image processing toolkit
- `data_preprocessing.py` - Data preparation utilities
- `baselines.py` - Supervised learning baselines

### Tier 4: Utilities & Support
- `misc.py` - Caching decorators and utilities
- `cli_args.py` - Command-line interface helpers
- `log_config.py` - Centralized logging system
- `visualization.py` - Display and plotting

### Tier 5: Specialized Features
- `experimental.py` - Research features
- `workflows/` - High-level orchestration
- `config/` - Configuration management
- `autotest/` - Internal testing framework

## Legacy System Dependencies

### Critical Global State (`ptycho.params`)
Many modules depend on the global `params.cfg` dictionary:
- **Heavy users**: `raw_data.py`, `model.py`, `loader.py`, `baselines.py`
- **Behavior changes**: `gridsize` parameter completely changes data sampling algorithm
- **Documentation requirement**: Must document these hidden dependencies

### Configuration Flow
```
Modern Config (dataclass) → update_legacy_dict() → params.cfg → Module behavior
```

## Common Integration Patterns

### Data Transformation Modules
```python
# Typical pattern for tf_helper.py, fourier.py, etc.
input_tensor = load_data()  # Shape: (B, N, N, 1)
processed = module_function(input_tensor, parameters)  # Shape: (B, M, M, C)
output = downstream_consumer(processed)
```

### Logic/Control Modules  
```python
# Typical pattern for raw_data.py, model.py, etc.
config = load_configuration()
if config.mode == 'A':
    result = mode_a_processing(data)
else:
    result = mode_b_processing(data)
return result
```

### Workflow Integration
```python
# Typical high-level workflow pattern
config = setup_configuration()
data = load_and_process_data(config)
model = create_model(config)
results = train_model(model, data, config)
metrics = evaluate_results(results, ground_truth)
```

## Anti-Patterns to Document

1. **Hidden Side Effects**: Modules that perform complex operations on import
2. **Global State Dependency**: Functions that rely on `params.cfg` without explicit mention
3. **Implicit Data Flow**: Functions that don't clearly document input/output formats
4. **Legacy Assumptions**: Code that assumes grid-based processing only

## Documentation Requirements

### For Data Transformation Modules
- Input/output tensor shapes and types
- Key transformation functions and their effects
- Integration with tensor format conversion pipeline
- Performance characteristics

### For Logic/Control Modules  
- Behavioral modes and their triggers
- Configuration dependencies (especially `params.cfg`)
- Decision points and their effects on system behavior
- Integration with both legacy and modern systems

### For All Modules
- Position in the overall data pipeline
- Primary consumers and usage patterns
- Dependencies on other modules
- Legacy system interactions (if any)

## Success Criteria for Docstrings

1. **Architectural Awareness**: Correctly describes module's role in PtychoPINN system
2. **Data Contracts**: Explicitly documents data formats and shapes where relevant
3. **Integration Patterns**: Shows realistic multi-step usage examples
4. **Dependency Transparency**: Documents hidden dependencies (especially `params.cfg`)
5. **System Context**: Acknowledges dual-system architecture where applicable

This context should guide all sub-agents in creating documentation that accurately reflects the PtychoPINN system's design and usage patterns.