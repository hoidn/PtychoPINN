# PtychoPINN Module Dependency Analysis and Documentation Priority

## Executive Summary

Based on analysis of import statements and module relationships in the ptycho library, I've identified the dependency hierarchy and created a prioritized documentation order. The analysis reveals 4 distinct dependency levels, from foundational modules with no internal dependencies to high-level orchestration modules that integrate the entire pipeline.

## Dependency Analysis Method

I examined the `import` statements in each Python file to map internal dependencies within the ptycho library. External dependencies (numpy, tensorflow, etc.) were noted but not considered for prioritization. The focus was on understanding which ptycho modules depend on other ptycho modules.

## Dependency Levels and Documentation Priority

### Level 0: Foundational Modules (Document First)
**Characteristics**: No internal ptycho dependencies, provide core functionality

1. **ptycho/params.py** 
   - **Dependencies**: Only external (numpy, tensorflow)
   - **Role**: Legacy global configuration system
   - **Priority**: HIGHEST - Most imported module (23+ consumers)
   - **Status**: Deprecated but critical for backward compatibility

2. **ptycho/config/config.py**
   - **Dependencies**: Only external (dataclasses, pathlib, yaml)
   - **Role**: Modern dataclass-based configuration system
   - **Priority**: HIGHEST - Replacement for params.py
   - **Integration**: Updates legacy params.cfg for compatibility

### Level 1: Core Library Modules (Document Second)
**Characteristics**: Depend only on Level 0 modules, provide essential functionality

3. **ptycho/probe.py**
   - **Dependencies**: ptycho.params, external (tensorflow, numpy)
   - **Role**: Probe initialization and manipulation
   - **Priority**: HIGH - Core physics component

4. **ptycho/losses.py**
   - **Dependencies**: None (currently only comments)
   - **Role**: Custom loss function definitions (placeholder)
   - **Priority**: LOW - No active code, future development

5. **ptycho/diffsim.py**
   - **Dependencies**: ptycho.params, ptycho.tf_helper, external modules
   - **Role**: Forward physics simulation engine
   - **Priority**: HIGH - Core physics implementation

### Level 2: Infrastructure Modules (Document Third)
**Characteristics**: Depend on Level 0-1 modules, provide data pipeline and model infrastructure

6. **ptycho/tf_helper.py**
   - **Dependencies**: ptycho.params, ptycho.autotest.debug, ptycho.projective_warp_xla
   - **Role**: Core tensor operations for ptychographic reconstruction
   - **Priority**: VERY HIGH - Critical physics operations, heavily used
   - **Note**: Protected module - stable physics implementation

7. **ptycho/custom_layers.py**
   - **Dependencies**: Only external (tensorflow)
   - **Role**: Custom Keras layers for proper serialization
   - **Priority**: MEDIUM - Important for model persistence

8. **ptycho/raw_data.py**
   - **Dependencies**: ptycho.params, ptycho.config.config, ptycho.autotest.debug, ptycho.diffsim, ptycho.tf_helper
   - **Role**: Core data ingestion and preprocessing
   - **Priority**: HIGH - First stage of data pipeline

### Level 3: High-Level Processing Modules (Document Fourth)
**Characteristics**: Depend on multiple lower-level modules, provide complete functionality

9. **ptycho/loader.py**
   - **Dependencies**: ptycho.params, ptycho.autotest.debug, ptycho.diffsim, ptycho.tf_helper, ptycho.raw_data
   - **Role**: TensorFlow-ready data pipeline finalizer
   - **Priority**: HIGH - Final stage of data pipeline (9 importing modules)

10. **ptycho/model.py**
    - **Dependencies**: ptycho.custom_layers, ptycho.loader, ptycho.tf_helper, ptycho.params, ptycho.probe, ptycho.gaussian_filter
    - **Role**: Core physics-informed neural network architecture
    - **Priority**: VERY HIGH - Heart of the PtychoPINN system
    - **Note**: Protected module - stable, validated implementation

11. **ptycho/evaluation.py**
    - **Dependencies**: ptycho.params, ptycho.misc
    - **Role**: Quality assessment and metrics orchestration
    - **Priority**: MEDIUM - Important for model validation

12. **ptycho/model_manager.py**
    - **Dependencies**: ptycho.params
    - **Role**: Model lifecycle management and persistence
    - **Priority**: MEDIUM - Important for model saving/loading

### Level 4: Orchestration Modules (Document Last)
**Characteristics**: Integrate multiple modules into complete workflows

13. **ptycho/workflows/components.py**
    - **Dependencies**: ptycho.params, ptycho.probe, ptycho.loader, ptycho.raw_data, ptycho.config.config, ptycho.image.reassemble_patches
    - **Role**: High-level workflow orchestration
    - **Priority**: MEDIUM - Integrates other modules but not core functionality

## Recommended Documentation Order

### Phase 1: Foundation (Critical Path)
1. **ptycho/params.py** - Most critical due to widespread usage
2. **ptycho/config/config.py** - Modern replacement system
3. **ptycho/tf_helper.py** - Core physics operations

### Phase 2: Core Physics & Data (Essential Components)
4. **ptycho/probe.py** - Core physics component
5. **ptycho/diffsim.py** - Forward physics simulation
6. **ptycho/raw_data.py** - Data pipeline entry point

### Phase 3: Model Architecture (Core Implementation)
7. **ptycho/model.py** - Main neural network architecture
8. **ptycho/loader.py** - Data pipeline finalizer
9. **ptycho/custom_layers.py** - Model persistence support

### Phase 4: Support Systems (Completion)
10. **ptycho/evaluation.py** - Metrics and assessment
11. **ptycho/model_manager.py** - Model persistence
12. **ptycho/workflows/components.py** - High-level orchestration
13. **ptycho/losses.py** - Future development (low priority)

## Key Insights

1. **Critical Dependencies**: params.py is the most critical module with 23+ consumers
2. **Protected Modules**: model.py, diffsim.py, and tf_helper.py are marked as stable core physics
3. **Migration Pattern**: config/config.py represents modernization of params.py
4. **Data Pipeline**: raw_data.py → loader.py → model.py represents the core data flow
5. **Orchestration Layer**: workflows/components.py integrates everything for complete workflows

## Documentation Strategy Recommendations

1. **Start with params.py** despite deprecation - it's the most imported module
2. **Prioritize physics modules** (tf_helper.py, diffsim.py, model.py) as they're marked protected/stable
3. **Document data pipeline in order** (raw_data.py → loader.py)
4. **Treat model.py as central hub** - it integrates many other modules
5. **Leave orchestration modules for last** as they primarily integrate existing functionality

This analysis provides a clear roadmap for systematic documentation that respects the actual dependency relationships and usage patterns in the codebase.