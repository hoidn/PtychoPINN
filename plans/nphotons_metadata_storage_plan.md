# NPZ Metadata Storage Plan: Comprehensive nphotons Integration

## Executive Summary

This plan addresses a critical data provenance issue in PtychoPINN: **nphotons parameter values are lost when NPZ datasets are created, leading to inconsistencies between simulation, training, and inference phases**. The solution implements a robust metadata storage system that preserves nphotons (and other critical parameters) within NPZ files while maintaining backward compatibility with existing datasets.

**Impact**: Eliminates parameter mismatches, enables proper data provenance, and supports advanced workflows like multi-photon studies and dataset merging.

## Problem Statement

### Current Issues

1. **Lost Data Provenance**: NPZ files contain no record of the nphotons value used during creation
   ```python
   # Current workflow - nphotons is lost
   config = TrainingConfig(nphotons=1e6, ...)  
   simulate_and_save(config, "sim_data.npz")  # Creates NPZ but nphotons not stored
   
   # Later, during training - no way to know original nphotons
   raw_data = RawData.from_file("sim_data.npz")  # nphotons metadata missing
   config_train = TrainingConfig(nphotons=1e9, ...)  # Different value!
   ```

2. **Parameter Inconsistencies**: Different nphotons values between simulation and training/inference
   - Simulation uses nphotons=1e6 (stored in config only)
   - Training loads same data but uses nphotons=1e9 (from new config)
   - Results in incorrect physics scaling and poor model performance

3. **Workflow Limitations**: 
   - Cannot determine simulation parameters from existing datasets
   - Multi-photon comparison studies require manual parameter tracking
   - Dataset merging loses metadata from individual sources
   - Tools that modify NPZ files don't preserve creation parameters

### Specific Code Examples

**Current Simulation (metadata lost):**
```python
# scripts/simulation/simulate_and_save.py - line 117
np.savez_compressed(output_file_path, **data_dict)  # No metadata!
```

**Current Loading (no metadata available):**
```python
# ptycho/raw_data.py - line 295
train_data = np.load(train_data_file_path)  # Missing nphotons info
# No way to validate config.nphotons matches original simulation
```

**Current Physics Simulation (parameter mismatch):**
```python
# ptycho/diffsim.py - line 185
norm = tf.math.sqrt(p.get('nphotons') / mean_photons)  # Uses current config
# But current config might differ from original simulation parameters!
```

## Proposed Solution Architecture

### 1. Metadata Storage Schema

**New NPZ Structure:**
```
dataset.npz:
├── diffraction        # Existing data arrays
├── xcoords           
├── ycoords
├── objectGuess
├── probeGuess
└── _metadata         # NEW: JSON string containing all parameters
```

**Metadata Format (JSON within NPZ):**
```python
metadata = {
    "creation_info": {
        "timestamp": "2025-08-26T10:30:00Z",
        "script": "scripts/simulation/simulate_and_save.py", 
        "version": "1.0.0",
        "hostname": "compute-node-01"
    },
    "physics_parameters": {
        "nphotons": 1e6,
        "gridsize": 1,
        "N": 64,
        "probe_trainable": false,
        "intensity_scale_trainable": true
    },
    "simulation_parameters": {  # For simulated data
        "data_source": "synthetic",
        "buffer": 32,
        "seed": 42,
        "n_images": 2000
    },
    "data_transformations": [  # Track processing history
        {
            "tool": "split_dataset_tool.py",
            "timestamp": "2025-08-26T11:00:00Z",
            "operation": "train_test_split",
            "parameters": {"split_fraction": 0.8, "split_axis": "x"}
        }
    ],
    "compatibility": {
        "min_ptychopinn_version": "2.0.0",
        "data_contract_version": "1.2"
    }
}
```

### 2. Core Implementation Components

#### A. MetadataManager Class (NEW)
```python
# ptycho/metadata.py (NEW FILE)
class MetadataManager:
    @staticmethod
    def create_metadata(config: TrainingConfig, script_name: str, **kwargs) -> dict:
        """Create metadata dictionary from configuration."""
        
    @staticmethod 
    def save_with_metadata(file_path: str, data_dict: dict, metadata: dict) -> None:
        """Save NPZ with embedded JSON metadata."""
        
    @staticmethod
    def load_with_metadata(file_path: str) -> tuple[dict, Optional[dict]]:
        """Load NPZ and extract metadata if present."""
        
    @staticmethod
    def validate_parameters(metadata: dict, current_config: TrainingConfig) -> list[str]:
        """Validate current config against stored metadata."""
```

#### B. RawData Integration
```python
# ptycho/raw_data.py - ENHANCED
class RawData:
    def __init__(self, ..., metadata: Optional[dict] = None):
        # Add metadata attribute
        self.metadata = metadata
    
    @staticmethod
    def from_file(train_data_file_path: str, validate_config: bool = True) -> 'RawData':
        """Enhanced to load and validate metadata."""
        data_dict, metadata = MetadataManager.load_with_metadata(train_data_file_path)
        
        if validate_config and metadata:
            current_config = get_current_config()  # From global params or context
            warnings = MetadataManager.validate_parameters(metadata, current_config)
            for warning in warnings:
                logging.warning(f"Parameter mismatch: {warning}")
        
        return RawData(..., metadata=metadata)
    
    def to_file(self, file_path: str) -> None:
        """Enhanced to preserve metadata."""
        if self.metadata:
            MetadataManager.save_with_metadata(file_path, data_dict, self.metadata)
        else:
            # Fallback for legacy data
            np.savez(file_path, **data_dict)
```

#### C. Configuration System Integration
```python
# ptycho/config/config.py - ENHANCED
@dataclass(frozen=True)
class TrainingConfig:
    # Existing fields...
    preserve_metadata: bool = True  # NEW: Control metadata handling
    require_metadata_validation: bool = True  # NEW: Strict parameter checking

def load_config_from_metadata(metadata: dict) -> TrainingConfig:
    """Recreate configuration from NPZ metadata."""
    physics_params = metadata.get("physics_parameters", {})
    return TrainingConfig(
        nphotons=physics_params.get("nphotons", 1e9),
        model=ModelConfig(
            N=physics_params.get("N", 64),
            gridsize=physics_params.get("gridsize", 1)
        )
    )
```

### 3. Workflow Integration Points

#### A. Simulation Scripts
```python
# scripts/simulation/simulate_and_save.py - ENHANCED
def simulate_and_save(config: TrainingConfig, input_file_path, output_file_path, **kwargs):
    # Existing simulation logic...
    
    # NEW: Create comprehensive metadata
    metadata = MetadataManager.create_metadata(
        config=config,
        script_name="simulate_and_save.py",
        simulation_type="synthetic",
        buffer=buffer,
        seed=seed
    )
    
    # NEW: Save with metadata
    MetadataManager.save_with_metadata(
        output_file_path, 
        data_dict, 
        metadata
    )
```

#### B. Data Processing Tools
```python
# scripts/tools/split_dataset_tool.py - ENHANCED
def split_dataset(input_path: Path, output_dir: Path, split_fraction: float, split_axis: str):
    # Load with metadata
    data_dict, metadata = MetadataManager.load_with_metadata(str(input_path))
    
    # Split data (existing logic)...
    
    # NEW: Update metadata for both outputs
    train_metadata = copy.deepcopy(metadata) if metadata else {}
    test_metadata = copy.deepcopy(metadata) if metadata else {}
    
    # Add transformation record
    transform_info = {
        "tool": "split_dataset_tool.py",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "operation": "train_test_split", 
        "parameters": {
            "split_fraction": split_fraction,
            "split_axis": split_axis,
            "parent_file": str(input_path)
        }
    }
    
    for meta in [train_metadata, test_metadata]:
        meta.setdefault("data_transformations", []).append(transform_info)
    
    # Save with preserved and updated metadata
    MetadataManager.save_with_metadata(train_path, train_dict, train_metadata)
    MetadataManager.save_with_metadata(test_path, test_dict, test_metadata)
```

#### C. Training Integration
```python
# scripts/training/train.py - ENHANCED
def main():
    args = parse_arguments()
    config = setup_configuration(args, args.config)
    
    # NEW: Load data with metadata validation
    ptycho_data = load_data_with_validation(
        str(config.train_data_file), 
        current_config=config,
        require_validation=config.require_metadata_validation
    )
    
    # Check for parameter mismatches
    if ptycho_data.metadata:
        original_nphotons = ptycho_data.metadata.get("physics_parameters", {}).get("nphotons")
        if original_nphotons and abs(original_nphotons - config.nphotons) > 1e-6:
            if config.require_metadata_validation:
                raise ValueError(f"Config nphotons ({config.nphotons}) != dataset nphotons ({original_nphotons})")
            else:
                logger.warning(f"Using config nphotons ({config.nphotons}) instead of dataset value ({original_nphotons})")
```

### 4. Backward Compatibility Strategy

#### A. Graceful Degradation
- **Old NPZ files**: Work unchanged, no metadata available but no errors
- **New NPZ files**: Provide full metadata capabilities
- **Mixed workflows**: Detect and handle both formats seamlessly

#### B. Migration Utilities
```python
# scripts/tools/add_metadata_tool.py (NEW)
def add_metadata_to_legacy_file(
    input_path: str, 
    output_path: str,
    nphotons: float,
    **estimated_params
) -> None:
    """Add estimated metadata to legacy NPZ files."""
    data_dict, _ = MetadataManager.load_with_metadata(input_path)
    
    # Create best-guess metadata
    metadata = {
        "creation_info": {
            "timestamp": "unknown",
            "script": "legacy_data", 
            "version": "unknown"
        },
        "physics_parameters": {
            "nphotons": nphotons,
            **estimated_params
        },
        "data_transformations": [
            {
                "tool": "add_metadata_tool.py",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "operation": "metadata_estimation",
                "note": "Metadata estimated retroactively"
            }
        ]
    }
    
    MetadataManager.save_with_metadata(output_path, data_dict, metadata)
```

#### C. Validation Modes
```python
# Three validation modes for different use cases:
ValidationMode.STRICT     # Require metadata, fail if mismatch
ValidationMode.WARN       # Warn about mismatches, continue
ValidationMode.PERMISSIVE # No validation, backward compatibility
```

## Implementation Phases

### Phase 1: Core Infrastructure (Week 1-2)
**Goal**: Establish metadata framework and basic validation
**Deliverables**: 
- `ptycho/metadata.py` with MetadataManager class
- Enhanced RawData class with metadata support
- Basic validation and warning system
- Unit tests for core functionality

**Success Criteria**:
- Can create, save, and load metadata from NPZ files
- RawData properly handles both old and new format files
- Parameter validation warns about mismatches

**Implementation Tasks**:
```python
# Test implementation
def test_metadata_roundtrip():
    config = TrainingConfig(nphotons=1e6, model=ModelConfig(N=64))
    metadata = MetadataManager.create_metadata(config, "test_script")
    
    # Save and load
    data_dict = {"test_array": np.random.rand(10, 64, 64)}
    MetadataManager.save_with_metadata("test.npz", data_dict, metadata)
    loaded_data, loaded_metadata = MetadataManager.load_with_metadata("test.npz")
    
    assert loaded_metadata["physics_parameters"]["nphotons"] == 1e6
    assert loaded_data["test_array"].shape == (10, 64, 64)
```

### Phase 2: Simulation Integration (Week 3)
**Goal**: All simulation scripts save metadata
**Deliverables**:
- Enhanced `simulate_and_save.py` with metadata
- Updated `nongrid_simulation.py` integration
- Simulation validation utilities

**Success Criteria**:
- All simulated datasets include comprehensive metadata
- Simulation parameters fully preserved
- Can recreate simulation config from NPZ metadata

**Code Changes**:
```python
# Enhanced simulate_and_save.py
def simulate_and_save(config: TrainingConfig, ...):
    # Existing simulation logic
    raw_data_instance, ground_truth_patches = generate_simulated_data(...)
    
    # NEW: Create metadata with full parameter set
    metadata = MetadataManager.create_metadata(
        config=config,
        script_name="simulate_and_save.py",
        input_file=str(input_file_path),
        buffer=buffer,
        seed=seed,
        simulation_type="coordinate_based"
    )
    
    data_dict = {
        'xcoords': raw_data_instance.xcoords,
        'ycoords': raw_data_instance.ycoords,
        # ... existing keys
        'ground_truth_patches': ground_truth_patches
    }
    
    # Save with metadata
    MetadataManager.save_with_metadata(output_file_path, data_dict, metadata)
```

### Phase 3: Training/Inference Integration (Week 4)  
**Goal**: Training and inference validate metadata
**Deliverables**:
- Enhanced training scripts with parameter validation
- Inference scripts that use stored metadata
- Comprehensive parameter mismatch detection

**Success Criteria**:
- Training detects and warns about parameter mismatches
- Inference can use original simulation parameters when needed
- Clear error messages for incompatible configurations

**Key Integration Points**:
```python
# Enhanced training workflow
def load_data_with_validation(file_path: str, current_config: TrainingConfig):
    raw_data = RawData.from_file(file_path, validate_config=True)
    
    if raw_data.metadata:
        original_params = raw_data.metadata["physics_parameters"]
        mismatches = []
        
        if abs(original_params.get("nphotons", 0) - current_config.nphotons) > 1e-6:
            mismatches.append(f"nphotons: dataset={original_params['nphotons']}, config={current_config.nphotons}")
        
        if original_params.get("N") != current_config.model.N:
            mismatches.append(f"N: dataset={original_params['N']}, config={current_config.model.N}")
            
        if mismatches and current_config.require_metadata_validation:
            raise ValueError(f"Parameter mismatches detected: {'; '.join(mismatches)}")
        elif mismatches:
            for mismatch in mismatches:
                logger.warning(f"Parameter mismatch: {mismatch}")
    
    return raw_data
```

### Phase 4: Tools Integration (Week 5)
**Goal**: All data processing tools preserve metadata
**Deliverables**:
- Enhanced split, update, transpose, and conversion tools
- Metadata transformation tracking
- Legacy data migration utilities

**Success Criteria**:
- All tools preserve and update metadata appropriately
- Processing history tracked in transformed datasets
- Can trace data lineage from any NPZ file

**Tool Enhancement Pattern**:
```python
# Pattern for enhancing tools
def enhanced_tool_function(input_path: str, output_path: str, **tool_params):
    # Load with metadata
    data_dict, metadata = MetadataManager.load_with_metadata(input_path)
    
    # Perform tool-specific processing
    processed_data = process_data(data_dict, **tool_params)
    
    # Update metadata with transformation record
    if metadata is None:
        metadata = {"data_transformations": []}
        
    transformation = {
        "tool": os.path.basename(__file__),
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "operation": "tool_specific_operation",
        "parameters": tool_params,
        "input_file": input_path
    }
    metadata.setdefault("data_transformations", []).append(transformation)
    
    # Save with updated metadata
    MetadataManager.save_with_metadata(output_path, processed_data, metadata)
```

### Phase 5: Advanced Features (Week 6)
**Goal**: Multi-photon studies and advanced metadata features
**Deliverables**:
- Multi-photon dataset merging with metadata preservation
- Advanced validation and parameter suggestion systems
- Dataset provenance visualization tools

**Success Criteria**:
- Can merge datasets with different nphotons values intelligently
- Automated parameter suggestion based on dataset history
- Visual tools for understanding dataset lineage

**Advanced Features**:
```python
# Multi-photon dataset merging
def merge_datasets_with_metadata(
    dataset_paths: List[str], 
    output_path: str,
    merge_strategy: str = "preserve_separate"
) -> None:
    """Merge multiple datasets while preserving individual metadata."""
    
    all_data = []
    all_metadata = []
    
    for path in dataset_paths:
        data, metadata = MetadataManager.load_with_metadata(path)
        all_data.append(data)
        all_metadata.append(metadata)
    
    # Merge data arrays
    merged_data = merge_data_arrays(all_data)
    
    # Create composite metadata
    composite_metadata = {
        "creation_info": {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "script": "merge_datasets_tool.py",
            "operation": "dataset_merge"
        },
        "source_datasets": [
            {
                "path": path,
                "metadata": metadata,
                "n_images": len(data["xcoords"])
            } 
            for path, data, metadata in zip(dataset_paths, all_data, all_metadata)
        ],
        "merge_strategy": merge_strategy,
        "physics_parameters": determine_merged_physics_params(all_metadata)
    }
    
    MetadataManager.save_with_metadata(output_path, merged_data, composite_metadata)
```

## Testing Strategy

### Unit Tests
```python
# tests/test_metadata_manager.py
class TestMetadataManager:
    def test_create_metadata_from_config(self):
        """Test metadata creation from TrainingConfig."""
        
    def test_save_and_load_roundtrip(self):
        """Test saving and loading NPZ with metadata."""
        
    def test_parameter_validation(self):
        """Test parameter mismatch detection."""
        
    def test_backward_compatibility(self):
        """Test handling of legacy NPZ files without metadata."""

# tests/test_integration_with_metadata.py  
class TestMetadataIntegration:
    def test_full_simulation_to_training_pipeline(self):
        """Test complete pipeline preserves parameters."""
        
    def test_data_tool_metadata_preservation(self):
        """Test that data tools preserve metadata."""
        
    def test_multi_photon_workflow(self):
        """Test workflows with different nphotons values."""
```

### Integration Tests
```python
def test_simulation_training_consistency():
    """Verify simulation and training use consistent parameters."""
    # Create simulation with known parameters
    config = TrainingConfig(nphotons=1e6, model=ModelConfig(N=64))
    simulate_and_save(config, "test_sim.npz")
    
    # Load for training with different config  
    train_config = TrainingConfig(nphotons=1e9, model=ModelConfig(N=64))
    
    # Should detect mismatch
    with pytest.warns(UserWarning, match="nphotons mismatch"):
        raw_data = RawData.from_file("test_sim.npz", validate_config=True)

def test_tool_chain_metadata_preservation():
    """Test metadata preservation through tool chain."""
    # Create initial dataset with metadata
    # Split dataset -> should preserve metadata
    # Update object -> should preserve and extend metadata
    # Final dataset should have complete transformation history
```

### Validation Tests
```python  
def test_parameter_validation_modes():
    """Test different validation modes."""
    # STRICT mode should fail on mismatch
    # WARN mode should continue with warning
    # PERMISSIVE mode should ignore mismatches

def test_metadata_schema_validation():
    """Test metadata follows expected schema."""
    # Validate JSON structure
    # Check required fields
    # Verify data types
```

## Potential Risks and Mitigations

### Risk 1: File Size Increase
**Risk**: Adding metadata increases NPZ file sizes
**Mitigation**: 
- JSON metadata typically <1KB, negligible for GB-sized datasets
- Use JSON compression within NPZ
- Optional metadata storage via config flag

### Risk 2: Backward Compatibility Issues  
**Risk**: Changes break existing workflows
**Mitigation**:
- All changes are additive - old NPZ files work unchanged
- Graceful degradation when metadata absent
- Comprehensive testing with legacy datasets
- Phased rollout with opt-in validation

### Risk 3: Complex Parameter Validation Logic
**Risk**: Validation becomes overly complex and error-prone
**Mitigation**:
- Start with simple parameter matching (nphotons, N, gridsize)
- Add complexity incrementally based on user feedback
- Clear error messages and suggested fixes
- Multiple validation modes (strict/warn/permissive)

### Risk 4: Performance Impact
**Risk**: Metadata loading/saving adds overhead
**Mitigation**:
- JSON parsing is fast for small metadata
- Lazy loading - only parse metadata when needed
- Cache parsed metadata in RawData objects
- Benchmark critical workflows

### Risk 5: Metadata Schema Evolution
**Risk**: Future schema changes break compatibility
**Mitigation**:
- Version metadata schema explicitly
- Design schema for forward/backward compatibility
- Migration utilities for schema upgrades
- Comprehensive schema documentation

## Migration Strategy for Existing Datasets

### 1. Automatic Detection
```python
def detect_dataset_parameters(npz_path: str) -> dict:
    """Automatically detect parameters from dataset characteristics."""
    data = np.load(npz_path)
    
    # Infer parameters from data properties
    estimated_params = {
        "N": data["probeGuess"].shape[0],
        "n_images": len(data["xcoords"]), 
        "gridsize": 1,  # Default assumption
    }
    
    # Analyze diffraction patterns for nphotons estimation
    mean_intensity = np.mean(data["diffraction"]**2)
    estimated_params["nphotons"] = estimate_nphotons_from_intensity(mean_intensity)
    
    return estimated_params
```

### 2. User-Guided Migration
```bash
# Interactive migration tool
python scripts/tools/migrate_legacy_datasets.py \
    --input-dir datasets/legacy/ \
    --output-dir datasets/with_metadata/ \
    --interactive  # Prompt for unknown parameters
```

### 3. Batch Migration with Parameter Files
```yaml
# migration_config.yaml
datasets:
  - path: "datasets/fly64_train.npz"
    nphotons: 1e6
    gridsize: 1
    data_source: "experimental"
    
  - path: "datasets/synthetic_lines.npz" 
    nphotons: 1e9
    gridsize: 2
    data_source: "synthetic"
```

## Expected Benefits

### 1. Eliminated Parameter Mismatches
- **Before**: Silent parameter inconsistencies leading to poor model performance
- **After**: Automatic detection and validation of parameter mismatches

### 2. Complete Data Provenance  
- **Before**: No record of how datasets were created or processed
- **After**: Full transformation history from creation through all processing steps

### 3. Advanced Workflows Enabled
- **Multi-photon Studies**: Compare models trained on different photon levels
- **Parameter Optimization**: Systematic studies of parameter effects
- **Dataset Merging**: Intelligently combine datasets with different parameters

### 4. Improved Reproducibility
- **Before**: Difficult to recreate exact simulation conditions
- **After**: All parameters preserved for perfect reproducibility

### 5. Better Error Messages
- **Before**: Cryptic TensorFlow errors from parameter mismatches  
- **After**: Clear warnings about specific parameter conflicts with suggested fixes

## Implementation Priority

**High Priority (Essential)**:
1. MetadataManager core functionality
2. RawData integration with validation
3. simulate_and_save.py metadata creation
4. Basic parameter validation (nphotons, N, gridsize)

**Medium Priority (Important)**:
5. Training script validation integration
6. Data tools metadata preservation
7. Migration utilities for legacy data
8. Comprehensive test suite

**Low Priority (Nice-to-have)**:
9. Advanced merging and analysis tools
10. Metadata visualization utilities  
11. Automated parameter suggestion
12. Schema evolution management

## Conclusion

This comprehensive plan addresses the critical issue of lost nphotons metadata in PtychoPINN's data pipeline. By implementing a robust metadata storage system within NPZ files, we eliminate parameter mismatches, enable complete data provenance, and unlock advanced workflows while maintaining full backward compatibility with existing datasets.

The phased implementation approach ensures minimal disruption to current workflows while providing immediate benefits from the core infrastructure. The extensive testing strategy and risk mitigation measures ensure reliable operation across all use cases.

**Key Success Metrics**:
- Zero parameter mismatch errors in production workflows
- 100% data provenance for new datasets
- Successful migration of all legacy datasets
- No performance degradation in critical workflows
- Full backward compatibility maintained

This implementation will significantly improve the reliability and reproducibility of PtychoPINN workflows while enabling new capabilities for advanced parameter studies and dataset analysis.