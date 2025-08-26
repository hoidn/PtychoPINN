# NPhotons Metadata Integration Test Suite

## Overview

This comprehensive integration test suite (`test_nphotons_metadata_integration.py`) validates the complete nphotons metadata system in PtychoPINN. It ensures that the critical physics parameter `nphotons` is correctly preserved and validated throughout the entire workflow from simulation through training to inference.

## Background

The `nphotons` parameter is crucial for photon noise modeling and physics simulation in ptychography. Inconsistent values between simulation, training, and inference can lead to incorrect reconstructions. The metadata system stores physics parameters in NPZ files to maintain consistency and enable validation.

## Test Structure

The test suite follows PtychoPINN's integration testing patterns:
- Uses `subprocess` calls to test actual command-line workflows
- Creates isolated temporary directories for each test
- Tests real user scenarios across separate processes
- Validates both successful workflows and error conditions

## Test Coverage

### 1. **Single Workflow Test** (`test_metadata_persistence_single_nphotons`)
- **Purpose**: Validates complete simulation → training → inference workflow
- **Process**: 
  - Simulates data with nphotons=1e6
  - Trains model with the simulated data
  - Runs inference with the trained model
  - Verifies output files are created correctly
- **Validates**: End-to-end metadata persistence

### 2. **Multiple Values Consistency** (`test_multiple_nphotons_metadata_consistency`) 
- **Purpose**: Tests metadata accuracy across different nphotons values
- **Process**: Simulates datasets with nphotons=[1e4, 1e6, 1e8]
- **Validates**: Each dataset stores correct nphotons value in metadata

### 3. **Configuration Mismatch Warnings** (`test_configuration_mismatch_warnings`)
- **Purpose**: Tests parameter validation and warning generation
- **Process**: Creates config with different nphotons than stored in metadata
- **Validates**: Warning system detects and reports mismatches

### 4. **Training with Mismatched Config** (`test_training_with_mismatched_config_warns_but_continues`)
- **Purpose**: Ensures training continues despite parameter mismatches
- **Process**: Trains with data having nphotons=1e4 using config with nphotons=1e6
- **Validates**: Training succeeds with warnings (doesn't fail)

### 5. **End-to-End Consistency** (`test_end_to_end_workflow_consistency`)
- **Purpose**: Comprehensive workflow validation with unusual nphotons value
- **Process**: Complete workflow with nphotons=5e5 (unusual value to ensure preservation)
- **Validates**: Value consistency across all workflow stages

### 6. **Backward Compatibility** (`test_metadata_backward_compatibility`)
- **Purpose**: Ensures legacy NPZ files without metadata work correctly
- **Process**: Creates NPZ file without metadata, tests graceful handling
- **Validates**: Legacy files return None metadata with appropriate defaults

## Key Implementation Details

### Metadata System Components Tested

1. **`MetadataManager.create_metadata()`**: Creates metadata from TrainingConfig
2. **`MetadataManager.save_with_metadata()`**: Embeds JSON metadata in NPZ files
3. **`MetadataManager.load_with_metadata()`**: Extracts metadata during data loading
4. **`MetadataManager.validate_parameters()`**: Validates config against metadata
5. **`MetadataManager.get_nphotons()`**: Safely extracts nphotons with defaults

### Workflow Integration Points

1. **Simulation**: `simulate_and_save.py` embeds metadata using `--n-photons` argument
2. **Training**: `train.py` loads and validates metadata using `--nphotons` argument  
3. **Data Loading**: `raw_data.py` automatically loads metadata during data loading
4. **Physics**: `diffsim.py` uses nphotons from legacy params system for scaling

### Critical Test Patterns

- **Subprocess Testing**: Uses actual command-line scripts for realistic testing
- **Temporary Directories**: Isolated test environments with automatic cleanup
- **Real Data Handling**: Uses existing datasets when available, creates minimal test data otherwise
- **Error Condition Testing**: Validates both success and failure scenarios

## Running the Tests

### Run Full Suite
```bash
python -m unittest tests.test_nphotons_metadata_integration -v
```

### Run Individual Tests
```bash
python -m unittest tests.test_nphotons_metadata_integration.TestNphotonsMetadataIntegration.test_metadata_persistence_single_nphotons -v
```

### Expected Runtime
- Full suite: ~2-3 minutes (6 tests with actual training/inference)
- Individual tests: 5-60 seconds depending on complexity

## Test Dependencies

- Working TensorFlow/CUDA setup for training
- Access to simulation and training scripts
- Temporary file system access
- Base dataset or ability to create minimal test data

## Validation Criteria

### Success Indicators
- ✅ All subprocess calls return exit code 0
- ✅ Metadata correctly embedded in simulation outputs
- ✅ Training loads and validates metadata without errors
- ✅ Inference produces expected output files
- ✅ nphotons values preserved exactly through workflow
- ✅ Warnings generated for parameter mismatches
- ✅ Legacy files handled gracefully without metadata

### Failure Indicators  
- ❌ Any subprocess returns non-zero exit code
- ❌ Missing metadata in simulated files
- ❌ Incorrect nphotons values in metadata
- ❌ Missing warnings for parameter mismatches
- ❌ Training fails due to metadata issues
- ❌ Output files not created or too small

## Integration with Existing Tests

This test suite complements the existing test infrastructure:

- **`test_integration_workflow.py`**: Tests basic train→save→load→infer cycle
- **`test_model_manager_persistence.py`**: Tests model serialization and parameter restoration
- **`test_simulate_and_save.py`**: Tests simulation script functionality
- **`test_nphotons_metadata_integration.py`**: Tests nphotons metadata system specifically

## Future Enhancements

Potential extensions for comprehensive coverage:

1. **Multi-dataset Merging**: Test metadata merging for combined datasets
2. **Parameter Override Testing**: Test explicit parameter overrides during inference
3. **Cross-gridsize Validation**: Test metadata consistency across different gridsizes
4. **Performance Impact**: Measure metadata overhead on large datasets
5. **Schema Evolution**: Test metadata version migration and backward compatibility

## Troubleshooting

### Common Issues

1. **CUDA Warnings**: TensorFlow CUDA registration warnings are normal and don't affect tests
2. **Missing Base Data**: Tests create minimal synthetic data if real datasets unavailable
3. **Timeout Errors**: Increase timeout for slow systems or reduce test epochs
4. **Path Issues**: Ensure correct project root path resolution for script locations

### Debug Tips

1. **Verbose Output**: Use `-v` flag to see detailed test progression
2. **Individual Tests**: Run single tests to isolate issues
3. **Temporary Directories**: Check `/tmp/tmp*` directories during test failures
4. **Subprocess Errors**: Examine stderr output in test failure messages
5. **Log Files**: Check `debug.log` files in training output directories

This test suite provides comprehensive validation of the nphotons metadata system, ensuring physics parameter consistency across the entire PtychoPINN workflow.