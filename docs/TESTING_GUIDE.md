# PtychoPINN Testing Guide

This document provides comprehensive guidance on testing strategies, running the test suite, and contributing new tests to the PtychoPINN project.

## Running the Full Test Suite

To run all automated tests, execute the following command from the project's root directory:

```bash
python -m unittest discover tests/
```

This command will discover and run all test files in the `tests/` directory that follow the `test_*.py` naming convention.

### Alternative: Using pytest

If you prefer pytest (optional dependency), you can run:

```bash
pytest tests/
```

**Important:** Always run tests from the project root directory to ensure proper path resolution and module imports.

## Test Types

The PtychoPINN test suite includes two main categories of tests:

### Unit Tests

Fast, focused tests that validate individual components and functions in isolation.

- **Location:** `tests/test_*.py` files
- **Purpose:** Verify specific functionality of individual modules
- **Examples:**
  - `test_cli_args.py` - Tests command-line argument parsing
  - `test_misc.py` - Tests utility functions
  - `test_model_manager.py` - Tests model management functionality
  - `test_tf_helper.py` - Tests TensorFlow helper functions
- **Execution time:** Typically < 1 second per test
- **Dependencies:** Minimal, often using mocks or small test fixtures

### Integration Tests

Comprehensive tests that validate end-to-end workflows and interactions between components.

- **Location:** `tests/test_integration_*.py` files
- **Purpose:** Ensure complete workflows function correctly across process boundaries
- **Key test:** `test_integration_workflow.py`
  - Validates the complete train → save → load → infer cycle
  - Runs training and inference as separate subprocess calls
  - Verifies model persistence and restoration across processes
  - This is the ultimate check for the model persistence layer
- **Execution time:** Can take several seconds to minutes
- **Dependencies:** Requires actual data files and complete environment setup

## The Critical Integration Test

### test_integration_workflow.py

This test is particularly important as it validates the entire machine learning workflow:

1. **Training Phase:** Trains a model using a subprocess call to `scripts/training/train.py`
2. **Save Phase:** Verifies that the model artifact (`.h5.zip`) is correctly saved
3. **Load Phase:** Loads the saved model in a new process via `scripts/inference/inference.py`
4. **Inference Phase:** Runs inference on test data and generates output visualizations
5. **Validation:** Checks that reconstruction images are created and have valid content

This test ensures that:
- Model serialization and deserialization work correctly
- Training and inference scripts can be used independently
- The saved model format is compatible across different execution contexts
- The complete user workflow functions as expected

## Running Specific Tests

To run a specific test file:
```bash
python -m unittest tests.test_model_manager
```

To run a specific test class:
```bash
python -m unittest tests.test_integration_workflow.TestFullWorkflow
```

To run a specific test method:
```bash
python -m unittest tests.test_integration_workflow.TestFullWorkflow.test_train_save_load_infer_cycle
```

## How to Add New Tests

When contributing new tests to the project, follow these guidelines:

### 1. File Placement

Place all new test files in the top-level `tests/` directory or its subdirectories:
- Unit tests: `tests/test_<module_name>.py`
- Integration tests: `tests/test_integration_<workflow_name>.py`
- Specialized tests: Can be organized in subdirectories like `tests/studies/` or `tests/image/`

### 2. Naming Convention

Follow the `test_*.py` naming convention for all test files to ensure automatic discovery.

### 3. Test Structure

```python
import unittest
import sys
from pathlib import Path

# Add project root to path if needed
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

class TestYourFeature(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures."""
        pass
    
    def tearDown(self):
        """Clean up after tests."""
        pass
    
    def test_specific_functionality(self):
        """Test description."""
        # Arrange
        # Act
        # Assert
        self.assertEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
```

### 4. Best Practices

- Write descriptive test names that explain what is being tested
- Use setUp() and tearDown() for test fixtures
- Clean up temporary files and directories in tearDown()
- Include docstrings explaining the test's purpose
- Use appropriate assertion methods (assertEqual, assertTrue, assertRaises, etc.)
- Keep unit tests fast and focused
- Mock external dependencies when appropriate
- For integration tests, use temporary directories for outputs

## Test Coverage

While we don't enforce strict coverage metrics, aim to test:
- Happy path scenarios
- Edge cases and boundary conditions
- Error handling and exceptions
- Critical workflows and data pipelines

## Continuous Integration

Tests are automatically run on pull requests. Ensure all tests pass before merging changes.

## Troubleshooting

### Common Issues

1. **Import errors:** Ensure you're running tests from the project root
2. **Missing dependencies:** Install test requirements with `pip install -e .[test]` (if available)
3. **Data file not found:** Some tests require the example dataset in `ptycho/datasets/`
4. **Slow tests:** Use `-v` flag for verbose output to identify slow tests

### Getting Help

If you encounter issues with tests:
1. Check test output for specific error messages
2. Ensure your environment matches project requirements
3. Consult the test file's docstrings for specific requirements
4. Open an issue on GitHub if problems persist

## Related Documentation

- [README.md](../README.md) - Project overview and quick start
- [scripts/README.md](../scripts/README.md) - Information about training and inference scripts
- [tests/README_PERSISTENCE_TESTS.md](../tests/README_PERSISTENCE_TESTS.md) - Specific documentation about persistence tests