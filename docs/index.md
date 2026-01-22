# PtychoPINN Documentation Hub

This hub is generated from the current branch state. It is intentionally concise and points to the
most relevant documents for training, inference, data handling, and debugging.

## Quick Start

- [Project README](../README.md)
- [Commands Reference](COMMANDS_REFERENCE.md)
- [Workflow Guide](WORKFLOW_GUIDE.md)

## Critical Gotchas

- `ptycho/config/config.py` uses dataclass configs, but core code still reads `ptycho/params.cfg`.
  If you use `scripts/training/train.py`, it calls `update_legacy_dict()` to sync values.
- Data files must match the configured `N` and `gridsize`. Mismatches cause silent shape errors
  in the loader and model.
- Most scripts expect `.npz` datasets with specific keys. See [Data Contracts](../specs/data_contracts.md).

## Architecture

- [Architecture Overview](architecture.md)
- [TensorFlow Architecture](architecture_tf.md)
- [Inference Pipeline](architecture_inference.md)
- [PyTorch Architecture](architecture_torch.md)

## Data and Configuration

- [Configuration](CONFIGURATION.md)
- [Data Generation](DATA_GENERATION_GUIDE.md)
- [Data Management](DATA_MANAGEMENT_GUIDE.md)
- [Data Normalization](DATA_NORMALIZATION_GUIDE.md)
- [Grid and Grouping](GRIDSIZE_N_GROUPS_GUIDE.md)

## Workflows

- [Workflow Guide](WORKFLOW_GUIDE.md)
- [Model Comparison](MODEL_COMPARISON_GUIDE.md)
- [PyTorch Workflow](workflows/pytorch.md)

## Testing and Debugging

- [Testing Guide](TESTING_GUIDE.md)
- [Test Suite Index](development/TEST_SUITE_INDEX.md)
- [Debugging Guide](debugging/debugging.md)
- [Troubleshooting](debugging/TROUBLESHOOTING.md)
- [Quick Reference: Params](debugging/QUICK_REFERENCE_PARAMS.md)

## Specifications

- [Specs Directory](../specs)
