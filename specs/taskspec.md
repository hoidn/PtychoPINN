# Specification Template
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase to allow per-sample input for probe tensors instead of a global variable in the ptychography processing pipeline.

## Mid-Level Objective

- Create a new `MultiPtychoDataContainer` class to manage multiple datasets and probes.
- Modify existing data loading functions to support and correctly initialize the new structure.
- Ensure that all training and evaluation processes recognize and utilize the probe indices.

## Implementation Notes

- Follow the existing coding standards and documentation styles across the project.
- Ensure data type consistency, especially ensuring probe indices are stored as `int64`.
- Adapt logic appropriately in dependent files, observing any current assumptions regarding the number and shape of probe tensors.

## Context

### Beginning context
- `../ptycho/loader.py`
- `../ptycho/raw_data.py`
- `../ptycho/workflows/components.py`
- `../ptycho/train_pinn.py`
- `../ptycho/model.py`
- `../ptycho/tf_helper.py`

### Ending context  
- `../ptycho/loader.py` (updated)
- `../ptycho/raw_data.py` (updated)
- `../ptycho/workflows/components.py` (updated)
- `../ptycho/train_pinn.py` (updated)
- `../ptycho/model.py` (updated)
- `../ptycho/tf_helper.py` (updated)

## Low-Level Tasks
> Ordered from start to finish

1. Create a New MultiPtychoDataContainer Class

```aider
CREATE ../ptycho/loader.py:
    DEFINE a new class MultiPtychoDataContainer that includes:
        - a list of probe tensors.
        - an attribute `probe_indices` of dtype int64 to store probe mappings.