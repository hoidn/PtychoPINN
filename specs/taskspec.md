# Specification Template
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase such that the probe tensor is a per-sample input to the model instead of a global variable.

## Mid-Level Objective

- Define a new data container class, `MultiPtychoDataContainer`, that supports multiple datasets and stores a list of probe tensors.
- Add an attribute, `probe_indices`, to the new container to track probe references for each sample.
- Modify existing methods in relevant classes to accommodate this new structure and ensure proper data management.
- Ensure that training processes shuffle samples while maintaining the correct association with their respective probes.

## Implementation Notes
- Use `int64` for probe indices.
- Ensure that all probes maintain the same shape and data type.
- Validate that changes do not break functionality for the testing phase which should not involve sample shuffling.
- Follow the existing coding standards and practices within the codebase.

## Context

### Beginning Context
- `./ptycho/loader.py`
- `./ptycho/raw_data.py`
- `./ptycho/workflows/components.py`
- `./ptycho/train_pinn.py`
- `./ptycho/model.py`
- `./ptycho/tf_helper.py`

### Ending Context
- `./ptycho/loader.py` (updated for MultiPtychoDataContainer)
- `./ptycho/raw_data.py` (updated for multiple probe support)
- `./ptycho/workflows/components.py` (adapted for MultiPtychoDataContainer)
- `./ptycho/train_pinn.py` (modified for probe indexing)
- `./ptycho/model.py` (updated for per-sample probes)
- `./ptycho/tf_helper.py` (updated for multiple probe handling)

## Low-Level Tasks
> Ordered from start to finish

1. Create `MultiPtychoDataContainer` Class in `loader.py`

```aider
UPDATE ./ptycho/loader.py:
    CREATE class MultiPtychoDataContainer which inherits from PtychoDataContainer.
        MODIFY initialization to store multiple probes and add probe_indices.
```

2. Modify Data Loading Methods for `MultiPtychoDataContainer`

```aider
UPDATE ./ptycho/raw_data.py:
    MODIFY methods to generate and fill `MultiPtychoDataContainer` instances.
    ENSURE assignment of probe indices aligns with samples.
```

3. Update Factory Function in `workflows/components.py`

```aider
UPDATE ./ptycho/workflows/components.py:
    MODIFY to use `MultiPtychoDataContainer` for loading datasets.
    IMPLEMENT shuffling logic to interleave samples correctly.
```

4. Adapt Training Functions in `train_pinn.py`

```aider
UPDATE ./ptycho/train_pinn.py:
    ADJUST input handling to include probe indices during model training.
```

5. Update Model Definition in `model.py`

```aider
UPDATE ./ptycho/model.py:
    MODIFY inputs and layers to access probe tensors via indices.
```

6. Update Probe Handling in `tf_helper.py`

```aider
UPDATE ./ptycho/tf_helper.py:
    MODIFY probe operations to accommodate per-sample probe inputs.
```