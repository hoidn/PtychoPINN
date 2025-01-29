# Specification: Adapt Codebase for Per-Sample Probe Handling

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase to support per-sample probes in the ptychography model, replacing the current global probe assumption.

## Mid-Level Objectives

- Update `PtychoDataContainer` to store multiple probes and per-sample probe indices.
- Modify data loading and merging processes to handle datasets with multiple probes.
- Update the model architecture to accept per-sample probes and use the probe index for probe selection.
- Ensure training and inference workflows accommodate per-sample probe handling without breaking existing functionalities.
- Maintain consistency and compatibility across the codebase, updating dependencies as required.

## Implementation Notes

- **Probe Indices**: Implement a new attribute `probe_indices` in `PtychoDataContainer` matching the first dimension (batch size) of `self.X`, `self.Y_I`, etc., with data type `int64`.
  
- **Probe List**: Store multiple probes as a list or tensor of `tf.Tensor`, ensuring all probes have the same shape and data type.

- **Data Merging**: When merging datasets for training, shuffle samples to interleave datasets, and assign appropriate `probe_indices`.

- **Model Inputs**: Modify the model to accept per-sample probes and probe indices as inputs. Update the `ProbeIllumination` layer to select the appropriate probe for each sample based on `probe_indices`.

- **Compatibility**: Ensure that the test set handling remains consistent with minimal changes, primarily not shuffling test samples.

- **Dependencies and Impact**: Update dependent modules, including `model.py`, `train_pinn.py`, `components.py`, `raw_data.py`, and others affected by the changes.

- **Type Hints**: Include type hints in function signatures to aid clarity and maintain consistency.

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/model.py`
- `./ptycho/components.py`
- `./ptycho/raw_data.py`
- `./ptycho/train_pinn.py`

### Ending Context

- Updated versions of:
  - `./ptycho/loader.py`
  - `./ptycho/model.py`
  - `./ptycho/components.py`
  - `./ptycho/raw_data.py`
  - `./ptycho/train_pinn.py`

## Low-Level Tasks

> Ordered from start to finish

1. **Update `PtychoDataContainer` to Support Multiple Probes**

```aider
UPDATE ./ptycho/loader.py:
  - Modify `PtychoDataContainer` class:
    - ADD attribute `probe_indices: tf.Tensor` of dtype `int64`, shape matching `self.X`
    - MODIFY constructor to accept `probe_list: List[tf.Tensor]` and `probe_indices: tf.Tensor`
    - UPDATE methods to handle per-sample probes using `probe_indices`
```

2. **Modify Data Loading Functions to Assign Probe Indices**

```aider
UPDATE ./ptycho/raw_data.py:
  - MODIFY `RawData` class:
    - ADD attribute `probe_index: int64` (since each `RawData` corresponds to a single probe)
  - UPDATE `generate_grouped_data()` method:
    - Ensure `probe_index` is propagated and expanded to match grouped data
  - UPDATE any methods involved in merging datasets to handle `probe_indices`
```

3. **Implement Dataset Merging with Probe Handling**

```aider
UPDATE ./ptycho/loader.py:
  - ADD method `def merge_ptycho_data_containers(containers: List[PtychoDataContainer]) -> PtychoDataContainer`:
    - SHUFFLE samples to interleave datasets
    - MERGE data attributes (`X`, `Y_I`, etc.) from input containers
    - CONCATENATE `probe_list` from containers
    - ASSIGN `probe_indices` to each sample, indicating which probe to use
```

4. **Update the Model to Accept Per-Sample Probes**

```aider
UPDATE ./ptycho/model.py:
  - Modify model inputs:
    - ADD input tensor `probe_list: tf.Tensor` of shape `(num_probes, ...)`
    - ADD input tensor `probe_indices: tf.Tensor` of dtype `int64`, shape `(batch_size,)`
  - Modify `ProbeIllumination` layer:
    - UPDATE to select per-sample probe based on `probe_indices`
    - IMPLEMENT selection logic within `call()` method
    - Example signature: `def call(self, inputs: Tuple[tf.Tensor, tf.Tensor, tf.Tensor]) -> tf.Tensor`
      - Inputs are `(padded_objs_with_offsets, probe_list, probe_indices)`
```

5. **Adjust Training Workflow to Include Probe Indices**

```aider
UPDATE ./ptycho/train_pinn.py:
  - MODIFY `prepare_inputs(train_data: PtychoDataContainer)`:
    - RETURN `[train_data.X * cfg.get('intensity_scale'), train_data.coords, train_data.probe_indices]`
  - MODIFY `train()` function:
    - ENSURE `probe_list` and `probe_indices` are passed to the model during training
    - UPDATE any relevant parts to handle the new model inputs
```

6. **Update Components for Data Preparation**

```aider
UPDATE ./ptycho/components.py:
  - MODIFY data preparation functions to handle `probe_indices`
  - ENSURE that when creating `PtychoDataContainer`, `probe_indices` and `probe_list` are correctly assigned
```

7. **Ensure Consistency and Update Dependencies**

```aider
UPDATE dependencies:
  - CHECK and UPDATE any other modules or functions that rely on global probe assumptions
  - ENSURE unit tests and validation scripts are updated to accommodate per-sample probes
```