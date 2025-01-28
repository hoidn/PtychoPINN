# Specification for Adapting Codebase to Per-Sample Probes

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase to support per-sample probe tensors, allowing each sample to use an individual probe specified by an index, instead of using a global probe tensor.

## Mid-Level Objective

- Introduce a new data container class that handles multiple probes and associates each sample with a probe index.
- Modify data loading and preprocessing functions to assign appropriate probe indices to each sample and handle multiple probes during dataset merging and shuffling.
- Update the model architecture to accept per-sample probes based on probe indices, replacing the use of a global probe variable.
- Ensure that the training and evaluation processes correctly associate each input sample with its corresponding probe.
- Maintain backward compatibility with existing CLI scripts (`train.py`, `inference.py`) so that existing run commands continue to function without changes.

## Implementation Notes

- **Probe Indices:** Add a new attribute `probe_indices` to data containers, matching the first dimension of `self.X`, `self.Y_I`, etc. `probe_indices` should be of dtype `int64`.
- **Probe Storage:** Store probes as a list of `tf.Tensor` objects. All probes must have the same shape and dtype.
- **Data Merging:** When merging datasets for training, interleave and shuffle samples from different datasets without preserving original dataset boundaries. Use probe indices to keep track of probe associations.
- **Model Inputs:** Modify model inputs to include per-sample probes based on probe indices. Ensure efficient retrieval of the correct probe tensor for each sample during training and inference.
- **Fixed Probes:** Probes should remain fixed during training; do not implement probe updates in the multi-probe setting.
- **Backward Compatibility:** Existing CLI scripts should work with existing run commands. If necessary, update default behaviors or parameters to maintain compatibility.

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/raw_data.py`
- `./ptycho/workflows/components.py`
- `./ptycho/train_pinn.py`
- `./ptycho/model.py`
- `./ptycho/tf_helper.py`

### Ending Context

- `./ptycho/loader.py` (updated)
- `./ptycho/raw_data.py` (updated)
- `./ptycho/workflows/components.py` (updated)
- `./ptycho/train_pinn.py` (updated)
- `./ptycho/model.py` (updated)
- `./ptycho/tf_helper.py` (updated)
- Possibly new data container class (e.g., `MultiPtychoDataContainer`)

## Low-Level Tasks

> Ordered from start to finish

1. **Create a New Data Container Class to Handle Multiple Probes**

```aider
UPDATE ./ptycho/loader.py:
- CREATE a new class `MultiPtychoDataContainer` inheriting from `PtychoDataContainer`.
- ADD attributes:
  - `probes: List[tf.Tensor]` to store the list of probe tensors.
  - `probe_indices: tf.Tensor` of dtype `int64`, matching the first dimension of `self.X`.
- MODIFY existing methods to handle the new attributes where necessary.
```

2. **Update Data Generation Methods to Assign Probe Indices**

```aider
UPDATE ./ptycho/raw_data.py:
- MODIFY data generation functions to accept multiple probes.
- IMPLEMENT logic to assign `probe_indices` to each sample during data creation.
- ENSURE that `probe_indices` correctly reference the probes in the `probes` list.
```

3. **Modify Data Loading and Merging Functions**

```aider
UPDATE ./ptycho/loader.py:
- UPDATE functions that load and merge datasets to handle multiple probes.
- IMPLEMENT sample shuffling and interleaving without preserving original dataset boundaries.
- ENSURE that `probe_indices` are correctly assigned and maintained after shuffling.
```

4. **Update the Model to Accept Per-Sample Probes**

```aider
UPDATE ./ptycho/model.py:
- MODIFY the model input to include `probe_indices` and the list of `probes`.
- REPLACE the global probe variable with per-sample probe selection using `tf.gather` or similar methods.
- ENSURE that during the forward pass, each sample uses the correct probe based on its `probe_index`.
- ADJUST the `ProbeIllumination` layer or equivalent to handle per-sample probes.
```

5. **Adjust Helper Functions to Support Multiple Probes**

```aider
UPDATE ./ptycho/tf_helper.py:
- MODIFY probe-related helper functions to accept `probes` and `probe_indices` as inputs.
- ENSURE that functions correctly retrieve and apply the per-sample probe tensors.
- UPDATE any functions that assume a single global probe to handle multiple probes appropriately.
```

6. **Update Training and Evaluation Processes**

```aider
UPDATE ./ptycho/train_pinn.py:
- ADJUST training functions to include `probe_indices` in the input data.
- ENSURE that the model receives and uses `probe_indices` during training and evaluation.
- VERIFY that each input sample is correctly associated with its corresponding probe.
```

7. **Modify Components to Maintain Backward Compatibility**

```aider
UPDATE ./ptycho/workflows/components.py:
- MODIFY the data container creation to instantiate `MultiPtychoDataContainer` when multiple probes are used.
- ADD logic to handle both single-probe and multi-probe datasets, defaulting to single-probe for existing run commands.
- ENSURE that existing CLI scripts (`train.py`, `inference.py`) continue to function without requiring changes from the user.
```

8. **Test the System with Multiple Probes**

```aider
- CREATE unit tests or use existing test scripts to verify the correct functionality with multiple probes.
- ENSURE that samples are correctly associated with their probes during training and inference.
- CONFIRM that the system behaves as expected and that performance is acceptable.
```