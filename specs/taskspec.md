# Adapt Probe Handling to Per-Sample Inputs Specification

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase such that the probe tensor is a per-sample input to the model instead of a global variable.

## Mid-Level Objective

- Modify `PtychoDataContainer` to support storing multiple probes and per-sample probe indices.
- Update data loading functions to handle multiple probes and probe indices.
- Modify the model to use per-sample probes selected via probe indices.
- Ensure the training and evaluation pipelines handle datasets with multiple probes correctly.
- Maintain compatibility with existing functionality where a single global probe is used.

## Implementation Notes

- Add a `probe_indices` attribute to `PtychoDataContainer` with `dtype=tf.int64`, matching the first dimension of `self.X`.
- Store probes as a `tf.Tensor`, all probes having the same shape and `dtype`.
- Update relevant methods to handle the new `probe_indices` and list of probes.
- In the model, the `ProbeIllumination` layer should use the per-sample probe selected using the `probe_indices`.
- Data merging and shuffling logic should accommodate multiple datasets with different probes; original dataset boundaries do not need to be preserved.
- For testing datasets, samples are not shuffled, but the handling of per-sample probes is the same as for training sets.

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/raw_data.py`
- `./ptycho/model.py`
- `./ptycho/workflows/components.py`
- `./ptycho/train_pinn.py`

### Ending Context

- `./ptycho/loader.py` (updated)
- `./ptycho/raw_data.py` (updated)
- `./ptycho/model.py` (updated)
- `./ptycho/workflows/components.py` (updated)
- `./ptycho/train_pinn.py` (updated)

## Low-Level Tasks

> Ordered from start to finish

1. Modify `PtychoDataContainer` in `loader.py` to handle per-sample probes.

    ```aider
    UPDATE ./ptycho/loader.py:

    - In `PtychoDataContainer` class:
        - ADD attribute `self.probe_indices: tf.Tensor` of `dtype=tf.int64`, shape matching `self.X`'s first dimension.
        - MODIFY `__init__` to accept `probe_indices: tf.Tensor`, `probes: tf.Tensor`.
        - ADD attribute `self.probes: tf.Tensor` containing the list of probes.
        - ENSURE that `self.probes` has shape `(num_probes, N, N)`.
    - UPDATE methods that utilize `self.probe` to accommodate `self.probes` and `self.probe_indices`.
        - REPLACE any usage of `self.probe` with the appropriate selection from `self.probes` using `self.probe_indices`.

    ```

2. Update data loading functions in `loader.py` to handle probe indices.

    ```aider
    UPDATE ./ptycho/loader.py:

    - In `from_raw_data_without_pc()` and other relevant methods:
        - ACCEPT an additional parameter `probe_index: np.ndarray` or `tf.Tensor`.
        - PASS `probe_indices` when creating `PtychoDataContainer` instances.
        - ENSURE that `probe_indices` is correctly assigned and matches the samples in `self.X`.

    ```

3. Update `RawData` class in `raw_data.py` to include probe indices.

    ```aider
    UPDATE ./ptycho/raw_data.py:

    - In `RawData` class:
        - ADD attribute `self.probe_index: np.ndarray` of `dtype=np.int64`, matching the length of `self.xcoords`.
        - INCLUDE `probe_index` in the `__init__` method.
    - ENSURE that `generate_grouped_data()` and other methods preserve `probe_index`.
    - UPDATE any methods that manipulate or split data to handle `probe_index` accordingly.

    ```

4. Modify `ProbeIllumination` layer in `model.py` to use per-sample probes.

    ```aider
    UPDATE ./ptycho/model.py:

    - In `ProbeIllumination` class:
        - REMOVE `self.w` (single probe) and ADD `self.probes`, a `tf.Variable` containing the list of probes.
        - MODIFY `__init__` to accept `probes: tf.Variable`.
        - MODIFY `call()` method to accept an additional input: `probe_indices`.
        - SELECT the appropriate probe for each sample using `tf.gather(self.probes, probe_indices)`.
        - ENSURE that the selected probes have the correct shape for multiplication with the input.
    - UPDATE the model inputs to include `probe_indices`:
        - ADD `probe_indices = Input(shape=(None,), dtype=tf.int64, name='probe_indices')`
    - UPDATE the model to pass `probe_indices` to the `ProbeIllumination` layer.

    ```

5. Update training and evaluation scripts to handle per-sample probes.

    ```aider
    UPDATE ./ptycho/train_pinn.py:

    - In `prepare_inputs()` function:
        - INCLUDE `train_data.probe_indices` in the inputs returned.
    - In `train()` function:
        - MODIFY to accept `probe_indices` and pass them to the model.

    UPDATE ./ptycho/workflows/components.py:

    - In `create_ptycho_data_container()`:
        - PASS `probe_indices` from `RawData` to `PtychoDataContainer`.
    - In `train_cdi_model()`:
        - ENSURE that `probe_indices` are included in the data being passed to the model.

    ```

6. Ensure all functions and methods have appropriate type hints and update any dependent code.

    ```aider
    UPDATE ./ptycho/loader.py, ./ptycho/raw_data.py, ./ptycho/model.py, ./ptycho/train_pinn.py, ./ptycho/workflows/components.py:

    - ADD type hints to all modified functions and methods.
    - ENSURE consistency in data types across the codebase.
    - UPDATE any dependent code to accommodate the changes in data structures.
    - VERIFY that tests and examples run correctly with the updated code.

    ```