# Specification for Per-Sample Probe Integration in Ptycho Reconstruction Codebase

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase such that the probe tensor is a per-sample input to the model instead of a global variable.

## Mid-Level Objectives

- Update data structures to support multiple probes and per-sample probe indices.
- Modify data loading functions to assign probe indices to each sample and merge multiple datasets for training by interleaving samples.
- Adjust the model to accept per-sample probes and apply the correct probe to each sample during training and inference.
- Ensure that probes remain fixed (non-trainable) during training in the multi-probe setting.
- Maintain backward compatibility with existing single-probe datasets and CLI scripts.

## Implementation Notes

- Introduce a new attribute `probe_indices` in `PtychoDataContainer`, matching the first dimension of `X`, `Y_I`, etc., with dtype `int64`.
- Store the list of probes as a `List[tf.Tensor]`, ensuring all probes have the same shape and dtype.
- Limit the number of distinct probes to less than 10.
- Original dataset boundaries do not need to be preserved during training; samples should be shuffled and interleaved.
- Use `probe_indices` to keep track of which probe corresponds to each sample.
- In the multi-probe setting, probes should remain fixed (non-trainable) during training.
- For testing data, samples are not shuffled but are handled similarly to training data.
- Ensure backward compatibility by updating the model API in `components.py` so existing CLI scripts (`train.py`, `inference.py`) continue to work without changes to run commands.

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/raw_data.py`
- `./ptycho/workflows/components.py`
- `./ptycho/train_pinn.py`
- `./ptycho/model.py`
- `./ptycho/tf_helper.py`

### Ending Context

- Updated versions of the above files with support for per-sample probes and probe indices.
- Data structures and models that handle multiple probes and associate each sample with the correct probe based on `probe_indices`.
- Backward compatibility maintained for existing single-probe datasets and CLI scripts.

## Low-Level Tasks

> Ordered from start to finish.

1. **Update `PtychoDataContainer` to Support Multiple Probes and Probe Indices**

    ```aider
    UPDATE ./ptycho/loader.py:
        MODIFY class PtychoDataContainer:
            ADD attribute `probe_indices: tf.Tensor` with dtype `int64` and shape `[num_samples]`, matching the first dimension of `self.X`.
            MODIFY constructor to accept `probe_indices` and assign it to `self.probe_indices`.
            ADD attribute `probes: List[tf.Tensor]`, where each probe tensor has the same shape and dtype as the existing probe.
            MODIFY methods to handle the list of probes and probe indices where necessary.
    ```

2. **Modify Data Loading Functions to Assign Probe Indices and Handle Multiple Probes**

    ```aider
    UPDATE ./ptycho/raw_data.py:
        MODIFY RawData class methods:
            UPDATE `generate_grouped_data(...)`:
                IMPLEMENT logic to assign probe indices to each sample during data generation.
                ENSURE `probe_indices` is a `tf.Tensor` of dtype `int64` with the same length as the number of samples.
            UPDATE methods that create `PtychoDataContainer` instances to include `probe_indices` and `probes`.
    ```

3. **Adjust Data Merging for Training to Interleave Samples and Assign Probe Indices**

    ```aider
    UPDATE ./ptycho/loader.py:
        ADD function `merge_datasets_for_training(datasets: List[PtychoDataContainer]) -> PtychoDataContainer`:
            IMPLEMENT logic to merge multiple `PtychoDataContainer` instances for training.
            SHUFFLE and interleave samples from multiple datasets.
            CONCATENATE `X`, `Y_I`, `coords`, and other sample-specific attributes.
            CONCATENATE `probe_indices`, ensuring each sample points to the correct probe in the `probes` list.
            COMBINE the `probes` from all datasets into a single list, ensuring no duplicates if probes are identical.
    ```

4. **Update the Model to Accept Per-Sample Probes Based on Probe Indices**

    ```aider
    UPDATE ./ptycho/model.py:
        MODIFY model inputs and architecture:
            UPDATE the model to accept an additional input `probe_indices: tf.Tensor` of dtype `int64` and shape `[batch_size]`.
            MODIFY the `ProbeIllumination` layer to:
                ACCEPT a list of probes `probes: List[tf.Tensor]`.
                FETCH the correct probe for each sample based on `probe_indices` during the forward pass.
                APPLY the per-sample probe to the corresponding sample.
            ENSURE that the probes are treated as non-trainable (fixed) variables in the multi-probe setting.
    ```

5. **Adjust Training Functions to Handle Per-Sample Probes and Probe Indices**

    ```aider
    UPDATE ./ptycho/train_pinn.py:
        MODIFY `train(...)` and `train_eval(...)` functions:
            UPDATE training data preparation to include `probe_indices`.
            PASS `probe_indices` to the model during training.
            ENSURE that the correct probe is associated with each sample based on `probe_indices`.
        MODIFY `prepare_inputs(train_data: PtychoDataContainer)`:
            UPDATE to return `[train_data.X * cfg.get('intensity_scale'), train_data.coords, train_data.probe_indices]`.
    ```

6. **Update Helper Functions to Support Per-Sample Probes**

    ```aider
    UPDATE ./ptycho/tf_helper.py:
        MODIFY probe-related helper functions to accept per-sample probes:
            UPDATE functions like `probe_illumination(...)` to accept `probes` and `probe_indices`.
            ENSURE that the correct probe is applied to each sample during operations.
    ```

7. **Modify Components to Ensure Backward Compatibility and Updated Model API**

    ```aider
    UPDATE ./ptycho/workflows/components.py:
        MODIFY `create_ptycho_data_container(...)`:
            UPDATE to handle both single-probe and multi-probe datasets.
            IF only one probe is provided, SET `probe_indices` to zeros tensor of int64.
            ENSURE backward compatibility with existing datasets and CLI scripts.
        MODIFY functions that prepare data for training and testing to include `probe_indices`.
    ```

8. **Ensure Probes Remain Fixed During Training in Multi-Probe Setting**

    ```aider
    UPDATE ./ptycho/model.py:
        MODIFY the model to set probes as non-trainable variables when multiple probes are used:
            DETERMINE if multiple probes are provided.
            IF multiple probes, ENSURE that probe variables are set with `trainable=False`.
            UPDATE any relevant model layers or variables accordingly.
    ```

9. **Test and Validate the Updated Codebase**

    ```aider
    PERFORM testing to ensure:
        MULTI-PROBE training works correctly with per-sample probes.
        PROBES are correctly assigned to samples using `probe_indices`.
        PROBES remain fixed (non-trainable) during training in multi-probe setting.
        BACKWARD COMPATIBILITY is maintained for single-probe datasets and existing CLI scripts.
        TEST data is handled correctly, with samples not shuffled but otherwise processed similarly.
    ```