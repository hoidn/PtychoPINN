# Specification for Per-Sample Probe Integration in Ptycho Reconstruction Codebase (Revised)

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase such that the probe tensor is a per-sample input to the model instead of a global variable, handling probe merging at the PtychoDataContainer level.

## Mid-Level Objectives

- Update PtychoDataContainer to support multiple probes and per-sample probe indices
- Add functionality to merge multiple single-probe datasets at the PtychoDataContainer level
- Adjust the model to accept per-sample probes and apply the correct probe to each sample during training and inference
- Ensure that probes remain fixed (non-trainable) during training in the multi-probe setting
- Maintain backward compatibility with existing single-probe datasets and CLI scripts

## Implementation Notes

- Keep RawData focused on single-probe file loading and representation
- Introduce new attributes in PtychoDataContainer for probe indices and multiple probes
- Handle dataset merging and probe index assignment at PtychoDataContainer level
- Store the list of probes as a `List[tf.Tensor]` in PtychoDataContainer, ensuring all probes have the same shape and dtype
- Limit the number of distinct probes to less than 10
- Original dataset boundaries do not need to be preserved during training; samples should be shuffled and interleaved
- Use probe_indices to keep track of which probe corresponds to each sample
- In the multi-probe setting, probes should remain fixed (non-trainable) during training
- For testing data, samples are not shuffled but are handled similarly to training data

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/workflows/components.py`
- `./ptycho/train_pinn.py`
- `./ptycho/model.py`
- `./ptycho/tf_helper.py`

### Ending Context

- Updated versions of the above files with support for per-sample probes and probe indices
- RawData remains unchanged, focused on single-probe file loading
- Multi-probe functionality implemented at PtychoDataContainer level
- Backward compatibility maintained for existing single-probe datasets and CLI scripts

## Low-Level Tasks

1. **Update PtychoDataContainer to Support Multiple Probes and Probe Indices**

    ```aider
    UPDATE ./ptycho/loader.py:
        MODIFY class PtychoDataContainer:
            ADD attribute `probe_indices: tf.Tensor` with dtype `int64` and shape `[num_samples]`
            ADD attribute `probes: List[tf.Tensor]`
            MODIFY constructor to accept optional `probe_indices` and `probes`
            IF not provided, default to single probe behavior:
                SET `probe_indices` to zeros tensor
                SET `probes` to list containing single probe
            ADD validation to ensure all probes have same shape and dtype
    ```

2. **Add Dataset Merging Functionality to PtychoDataContainer**

    ```aider
    UPDATE ./ptycho/loader.py:
        ADD class method to PtychoDataContainer:
            ADD `merge_datasets(datasets: List[PtychoDataContainer], shuffle: bool = True) -> PtychoDataContainer`:
                VALIDATE that all datasets have compatible shapes and dtypes
                CONCATENATE X, Y_I, coords, and other sample-specific attributes
                COMBINE probes into unified list
                ADJUST probe_indices to point to correct probes in unified list
                IF shuffle is True:
                    SHUFFLE all samples while maintaining probe index associations
                RETURN new PtychoDataContainer with merged data
    ```

3. **Update the Model to Accept Per-Sample Probes Based on Probe Indices**

    ```aider
    UPDATE ./ptycho/model.py:
        MODIFY model inputs and architecture:
            UPDATE model to accept probe_indices input tensor
            MODIFY ProbeIllumination layer:
                UPDATE __init__ to accept list of probes
                MODIFY call method to select correct probe per sample using probe_indices
                ENSURE probes are non-trainable in multi-probe setting
            UPDATE model construction to handle probe_indices input
    ```

4. **Adjust Training Functions to Handle Per-Sample Probes**

    ```aider
    UPDATE ./ptycho/train_pinn.py:
        MODIFY train(...) and train_eval(...) functions:
            UPDATE to handle probe_indices during training
            ENSURE correct probe association per sample
        UPDATE prepare_inputs(train_data: PtychoDataContainer):
            MODIFY to include probe_indices in returned data
    ```

5. **Update Components for Multi-Probe Support**

    ```aider
    UPDATE ./ptycho/workflows/components.py:
        ADD function for merging multiple datasets:
            IMPLEMENT merge_training_data(data_files: List[str]) -> PtychoDataContainer:
                LOAD each file into separate PtychoDataContainer
                USE PtychoDataContainer.merge_datasets to combine
                RETURN merged container
        MODIFY create_ptycho_data_container:
            UPDATE to support both single and multi-probe scenarios
            MAINTAIN backward compatibility
    ```

6. **Update Helper Functions**

    ```aider
    UPDATE ./ptycho/tf_helper.py:
        MODIFY probe-related helper functions:
            UPDATE to handle multiple probes
            ENSURE correct probe selection per sample
            MAINTAIN efficiency in probe application
    ```

7. **Test and Validate**

    ```aider
    ADD smoke test to ensure:
        `python training/train.py --train_data_file Run1084_recon3_postPC_shrunk_3.npz --test_data_file Run1084_recon3_postPC_shrunk_3.npz` works as before
    ```
