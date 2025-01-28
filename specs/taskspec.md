# Specification Template
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the existing ptycho codebase to support per-sample probe tensors instead of a single global probe. This enhancement will allow the model to handle diverse probing conditions within a single dataset by associating each sample with its corresponding probe tensor.

## Mid-Level Objective

- Modify the data container classes to support multiple probes and associate each sample with the correct probe index.
- Update data loading and preprocessing pipelines to handle multiple probes and assign probe indices appropriately.
- Refactor the model architecture to utilize per-sample probe tensors based on probe indices.
- Ensure that dataset merging during training correctly interleaves samples from different datasets without preserving original dataset boundaries.
- Maintain consistency in probe tensor shapes and data types across the codebase.
- Ensure that test data handling remains unchanged in terms of probe associations but accommodates the absence of shuffling.

## Implementation Notes

- **Dependencies and Requirements:**
  - Changes in `loader.py` and `raw_data.py` will impact `workflows/components.py`, `train_pinn.py`, `model.py`, and `tf_helper.py`.
  - Ensure that all new attributes, such as `probe_indices`, maintain compatibility with existing data structures.
  - Use `int64` dtype for probe indices to match TensorFlow's requirements.
  - All probes must be stored as `tf.Tensor` and have consistent shapes and dtypes.

- **Coding Standards to Follow:**
  - Adhere to the existing coding style and conventions used in the ptycho codebase.
  - Ensure thorough documentation of new classes and functions introduced.
  - Maintain or enhance test coverage to validate the new functionalities.

- **Other Technical Guidance:**
  - Implement robust indexing and mapping mechanisms to accurately associate samples with their respective probes.
  - Optimize data loading and preprocessing to handle the increased complexity without significant performance degradation.
  - Ensure that model inputs are dynamically linked to the correct probe tensors based on the provided probe indices.

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

## Low-Level Tasks
> Ordered from start to finish

1. **Introduce MultiPtychoDataContainer in `loader.py`**
    ```aider
    UPDATE ./ptycho/loader.py:
        CREATE class `MultiPtychoDataContainer` that inherits from `PtychoDataContainer` and includes:
            - `probes_list: List[tf.Tensor]` to store multiple probe tensors.
            - `probe_indices: tf.Tensor` of dtype `int64` matching the first dimension of `X`, `Y_I`, etc., to specify the probe for each sample.
    ```

2. **Update Data Generation Methods in `raw_data.py`**
    ```aider
    UPDATE ./ptycho/raw_data.py:
        MODIFY existing data generation methods to accept multiple probe tensors.
        ADD logic to assign probe indices to each sample based on the dataset.
        Ensure that all probe tensors have the same shape and dtype.
        RETURN instances of `MultiPtychoDataContainer` instead of `PtychoDataContainer`.
    ```

3. **Refactor Data Loading in `workflows/components.py`**
    ```aider
    UPDATE ./ptycho/workflows/components.py:
        MODIFY the factory functions to create instances of `MultiPtychoDataContainer`.
        ADD logic to shuffle and interleave samples from multiple datasets during training.
        ASSOCIATE each sample with the correct probe index referencing the probes list.
    ```

4. **Adapt Training Functions in `train_pinn.py`**
    ```aider
    UPDATE ./ptycho/train_pinn.py:
        MODIFY training functions to accept `probe_indices` as part of the input data.
        ADJUST the model input pipelines to fetch the correct probe tensor based on `probe_indices` for each sample.
        ENSURE that the training process correctly associates each input sample with its corresponding probe.
    ```

5. **Update Model Architecture in `model.py`**
    ```aider
    UPDATE ./ptycho/model.py:
        MODIFY the model to include probe tensors as per-sample inputs.
        REFRACTOR layers that apply the probe to utilize the per-sample probe tensors instead of a single global probe.
        ENSURE the model can dynamically handle varying probe tensors for each input sample based on `probe_indices`.
    ```

6. **Enhance Probe Handling in `tf_helper.py`**
    ```aider
    UPDATE ./ptycho/tf_helper.py:
        MODIFY probe-related helper functions to accept and process multiple probe tensors.
        ADD functionality to handle fetching the correct probe tensor based on `probe_indices`.
        ENSURE compatibility with the updated model architecture supporting multiple probes.
    ```