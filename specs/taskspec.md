# Per-Sample Probe Support Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase to support per-sample probe tensors instead of a global probe variable.

## Mid-Level Objective

- Introduce a new data container that can handle multiple probes and per-sample probe indices.
- Update data loading and preprocessing to assign probe indices to each sample.
- Modify the model architecture to accept per-sample probe inputs based on probe indices.
- Ensure that the training process correctly associates each input sample with its corresponding probe.
- Update testing procedures to handle per-sample probes without shuffling the test samples.

## Implementation Notes

- **Probe Indices**:
  - Introduce a new attribute `probe_indices` in the data container.
  - Ensure `probe_indices` has dtype `int64` and matches the first dimension of `self.X`.

- **Probe List**:
  - Store probes as a list of `tf.Tensor` within the data container.
  - All probes must have the same shape and dtype.

- **Data Merging and Shuffling**:
  - When merging datasets for training, shuffle and interleave samples from multiple datasets.
  - The original dataset boundaries are not preserved.
  - Use `probe_indices` to identify the probe associated with each sample.

- **Testing Data**:
  - Test data handling is similar to training data but without shuffling.
  - The samples should maintain their original order.

- **Model Adjustments**:
  - Update the model inputs to include per-sample probe tensors.
  - Modify layers that apply the probe to utilize per-sample probes based on `probe_indices`.

- **Coding Standards**:
  - Use type hints and function signatures where possible.
  - Maintain code readability and consistency with existing codebase.

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/raw_data.py`
- `./ptycho/workflows/components.py`
- `./ptycho/train_pinn.py`
- `./ptycho/model.py`
- `./ptycho/tf_helper.py`

### Ending Context

- Modified versions of the above files with per-sample probe support.

## Low-Level Tasks
> Ordered from start to finish

1. **Update the Data Container to Support Multiple Probes**

```aider
UPDATE ./ptycho/loader.py:
  - Introduce a new data container class `MultiPtychoDataContainer` that extends `PtychoDataContainer`.
    ```python
    class MultiPtychoDataContainer(PtychoDataContainer):
        def __init__(
            self,
            X: tf.Tensor,
            Y_I: tf.Tensor,
            Y_phi: tf.Tensor,
            norm_Y_I: tf.Tensor,
            YY_full: Optional[tf.Tensor],
            coords_nominal: tf.Tensor,
            coords_true: tf.Tensor,
            nn_indices: np.ndarray,
            global_offsets: np.ndarray,
            local_offsets: np.ndarray,
            probes: List[tf.Tensor],
            probe_indices: tf.Tensor
        ):
            super().__init__(
                X, Y_I, Y_phi, norm_Y_I, YY_full,
                coords_nominal, coords_true,
                nn_indices, global_offsets, local_offsets,
                probeGuess=None  # Omit the global probe
            )
            self.probes = probes  # List of tf.Tensor
            self.probe_indices = probe_indices  # tf.Tensor of dtype int64
    ```
  - Remove usage of `self.probe` in this class, as probes are now per-sample.

UPDATE ./ptycho/raw_data.py:
  - Modify data generation methods to handle multiple probes.
    - When creating instances of `MultiPtychoDataContainer`, assign `probes` and `probe_indices`.
  - Ensure that `probe_indices` are assigned correctly and match the first dimension of `self.X`.
```

2. **Modify Data Loading Functions to Assign Probe Indices**

```aider
UPDATE ./ptycho/workflows/components.py:
  - In the function `create_ptycho_data_container`, adjust to create `MultiPtychoDataContainer` instances.
    ```python
    def create_ptycho_data_container(
        data: Union[RawData, PtychoDataContainer],
        config: TrainingConfig
    ) -> MultiPtychoDataContainer:
        # Existing code...
        # After loading or processing data, create probe indices
        # Assume `data_list` is a list of datasets to merge
        X_list = []
        Y_I_list = []
        Y_phi_list = []
        probe_indices_list = []
        probes = []
        current_probe_index = 0

        for dataset in data_list:
            X_list.append(dataset.X)
            Y_I_list.append(dataset.Y_I)
            Y_phi_list.append(dataset.Y_phi)
            # Assign the same probe index to all samples from this dataset
            probe_indices_list.append(
                tf.fill([tf.shape(dataset.X)[0]], current_probe_index)
            )
            probes.append(dataset.probe)
            current_probe_index += 1

        # Concatenate all datasets
        X = tf.concat(X_list, axis=0)
        Y_I = tf.concat(Y_I_list, axis=0)
        Y_phi = tf.concat(Y_phi_list, axis=0)
        probe_indices = tf.concat(probe_indices_list, axis=0)

        # Shuffle the data
        indices = tf.range(tf.shape(X)[0])
        indices = tf.random.shuffle(indices)
        X = tf.gather(X, indices)
        Y_I = tf.gather(Y_I, indices)
        Y_phi = tf.gather(Y_phi, indices)
        probe_indices = tf.gather(probe_indices, indices)

        return MultiPtychoDataContainer(
            X=X,
            Y_I=Y_I,
            Y_phi=Y_phi,
            norm_Y_I=data.norm_Y_I,
            YY_full=None,
            coords_nominal=data.coords_nominal,
            coords_true=data.coords_true,
            nn_indices=data.nn_indices,
            global_offsets=data.global_offsets,
            local_offsets=data.local_offsets,
            probes=probes,
            probe_indices=probe_indices
        )
    ```
```

3. **Adjust the Model to Accept Per-Sample Probes**

```aider
UPDATE ./ptycho/model.py:
  - Modify the model inputs to include per-sample probe tensors.
    ```python
    # Add a new input to the model
    input_probe = Input(shape=(N, N, 1), name='input_probe', dtype=tf.complex64)
    ```
  - Update the `ProbeIllumination` layer to use `input_probe` instead of `self.w`.
    ```python
    class ProbeIllumination(tf.keras.layers.Layer):
        def call(self, inputs):
            x = inputs[0]          # Input data
            probe = inputs[1]      # Per-sample probe
            illuminated = probe * x
            # Rest of the code remains the same
    ```
  - Adjust model definition to include the new input and pass it to `ProbeIllumination`.
    ```python
    # Modify the call to ProbeIllumination
    padded_objs_with_offsets, _ = probe_illumination([padded_objs_with_offsets, input_probe])

    # Update the model inputs and outputs
    autoencoder = Model(
        [input_img, input_positions, input_probe],
        [trimmed_obj, pred_amp_scaled, pred_intensity_sampled]
    )
    ```
```

4. **Update Training Pipeline to Include Probe Indices**

```aider
UPDATE ./ptycho/train_pinn.py:
  - In the `train` function, modify data preparation to include `probe_indices`.
    ```python
    def prepare_inputs(train_data: MultiPtychoDataContainer):
        # Fetch per-sample probes using probe indices
        probes = tf.gather(train_data.probes, train_data.probe_indices)
        return [train_data.X * cfg.get('intensity_scale'), train_data.coords, probes]
    ```
  - Ensure that the model receives the per-sample probes during training.
```

5. **Modify Helper Functions to Support Multiple Probes**

```aider
UPDATE ./ptycho/tf_helper.py:
  - Update probe-related helper functions to accept per-sample probes.
    - Modify functions that apply probes to data to use per-sample probes from inputs.
    - Ensure functions can handle batches where each sample has a different probe.
```

6. **Ensure Compatibility Across Modules**

```aider
- Review and update any other dependent modules or functions to ensure compatibility with the per-sample probe implementation.

- Add necessary unit tests to validate the new functionality.
  - Test that `MultiPtychoDataContainer` correctly assigns `probe_indices`.
  - Verify that the model correctly applies per-sample probes during training and inference.
  - Ensure that data shuffling and merging work as intended.

- Update documentation to reflect the changes in data handling and model inputs.
```