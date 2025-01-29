# Multi-Probe Ptychography Data Handling Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase so that the probe tensor is a per-sample input to the model instead of a global variable.
- Create a new class `MultiPtychoDataContainer` that contains multiple datasets and stores a list of probes.
- For each sample, include an index into the list of probes specifying which probe to use.

## Mid-Level Objectives

- Define the `MultiPtychoDataContainer` class with appropriate attributes to handle multiple datasets and probes.
- Modify data loading functions to generate `MultiPtychoDataContainer` instances by combining multiple `PtychoDataContainer` instances.
- Update data preprocessing to handle multiple probes per sample and shuffle samples for training data.
- Modify the model architecture to accept per-sample probe tensors instead of a global probe.
- Update training and evaluation code to handle per-sample probes correctly.
- Ensure that testing data is handled appropriately without shuffling.

## Implementation Notes

- **Probe Indices**:
  - Add a new attribute `self.probe_indices` to `MultiPtychoDataContainer`, matching the first dimension of `self.X`.
  - Probe indices should be of dtype `tf.int64`.
- **Probe List**:
  - Store a list of probes in `self.probes_list`, which is a list of `tf.Tensor`.
  - All probes must have the same shape and dtype.
- **Data Shuffling**:
  - Training data should be shuffled to interleave samples from different datasets.
  - Testing data should not be shuffled.
- **Dependencies and Requirements**:
  - Update all modules that use `PtychoDataContainer` to handle `MultiPtychoDataContainer`.
  - Ensure that functions and methods handle per-sample probes correctly.
- **Coding Standards**:
  - Use type hints throughout the codebase.
  - Ensure consistency in data types and tensor shapes.
  - Follow existing code style and conventions.
- **Other Technical Guidance**:
  - Ensure backward compatibility where possible.
  - Write unit tests for new functionalities.

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/workflows/components.py`
- `./ptycho/data_preprocessing.py`
- `./ptycho/train_pinn.py`
- `./ptycho/model.py`
- `./ptycho/tf_helper.py`
- `./ptycho/params.py`
- `./ptycho/evaluation.py`

### Ending Context

- `./ptycho/loader.py` (updated)
- `./ptycho/workflows/components.py` (updated)
- `./ptycho/data_preprocessing.py` (updated)
- `./ptycho/train_pinn.py` (updated)
- `./ptycho/model.py` (updated)
- `./ptycho/tf_helper.py` (updated)
- `./ptycho/params.py` (updated)
- `./ptycho/evaluation.py` (updated)

## Low-Level Tasks
> Ordered from start to finish

1. **Create the MultiPtychoDataContainer Class**

```aider
CREATE in ./ptycho/loader.py:
    class MultiPtychoDataContainer:
        def __init__(self,
                     X: tf.Tensor,
                     Y_I: tf.Tensor,
                     Y_phi: tf.Tensor,
                     norm_Y_I: Optional[tf.Tensor],
                     YY_full: Any,
                     coords_nominal: tf.Tensor,
                     coords_true: tf.Tensor,
                     nn_indices: Any,
                     global_offsets: Any,
                     local_offsets: Any,
                     probe_indices: tf.Tensor,
                     probes_list: List[tf.Tensor]):
            self.X = X
            self.Y_I = Y_I
            self.Y_phi = Y_phi
            self.norm_Y_I = norm_Y_I
            self.YY_full = YY_full
            self.coords_nominal = coords_nominal
            self.coords_true = coords_true
            self.nn_indices = nn_indices
            self.global_offsets = global_offsets
            self.local_offsets = local_offsets
            self.probe_indices = probe_indices  # dtype tf.int64
            self.probes_list = probes_list      # List of tf.Tensor

    # Implement methods for data manipulation as needed.
```

2. **Modify Data Loading Functions to Generate MultiPtychoDataContainer Instances**

```aider
UPDATE in ./ptycho/loader.py:
    - Implement `def merge_ptycho_data_containers(containers: List[PtychoDataContainer]) -> MultiPtychoDataContainer`:
        - Concatenate data from multiple `PtychoDataContainer` instances.
        - Assign `probe_indices` for each sample indicating which probe to use.
        - Ensure all attributes are combined appropriately.

    - Modify existing loading functions to use the new merging function when multiple datasets are present.

    - Update type hints and function signatures accordingly.
```

3. **Update Data Preprocessing to Handle Multiple Probes per Sample**

```aider
UPDATE in ./ptycho/data_preprocessing.py:
    - Modify functions to accept `MultiPtychoDataContainer`.
    - Update shuffling functions to shuffle `X`, `Y_I`, `Y_phi`, `coords_nominal`, `coords_true`, and `probe_indices` together.
    - Ensure that probe indices remain correctly associated with their samples after shuffling.
    - Adjust normalization routines if necessary to handle multiple probes.
    - Update type hints and function signatures accordingly.
```

4. **Modify the Model to Accept Per-Sample Probe Tensors**

```aider
UPDATE in ./ptycho/model.py:
    - Add a new input layer for per-sample probes:
        input_probe = Input(shape=(N, N, 1), name='input_probe')

    - Modify `ProbeIllumination` layer:
        class ProbeIllumination(tf.keras.layers.Layer):
            def call(self, inputs):
                x = inputs[0]         # Object patches
                probe = inputs[1]     # Per-sample probes
                # Use per-sample probes instead of self.w
                illuminated = probe * x
                # Rest of the logic remains the same

    - Update model definition to include the new input:
        autoencoder = Model([input_img, input_positions, input_probe], [trimmed_obj, pred_amp_scaled, pred_intensity_sampled])

    - Ensure that the forward pass uses the per-sample probes correctly.

    - Update type hints and function signatures accordingly.
```

5. **Update Training Code to Handle Per-Sample Probes**

```aider
UPDATE in ./ptycho/train_pinn.py:
    - Modify `prepare_inputs` function:
        def prepare_inputs(train_data: MultiPtychoDataContainer):
            # Retrieve per-sample probes using probe_indices
            probes = tf.gather(train_data.probes_list, train_data.probe_indices)
            return [train_data.X * cfg.get('intensity_scale'), train_data.coords, probes]

    - Ensure that the training loop passes per-sample probes to the model.

    - Adjust any batch generation or data pipeline code to include probe indices.

    - Update type hints and function signatures accordingly.
```

6. **Update Utility Functions to Handle Per-Sample Probes**

```aider
UPDATE in ./ptycho/tf_helper.py:
    - Refactor functions that use the global probe to accept probe tensors as parameters.

    - Example: Modify probe-related functions to accept `probe: tf.Tensor` as an argument.

    - Ensure compatibility with per-sample processing throughout the utility functions.

    - Update type hints and function signatures accordingly.
```

7. **Update Configuration to Support Multiple Probes**

```aider
UPDATE in ./ptycho/params.py:
    - Add new configuration parameters if needed to handle multiple probes.

    - Modify existing parameters related to probes to accommodate lists or per-sample data.

    - Ensure that the `validate()` method accounts for the new settings.

    - Update type hints and function signatures accordingly.
```

8. **Adjust Evaluation and Visualization Functions**

```aider
UPDATE in ./ptycho/evaluation.py:
    - Modify evaluation functions to retrieve the correct probe for each sample using `probe_indices`.

    - Update metrics calculations to consider per-sample probes.

    - Ensure consistency in handling probes during evaluation.

    - Update type hints and function signatures accordingly.

UPDATE in ./ptycho/visualize_results.py:
    - Adjust visualization functions to use per-sample probes.

    - Remove any dependence on a global probe variable.

    - Update plotting functions to accurately reflect per-sample probe effects.

    - Update type hints and function signatures accordingly.
```
