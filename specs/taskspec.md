# Implementation of Per-Sample Probe Handling in PtychoDataContainer

> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase to make the probe tensor a per-sample input to the model instead of a global variable.

## Mid-Level Objective

- Create a new data class `MultiPtychoDataContainer` that can hold multiple datasets and a list of probes.
- Modify the `PtychoDataContainer` and `RawData` classes to support multiple probes and probe indices.
- Update the data loading pipeline to handle merging of multiple datasets with their respective probes.
- Modify the model architecture to select the appropriate probe per sample based on probe indices.
- Update the training workflow to handle datasets with multiple probes.

## Implementation Notes

- Each `RawData` instance corresponds to a single probe.
- The `probe_indices` attribute should be added to `PtychoDataContainer` and match the first dimension of `self.X`, `self.Y_I`, etc.
- The `probe_indices` tensor should be of dtype `int64`.
- Original dataset boundaries do not need to be preserved after merging; samples should be shuffled and interleaved.
- The probe index will keep track of which original dataset a sample came from.
- For testing data, samples are not shuffled, and handling should be consistent.
- The `ProbeIllumination` layer in `model.py` should be modified to use per-sample probe selection based on the `probe_indices`.
- Backward compatibility should be maintained where possible.

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/model.py`
- `./ptycho/workflows/components.py`
- `./ptycho/raw_data.py`
- `./ptycho/train_pinn.py`

### Ending Context

- `./ptycho/loader.py` (updated)
- `./ptycho/model.py` (updated)
- `./ptycho/workflows/components.py` (updated)
- `./ptycho/raw_data.py` (updated)
- `./ptycho/train_pinn.py` (updated)

## Low-Level Tasks
> Ordered from start to finish

1. **Create `MultiPtychoDataContainer` Class**

```aider
UPDATE ./ptycho/loader.py:
    CREATE class MultiPtychoDataContainer(PtychoDataContainer):
        def __init__(
            self,
            X: tf.Tensor,
            Y_I: tf.Tensor,
            Y_phi: tf.Tensor,
            norm_Y_I: tf.Tensor,
            YY_full: Optional[tf.Tensor],
            coords_nominal: tf.Tensor,
            coords_true: tf.Tensor,
            nn_indices: tf.Tensor,
            global_offsets: tf.Tensor,
            local_offsets: tf.Tensor,
            probes: List[tf.Tensor],
            probe_indices: tf.Tensor,
        ):
            # Initialize superclass
            super().__init__(
                X, Y_I, Y_phi, norm_Y_I, YY_full,
                coords_nominal, coords_true, nn_indices,
                global_offsets, local_offsets, probeGuess=None
            )
            self.probes = probes  # List of probe tensors
            self.probe_indices = probe_indices  # Tensor of int64 indices matching samples
```

2. **Update `load()` Function to Support Multiple Probes**

```aider
UPDATE ./ptycho/loader.py:
    MODIFY def load(cb: Callable, probeGuess: Union[tf.Tensor, List[tf.Tensor]], which: str, create_split: bool) -> PtychoDataContainer:
        # Check if multiple probes are provided
        if isinstance(probeGuess, list):
            # Process for multiple probes
            # Assign probes and probe_indices appropriately
            # Return MultiPtychoDataContainer instance
        else:
            # Existing single probe handling
            # Return PtychoDataContainer instance
```

3. **Modify `RawData` Class to Include `probe_index` Field**

```aider
UPDATE ./ptycho/raw_data.py:
    ADD attribute to class RawData:
        self.probe_index: Optional[int] = None  # Index of the probe for this dataset

    # Update __init__ method to accept probe_index
    def __init__(
        self,
        xcoords: np.ndarray,
        ycoords: np.ndarray,
        xcoords_start: np.ndarray,
        ycoords_start: np.ndarray,
        diff3d: np.ndarray,
        probeGuess: np.ndarray,
        scan_index: np.ndarray,
        objectGuess: Optional[np.ndarray] = None,
        Y: Optional[np.ndarray] = None,
        norm_Y_I: Optional[np.ndarray] = None,
        probe_index: Optional[int] = None,
    ):
        self.probe_index = probe_index
        # Rest of the initialization
```

4. **Update `generate_grouped_data()` to Preserve `probe_index`**

```aider
UPDATE ./ptycho/raw_data.py:
    MODIFY def generate_grouped_data(self, N, K=7, nsamples=1):
        # Ensure that 'probe_index' is included in the returned dataset
        dset['probe_index'] = self.probe_index
        # Rest of the method remains the same
```

5. **Modify `ProbeIllumination` Layer to Use Per-Sample Probe Selection**

```aider
UPDATE ./ptycho/model.py:
    MODIFY class ProbeIllumination(tf.keras.layers.Layer):
        def call(self, inputs):
            x, probe_indices = inputs
            selected_probes = tf.gather(self.probes, probe_indices)
            # Ensure selected_probes has correct shape for multiplication
            illuminated = selected_probes * x
            # Rest of the method remains the same
```

6. **Update Model Inputs to Include `probe_indices`**

```aider
UPDATE ./ptycho/model.py:
    ADD input_probe_indices = Input(shape=(None,), dtype=tf.int64, name='probe_indices')

    # Modify autoencoder inputs
    autoencoder = Model(
        [input_img, input_positions, input_probe_indices],
        [trimmed_obj, pred_amp_scaled, pred_intensity_sampled]
    )

    # Update layers and methods to pass 'probe_indices' where necessary
```

7. **Modify `prepare_inputs()` to Include `probe_indices`**

```aider
UPDATE ./ptycho/train_pinn.py:
    MODIFY def prepare_inputs(train_data: MultiPtychoDataContainer):
        return [
            train_data.X * cfg.get('intensity_scale'),
            train_data.coords,
            train_data.probe_indices
        ]
```

8. **Update `train()` Function to Handle Multiple Probes**

```aider
UPDATE ./ptycho/train_pinn.py:
    MODIFY def train(epochs, trainset: MultiPtychoDataContainer):
        # Ensure that 'probe_indices' is passed through the training loop
        # Update any handling of probes to account for multiple probes
```

9. **Update `create_ptycho_data_container()` to Handle Multiple Datasets and Probes**

```aider
UPDATE ./ptycho/workflows/components.py:
    MODIFY def create_ptycho_data_container(
        data: Union[RawData, PtychoDataContainer],
        config: TrainingConfig
    ) -> PtychoDataContainer:
        if isinstance(data, list):
            # Handle merging multiple RawData instances into a MultiPtychoDataContainer
            data_containers = [
                create_ptycho_data_container(single_data, config)
                for single_data in data
            ]
            merged_container = MultiPtychoDataContainer.merge(data_containers)
            return merged_container
        else:
            # Existing single data handling
```

10. **Ensure Backward Compatibility and Add Unit Tests**

```aider
VERIFY all existing unit tests pass with the new changes.

ADD new unit tests in ./tests/ to cover scenarios with multiple probes:
    - Test loading multiple datasets and merging them
    - Test that 'probe_indices' correctly select the appropriate probe
    - Test the modified model with multiple probes during training and inference
```