# Modify Ptychography Codebase for Per-Sample Probes Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

- Adapt the codebase to support per-sample probes instead of a single global probe by introducing a `MultiPtychoDataContainer` class and updating the data pipeline and model to handle per-sample probes.

## Mid-Level Objectives

- Create a `MultiPtychoDataContainer` class to handle multiple datasets and store a list of probes along with probe indices for each sample.
- Update data loading and preparation code to handle multiple probes and assign correct probe indices to samples.
- Modify the model to accept per-sample probe inputs and use probe indices to select the appropriate probe for each sample during training and inference.
- Update the training workflow to handle datasets with multiple probes, including shuffling samples during training while preserving probe indices.
- Ensure probe indices are correctly propagated through the data pipeline to the model.

## Implementation Notes

- The `probe_indices` attribute should be added to data containers, matching the size of `self.X`, `self.Y_I`, etc.
- `probe_indices` should be of dtype `int64`.
- The `probe_list` should be a list of `tf.Tensor` objects.
- All probes in `probe_list` must have the same shape and dtype.
- For testing, samples are not shuffled, and probe indices should be handled appropriately.
- Create a new `MultiPtychoDataContainer` class without subclassing `PtychoDataContainer`.
- Ensure backward compatibility where possible and maintain coding standards throughout the codebase.

## Context

### Beginning Context

- `./ptycho/loader.py`
- `./ptycho/model.py`
- `./ptycho/workflows/components.py`
- `./ptycho/raw_data.py`
- `./ptycho/train_pinn.py`

**Note**: The code in these files is as provided in the initial context.

### Ending Context

- `./ptycho/loader.py` (updated)
- `./ptycho/model.py` (updated)
- `./ptycho/workflows/components.py` (updated)
- `./ptycho/raw_data.py` (updated)
- `./ptycho/train_pinn.py` (updated)

## Low-Level Tasks
> Ordered from start to finish

1. **Create `MultiPtychoDataContainer` Class in `loader.py`**

```aider
CREATE in `./ptycho/loader.py`:

    class MultiPtychoDataContainer:
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
            probe_list: List[tf.Tensor],
            probe_indices: tf.Tensor
        ):
            """
            Initialize a container for multiple ptychography datasets with per-sample probes.

            Args:
                X: Diffraction patterns tensor.
                Y_I: Intensity tensor.
                Y_phi: Phase tensor.
                norm_Y_I: Normalization factors for Y_I.
                YY_full: Full field data (if available).
                coords_nominal: Nominal coordinates.
                coords_true: True coordinates.
                nn_indices: Nearest neighbor indices.
                global_offsets: Global offsets.
                local_offsets: Local offsets.
                probe_list: List of probe tensors.
                probe_indices: Tensor of probe indices per sample (dtype int64).
            """
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
            self.probe_list = probe_list  # List of tf.Tensor probes
            self.probe_indices = probe_indices  # Tensor of int64 indices matching self.X's first dimension

        def __repr__(self):
            # Implement a representation method similar to PtychoDataContainer
            pass

    # Implement methods for merging multiple PtychoDataContainer instances, shuffling samples, and assigning probe indices.
```

2. **Update Data Loading Functions to Utilize `MultiPtychoDataContainer`**

```aider
UPDATE `./ptycho/loader.py`:

    ADD a function `merge_ptycho_data_containers` that takes a list of `PtychoDataContainer` instances and returns a `MultiPtychoDataContainer`.

    def merge_ptycho_data_containers(
        data_containers: List[PtychoDataContainer],
        shuffle: bool = True
    ) -> MultiPtychoDataContainer:
        """
        Merge multiple PtychoDataContainer instances into a MultiPtychoDataContainer.

        Args:
            data_containers: List of PtychoDataContainer instances to merge.
            shuffle: Whether to shuffle the samples after merging.

        Returns:
            A MultiPtychoDataContainer instance containing merged data.
        """
        # Implement merging logic, combining datasets and creating probe_list and probe_indices.
```

3. **Modify `ProbeIllumination` Layer in `model.py` to Use Per-Sample Probes**

```aider
UPDATE `./ptycho/model.py`:

    class ProbeIllumination(tf.keras.layers.Layer):
        def __init__(self, name=None):
            super(ProbeIllumination, self).__init__(name=name)
            self.probe_list = None  # List of probes
            self.sigma = cfg.get('gaussian_smoothing_sigma')

        def build(self, input_shape):
            # Assuming probe_list is passed during model building
            pass

        def call(self, inputs):
            x, probe_indices = inputs
            # Select the appropriate probe for each sample using probe_indices
            probes = tf.gather(self.probe_list, probe_indices)
            # Apply probe to the input x
            illuminated = probes * x
            # Apply Gaussian smoothing if required
            # Return the illuminated samples and any additional outputs as needed
```

4. **Update the Model to Accept `probe_indices` as Input**

```aider
UPDATE `./ptycho/model.py`:

    # Modify model inputs to include probe_indices
    input_img = Input(shape=(N, N, gridsize**2), name='input_img')
    input_positions = Input(shape=(1, 2, gridsize**2), name='input_positions')
    input_probe_indices = Input(shape=(), dtype=tf.int64, name='probe_indices')  # Scalar per sample

    # Update data flow to pass probe_indices to ProbeIllumination layer
    # Update the model's construction to include the new input
```

5. **Adjust the Data Preparation Functions to Include Probe Indices**

```aider
UPDATE `./ptycho/train_pinn.py`:

    def prepare_inputs(train_data: MultiPtychoDataContainer):
        """Prepare training inputs including probe indices."""
        return [
            train_data.X * cfg.get('intensity_scale'),
            train_data.coords_nominal,
            train_data.probe_indices  # Include probe indices in inputs
        ]

    # Update any other functions that rely on prepare_inputs
```

6. **Modify Training Workflow to Handle `MultiPtychoDataContainer`**

```aider
UPDATE `./ptycho/train_pinn.py`:

    def train(
        train_data: MultiPtychoDataContainer,
        intensity_scale=None,
        model_instance=None
    ):
        # Adjust the function to handle MultiPtychoDataContainer
        # Ensure probe_list is correctly passed to the model
        # Update any references to probe handling
```

7. **Update Data Loading and Preparation in `components.py`**

```aider
UPDATE `./ptycho/workflows/components.py`:

    def create_ptycho_data_container(
        data: Union[RawData, PtychoDataContainer],
        config: TrainingConfig
    ) -> Union[PtychoDataContainer, MultiPtychoDataContainer]:
        """
        Update to handle RawData instances that may correspond to multiple probes.
        Return MultiPtychoDataContainer when needed.
        """
        # Logic to check if data corresponds to multiple probes
        # If multiple probes, generate a MultiPtychoDataContainer
        # Else, return a standard PtychoDataContainer

    def train_cdi_model(
        train_data: Union[RawData, PtychoDataContainer, MultiPtychoDataContainer],
        test_data: Optional[Union[RawData, PtychoDataContainer, MultiPtychoDataContainer]],
        config: TrainingConfig
    ) -> Dict[str, Any]:
        """
        Update to handle MultiPtychoDataContainer instances.
        Ensure that probe indices and probe lists are passed throughout the training process.
        """
        # Adjust code to work with MultiPtychoDataContainer
```

8. **Add `probe_indices` Handling in `raw_data.py`**

```aider
UPDATE `./ptycho/raw_data.py`:

    class RawData:
        # Add a new attribute for probe_index
        def __init__(
            self,
            xcoords,
            ycoords,
            xcoords_start,
            ycoords_start,
            diff3d,
            probeGuess,
            scan_index,
            objectGuess=None,
            Y=None,
            norm_Y_I=None,
            probe_index: Optional[int] = None
        ):
            """
            Initialize RawData with an optional probe_index attribute.
            """
            self.probe_index = probe_index  # Integer indicating probe index for the dataset

    # Ensure that generate_grouped_data() propagates probe indices when processing data
```

9. **Ensure Backward Compatibility and Update Tests**

```aider
UPDATE all modified files:

    # Review and update any existing functions that may be affected by the changes
    # Ensure that single-probe datasets are still supported without modification
    # Update or add unit tests to cover new functionality with multiple probes
    # Validate that the codebase maintains backward compatibility
```