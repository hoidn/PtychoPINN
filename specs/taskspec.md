# Multi-Probe Support Implementation Specification

## High-Level Objective
Adapt codebase to support per-sample probe tensors instead of a global probe variable

## Mid-Level Objective
- Modify data containers to store multiple probes and probe indices
- Update model to select appropriate probe per sample
- Support merging datasets with probe handling
- Maintain separate test data handling

## Implementation Notes

Dependencies:
- tensorflow
- numpy 
- Existing ptycho modules

Data Structure Changes:
- probe_list: List[tf.Tensor] - Multiple probe tensors
- probe_indices: tf.Tensor[int64] - Maps samples to probe indices

Implementation Guidelines:
- Maintain backward compatibility where possible
- Preserve existing data normalization 
- Follow existing type hints and docstring patterns

## Context

### Beginning context
- loader.py
- model.py
- components.py
- raw_data.py
- train_pinn.py

### Ending context
Same files updated with multi-probe support

## Low-Level Tasks

1. Update RawData class in raw_data.py

```aider
UPDATE raw_data.py:
    UPDATE class RawData:
        ADD probe_index: int64 field to __init__ signature
        UPDATE __init__ to store probe_index
        UPDATE generate_grouped_data to preserve probe_index
        UPDATE from_file and to_file to handle probe_index
        UPDATE from_simulation to handle probe_index
```

2. Update PtychoDataContainer in loader.py

```aider
UPDATE loader.py:
    UPDATE class PtychoDataContainer:
        MODIFY __init__(self, X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, nn_indices, global_offsets, local_offsets, probe_list: List[tf.Tensor], probe_indices: tf.Tensor):
            - Replace probe with probe_list and probe_indices parameters
            - Add validation for probe shapes and dtypes
            - Update repr to show probe list info
        
        ADD merge_containers(containers: List[PtychoDataContainer], shuffle: bool = True) -> PtychoDataContainer:
            Merge multiple containers preserving probe associations
            For training data (shuffle=True): Interleave samples randomly
            For test data (shuffle=False): Concatenate in order
            Return new container with combined data and probes
```

3. Update model architecture in model.py

```aider
UPDATE model.py:
    UPDATE class ProbeIllumination:
        MODIFY call(self, inputs):
            - Take probe_list and probe_indices as input
            - Use tf.gather to select correct probe per sample
            - Apply illumination using selected probes

    UPDATE model input/output definitions:
        ADD input_probe_list = Input(shape=(None, N, N, 1), dtype=tf.complex64)
        ADD input_probe_indices = Input(shape=(), dtype=tf.int32) 
        MODIFY model input list to include new probe inputs
```

4. Update training workflow in train_pinn.py

```aider
UPDATE train_pinn.py:
    UPDATE prepare_inputs():
        MODIFY to include probe_list and probe_indices from container
    
    UPDATE train():
        MODIFY to handle datasets with multiple probes
        Ensure probe indices are passed correctly to model
```

5. Update data preparation in components.py

```aider
UPDATE components.py:
    UPDATE create_ptycho_data_container():
        MODIFY to handle probe_list and probe_indices
        Add validation for probe inputs
    
    UPDATE train_cdi_model():
        MODIFY to pass probe information to training
```