# Multiple Probe Dataset Support Specification

## High-Level Objective
- Adapt codebase to support per-sample probe tensors instead of global probe variables

## Mid-Level Objectives
- Implement new data container supporting multiple probes
- Modify model architecture to handle probe selection
- Update training pipeline for mixed probe datasets
- Maintain test/train split functionality

## Implementation Notes
- Use int64 for probe indices
- Preserve probe-sample relationships during shuffling
- Ensure all probes have same shape/dtype
- Test data handling differs only in shuffling
- Use tf.Tensor for probe storage

## Context

### Beginning Context
- loader.py - Core data container implementation
- model.py - Neural network model definition
- raw_data.py - Base data class implementation
- train_pinn.py - Training workflow

### Ending Context
Same files updated with multiple probe support

## Low-Level Tasks

1. Update PtychoDataContainer Class
```aider
UPDATE loader.py:
    UPDATE class PtychoDataContainer:
        ADD probes: List[tf.Tensor] - list of probe tensors 
        ADD probe_indices: tf.Tensor - int64 tensor mapping samples to probes
        UPDATE __init__(self, X, Y_I, Y_phi, norm_Y_I, YY_full, coords_nominal, coords_true, nn_indices, global_offsets, local_offsets, probes: List[tf.Tensor], probe_indices: tf.Tensor)
        ADD merge_containers(containers: List[PtychoDataContainer], shuffle: bool = True) -> PtychoDataContainer
```

2. Modify Model Probe Handling
```aider
UPDATE model.py:
    UPDATE class ProbeIllumination:
        UPDATE __init__() to accept probe list
        UPDATE call() to select probe by index:
            Input: x, probe_indices 
            Use tf.gather to select probe by index
```

3. Update Training Pipeline
```aider
UPDATE train_pinn.py:
    UPDATE prepare_inputs():
        ADD probe_indices to returned tensors
    UPDATE train():
        MODIFY to handle datasets with multiple probes
        ADD shuffle control based on train vs test
```

4. Update Raw Data Support
```aider
UPDATE raw_data.py:
    UPDATE class RawData:
        ADD probe_index: np.ndarray
        UPDATE generate_grouped_data() to preserve probe indices
```