# Multi-Probe Support Specification
> Implement per-sample probe selection replacing global probe variable

## High-Level Objective
- Adapt codebase to support per-sample probe selection instead of global probe variable

## Mid-Level Objectives
- Implement probe list and index storage in data containers
- Modify model architecture for probe selection
- Update training pipeline for multi-probe support
- Maintain data organization during training/testing

## Implementation Notes
- probe_indices must be int64 dtype
- probe_list elements must be tf.Tensor type
- All probes must have same shape/dtype
- Sample order preserved for testing, shuffled for training
- Probe index doubles as dataset identifier

## Context

### Beginning Context
- ./ptycho/loader.py
- ./ptycho/model.py
- ./ptycho/components.py
- ./ptycho/raw_data.py
- ./ptycho/train_pinn.py

### Ending Context
Same files modified to support per-sample probe selection

## Low-Level Tasks

1. Add Probe Support to RawData
```aider
UPDATE ptycho/raw_data.py:
    UPDATE class RawData:
        ADD probe_index: int64 field
        UPDATE __init__ signature to include probe_index: int64 = 0
        UPDATE generate_grouped_data() to propagate probe_index
```

2. Enhance PtychoDataContainer
```aider
UPDATE ptycho/loader.py:
    UPDATE class PtychoDataContainer:
        ADD probe_list: List[tf.Tensor]
        ADD probe_indices: tf.Tensor[int64]
        UPDATE __init__ signature:
            ADD probe_list: List[tf.Tensor], probe_indices: tf.Tensor[int64]
        CREATE merge_containers(containers: List[PtychoDataContainer], shuffle: bool = True) -> PtychoDataContainer
```

3. Update Model Architecture
```aider
UPDATE ptycho/model.py:
    UPDATE class ProbeIllumination:
        UPDATE call() to handle probe_list and probe_indices inputs
        UPDATE signature: def call(self, inputs: Tuple[tf.Tensor, List[tf.Tensor], tf.Tensor[int64]]) -> tf.Tensor
    
    UPDATE model input structure:
        ADD input_probe_list = Input(shape=(None, N, N, 1), dtype=tf.complex64, name='probe_list')
        ADD input_probe_indices = Input(shape=(), dtype=tf.int64, name='probe_indices')
```

4. Modify Training Pipeline
```aider
UPDATE ptycho/train_pinn.py:
    UPDATE prepare_inputs():
        ADD probe_list and probe_indices to return tuple
    UPDATE train():
        MODIFY to handle multiple probes during training
```