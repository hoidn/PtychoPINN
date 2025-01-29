# Multi-Probe Ptychography Support Specification

## High-Level Objective
Modify the codebase to support per-sample probe selection instead of using a global probe tensor, enabling training with multiple probe datasets.

## Mid-Level Objectives
- Add probe list and index support to data containers
- Implement dataset merging with probe handling 
- Update model architecture for probe selection
- Modify training pipeline for multi-probe support

## Implementation Notes
- Use int64 for probe indices
- Only shuffle samples during training, not testing
- Maintain backwards compatibility where possible
- Probe indices must match first dimension of X, Y_I etc.
- All probes must have same shape/dtype

## Context

### Beginning Context
- ptycho/loader.py
- ptycho/model.py  
- ptycho/components.py
- ptycho/raw_data.py
- ptycho/train_pinn.py

### Ending Context
Same files updated with multi-probe support

## Low-Level Tasks

1. Update RawData Class
```aider
UPDATE ptycho/raw_data.py:
    UPDATE class RawData:
        ADD probe_index: int64 attribute to __init__ params
        UPDATE generate_grouped_data():
            MODIFY to preserve probe_index when generating groups
        
        Type hints:
        def __init__(self, ..., probe_index: int64 = 0)
```

2. Update PtychoDataContainer Class
```aider
UPDATE ptycho/loader.py:
    UPDATE class PtychoDataContainer:
        ADD probe_list: List[tf.Tensor] attribute
        ADD probe_indices: tf.Tensor[int64] attribute
        UPDATE __init__() to accept new probe attributes
        ADD merge_containers(containers: List[PtychoDataContainer], shuffle: bool = True) -> PtychoDataContainer
        
        Type hints:
        def __init__(self, ..., probe_list: List[tf.Tensor], probe_indices: tf.Tensor)
```

3. Update Model Architecture
```aider
UPDATE ptycho/model.py:
    UPDATE class ProbeIllumination:
        MODIFY call() to select probes using indices:
            INPUT tensor shape: (batch_size, probe_dim1, probe_dim2, 1)
            INPUT indices shape: (batch_size,) 
            USE tf.gather() to select probes by index
    
    UPDATE model inputs to include:
        input_probe_list: Input(shape=(None, N, N, 1), dtype=tf.complex64)
        input_probe_indices: Input(shape=(), dtype=tf.int64)
```

4. Update Training Pipeline
```aider
UPDATE ptycho/train_pinn.py:
    UPDATE prepare_inputs():
        ADD probe_list and probe_indices to returned inputs
    UPDATE train():
        MODIFY to handle datasets with multiple probes
        ADD support for shuffled vs non-shuffled merge

    Type hints:
    def prepare_inputs(container: PtychoDataContainer) -> Tuple[tf.Tensor, tf.Tensor, List[tf.Tensor], tf.Tensor]
```

5. Add Merge Support Functions
```aider
UPDATE ptycho/loader.py:
    CREATE merge_containers(containers: List[PtychoDataContainer], shuffle: bool = True) -> PtychoDataContainer:
        Merge multiple containers:
        - Concatenate all tensors (X, Y_I etc)
        - Combine probe lists 
        - Update probe indices to point to merged probe list
        - Optionally shuffle samples (for training)
        RETURN merged container
```