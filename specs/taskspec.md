# Multi-Probe Ptychography Support Specification

## High-Level Objective
- Adapt codebase to support per-sample probe tensors via probe list and index selection

## Mid-Level Objectives
- Convert PtychoDataContainer to support probe lists and indices
- Update model to handle probe selection per sample
- Modify data loading pipeline for probe index tracking
- Add dataset merging with probe handling

## Implementation Notes
- probe_list must be List[tf.Tensor]
- probe_indices must be tf.Tensor(dtype=tf.int64)
- All probes must have same shape/dtype
- Training requires shuffling, testing preserves order
- Dataset merging must track probe associations

### Beginning Context
- loader.py
- model.py 
- raw_data.py
- train_pinn.py
- workflows/components.py

### Ending Context
Same files with probe list support added

## Low-Level Tasks

1. Update PtychoDataContainer Constructor
```aider
UPDATE loader.py:
    UPDATE class PtychoDataContainer:
        UPDATE __init__():
            ADD probe_list: List[tf.Tensor] parameter
            ADD probe_indices: tf.Tensor parameter
            REPLACE self.probe with self.probe_list,self.probe_indices
            ADD validation:
                - All probes same shape/dtype
                - probe_indices int64
                - probe_indices matches batch size
                - probe_indices in valid range
        
        ADD @property def probe():
            Return probe_list[0] for backward compatibility
```

2. Add Dataset Merging Support
```aider
UPDATE loader.py:
    UPDATE class PtychoDataContainer:
        CREATE @staticmethod merge_containers(containers: List[PtychoDataContainer], shuffle: bool = True) -> PtychoDataContainer:
            - Concatenate X,Y_I etc
            - Merge probe lists
            - Update probe indices with offset
            - Optional shuffling
            - Return new container
```

3. Update Model Probe Handling
```aider
UPDATE model.py:
    UPDATE class ProbeIllumination:
        ADD probe_indices input tensor to call()
        MODIFY illumination logic to select probe by index
        UPDATE output shape handling
```

4. Update Data Loading Pipeline 
```aider
UPDATE raw_data.py:
    UPDATE class RawData:
        ADD probe_index field
        UPDATE generate_grouped_data():
            Preserve probe indices when grouping
        UPDATE from_simulation():
            Support probe_index parameter

UPDATE workflows/components.py:
    UPDATE create_ptycho_data_container():
        Handle probe indices from RawData
    UPDATE train_cdi_model():
        Pass probe indices through training
```

5. Update Training Pipeline
```aider
UPDATE train_pinn.py:
    UPDATE prepare_inputs():
        ADD probe_indices to model inputs
    UPDATE train():
        Handle datasets with multiple probes
    UPDATE eval():
        Preserve probe associations
```