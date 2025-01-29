# Multiple Probe Support Specification
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective
- Adapt codebase to support per-sample probe inputs instead of global probe variable

## Mid-Level Objective
- Modify data containers to store multiple probes and probe indices
- Update model architecture for probe selection
- Enable training with mixed probe datasets
- Support dataset merging with probe tracking
- Preserve test set handling without shuffling

## Implementation Notes
- Use tf.Tensor for probe storage
- Probe indices dtype: int64  
- Probe indices dimension matches batch size
- All probes must have same shape/dtype
- Dataset merging requires shuffling for training only
- Probe index maintains dataset origin tracking

## Context

### Beginning Context
- loader.py: Data container definitions
- model.py: Neural network model
- components.py: Support components  
- raw_data.py: Base data classes
- train_pinn.py: Training workflow

### Ending Context
Same files updated with multi-probe support

## Low-Level Tasks

1. Update RawData Class
```aider
UPDATE raw_data.py:
    ADD probe_index: int64 to RawData class attributes
    UPDATE __init__ signature to include probe_index
    UPDATE from_simulation() to set probe_index 
    UPDATE generate_grouped_data() to preserve probe_index
```

2. Modify PtychoDataContainer
```aider
UPDATE loader.py:
    UPDATE PtychoDataContainer class:
        ADD probe_list: List[tf.Tensor]
        ADD probe_indices: tf.Tensor[int64]
        UPDATE __init__ signature for new attributes
        ADD merge_containers(containers: List[PtychoDataContainer], shuffle: bool = True) -> PtychoDataContainer
```

3. Update Model Architecture 
```aider
UPDATE model.py:
    UPDATE ProbeIllumination layer:
        UPDATE call(inputs) to use probe_indices for selection
        ADD support for probe_list input
    UPDATE model inputs to include probe tensors
    UPDATE training/inference functions for probe handling
```

4. Modify Training Pipeline
```aider
UPDATE train_pinn.py:
    UPDATE prepare_inputs() to include probe data
    UPDATE train() to handle multiple probes
    UPDATE eval() for probe handling
```

5. Update Support Components
```aider
UPDATE components.py:
    UPDATE create_ptycho_data_container() for probe indices
    UPDATE train_cdi_model() to handle probe inputs
```