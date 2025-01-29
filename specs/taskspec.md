# Multi-Probe Ptychography Analysis Support
> Ingest the information from this file, implement the Low-Level Tasks, and generate the code that will satisfy the High and Mid-Level Objectives.

## High-Level Objective

Enable ptychographic analysis of samples measured using different probe configurations, improving reconstruction quality by preserving probe-specific characteristics.

## Mid-Level Objectives

1. Support sample analysis across multiple probe configurations
   - Merge datasets from different probe measurements
   - Maintain probe-sample relationships
   - Validate probe compatibility

2. Enable efficient training on merged datasets
   - Support sample shuffling during training
   - Preserve probe associations
   - Scale to multiple probes efficiently

3. Provide clean interfaces for probe handling
   - Abstract probe management from core processing
   - Maintain backward compatibility 
   - Support probe-specific analysis

## Implementation Notes

### Core Dependencies
- ptycho.loader.PtychoDataContainer: Base data container class
- ptycho.model.ProbeIllumination: Core probe processing
- tensorflow>=2.4: For tensor operations
- numpy>=1.19: For array operations

### Data Flow Requirements
1. Data Loading:
   - RawData loads individual probe datasets
   - PtychoDataContainer processes single probe data
   - MultiPtychoDataContainer merges probe datasets

2. Model Processing:
   - Input tensors include probe indices
   - ProbeIllumination selects per-sample probes
   - Output maintains probe associations

3. Training Flow:
   - Optional shuffling during training
   - Fixed ordering during testing
   - Probe selection via tf.gather

### Technical Constraints
- Probe tensors: Same shape and dtype
- Probe indices: int64 dtype
- Shuffling: Only during training
- Backward compatibility: Single probe as special case
- Memory efficiency: Use tf.gather for probe selection

## Context

### Beginning Context
```python
# loader.py
class PtychoDataContainer:
    def __init__(self, X, Y_I, Y_phi, probe, ...):
        # Single probe handling
        self.probe = probe

# model.py
class ProbeIllumination:
    def call(self, inputs):
        # Global probe usage
        x = inputs[0]
        return self.probe * x
```

### Ending Context
```python
# loader.py
class MultiPtychoDataContainer:
    def __init__(self, X, Y_I, Y_phi, probe_list, probe_indices, ...):
        # Multi-probe handling
        self.probe_list = probe_list
        self.probe_indices = probe_indices

# model.py
class ProbeIllumination:
    def call(self, inputs):
        # Per-sample probe selection
        x, probe_indices = inputs
        probes = tf.gather(self.probe_list, probe_indices)
        return probes * x
```

## Low-Level Tasks
> Ordered from start to finish

1. Create MultiPtychoDataContainer Base Structure
```aider 
CREATE in ./ptycho/loader.py:
    def validate_probe_tensors(probe_list: List[tf.Tensor]) -> None:
        """Validate probe tensor compatibility.
        
        Args:
            probe_list: List of probe tensors to validate
            
        Raises:
            ValueError: If probes incompatible
        """

    class MultiPtychoDataContainer:
        def __init__(self,
            X: tf.Tensor,
            Y_I: tf.Tensor, 
            Y_phi: tf.Tensor,
            norm_Y_I: tf.Tensor,
            coords: tf.Tensor,
            probe_list: List[tf.Tensor],
            probe_indices: tf.Tensor
        ) -> None:
            """Initialize multi-probe container.
            
            Validates and stores probe tensors and indices.
            """
            validate_probe_tensors(probe_list)
            # Initialize attributes
```

2. Add Core MultiPtychoDataContainer Methods
```aider
UPDATE ./ptycho/loader.py class MultiPtychoDataContainer:
    def get_probe(self, index: int) -> tf.Tensor:
        """Get probe tensor by index.
        
        Args:
            index: Probe index to retrieve
            
        Returns:
            Selected probe tensor
            
        Raises:
            IndexError: If index invalid
        """

    def shuffle_samples(self) -> None:
        """Shuffle samples while maintaining probe associations."""
```

3. Create Dataset Merging Function
```aider
CREATE in ./ptycho/loader.py:
    def merge_containers(
        containers: List[PtychoDataContainer],
        shuffle: bool = True
    ) -> MultiPtychoDataContainer:
        """Merge containers preserving probe associations.
        
        Args:
            containers: List of containers to merge
            shuffle: Whether to shuffle samples
            
        Returns:
            Merged container instance
        """
```

4. Update ProbeIllumination Input Handling
```aider
UPDATE ./ptycho/model.py class ProbeIllumination:
    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: Optional[bool] = None
    ) -> tf.Tensor:
        """Apply probe illumination.
        
        Args:
            inputs: (samples, probe_indices)
            training: Training mode flag
            
        Returns:
            Illuminated samples
        """
        x, probe_indices = inputs
        probes = tf.gather(self.probe_list, probe_indices)
        return self._apply_probe(x, probes)
```

5. Add Model Input Layer
```aider
UPDATE ./ptycho/model.py:
    def create_model_inputs() -> List[tf.keras.layers.Input]:
        """Create model input layers.
        
        Returns:
            List of input layers including probe indices
        """
        return [
            Input(shape=(N, N, gridsize**2), name='input'),
            Input(shape=(1, 2, gridsize**2), name='positions'),
            Input(shape=(), dtype=tf.int64, name='probe_indices')
        ]
```

6. Update Data Container Creation
```aider
UPDATE ./ptycho/workflows/components.py:
    def create_container(
        data: Union[RawData, PtychoDataContainer],
        config: TrainingConfig
    ) -> Union[PtychoDataContainer, MultiPtychoDataContainer]:
        """Create appropriate data container.
        
        Args:
            data: Input data
            config: Configuration settings
            
        Returns:
            Container instance
        """
```

7. Modify Training Input Preparation
```aider
UPDATE ./ptycho/train_pinn.py:
    def prepare_inputs(
        data: Union[PtychoDataContainer, MultiPtychoDataContainer]
    ) -> List[tf.Tensor]:
        """Prepare model inputs including probe indices.
        
        Args:
            data: Training data container
            
        Returns:
            Model input tensors
        """
```

8. Add Raw Data Probe Support  
```aider
UPDATE ./ptycho/raw_data.py class RawData:
    def __init__(
        self,
        probe_index: Optional[int] = None,
        **kwargs
    ) -> None:
        """Initialize raw data with probe index.
        
        Args:
            probe_index: Index for dataset's probe
            **kwargs: Additional arguments
        """
```
