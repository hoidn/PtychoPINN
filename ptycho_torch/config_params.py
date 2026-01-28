from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, Union, List

# PyTorch is now a mandatory dependency (Phase F3.1 gate)
# Per plans/active/INTEGRATE-PYTORCH-001/phase_f_torch_mandatory.md F3.2
try:
    import torch
    TensorType = torch.Tensor
except ImportError as e:
    raise RuntimeError(
        "PyTorch is required for ptycho_torch modules. "
        "Install PyTorch >= 2.2 with: pip install torch>=2.2"
    ) from e

# Configuration dataclasses for PtychoNN (PyTorch version)

@dataclass
class DataConfig:
    """Configuration parameters related to data loading and generation."""
    nphotons: float = 1e5

    #General sizing parameters
    N: int = 64  # Size of the diffraction patterns/object patch
    C: int = 4    # Number of channels
    K: int = 6    # Number of nearest neighbors for lookup
    #Grid parameters specifically for overlap constraint
    K_quadrant: int = 30 # Number of nearest neighbors for quadrant lookup
    n_subsample: int = 7 # Subsampling factor for coordinates (if applicable)
    subsample_seed: Optional[int] = None # Random seed for reproducible subsampling
    grid_size: Tuple[int, int] = (2, 2) # Grid size for scanning positions
    neighbor_function: Literal['Nearest','Min_dist','4_quadrant'] = 'Nearest'
    min_neighbor_distance: float = 0.0
    max_neighbor_distance: float = 3.0
    scan_pattern: Literal['Isotropic', 'Rectangular'] = 'Isotropic' # Scan pattern, used for 4_quadrant neighbor function

    #Miscellaneous
    normalize: Literal['Group', 'Batch'] = 'Batch' # Whether to normalize the data
    probe_scale: float = 1.0
    probe_normalize: bool = True
    data_scaling: Literal['Parseval','Max'] = 'Parseval'
    phase_subtraction: bool = True #Only useful for supervised training dataset

    #Bounding parameters for scan positions
    x_bounds: Tuple[float, float] = (0.1,0.9)
    y_bounds: Tuple[float, float] = (0.1,0.9)

@dataclass
class ModelConfig:
    """Configuration parameters related to the model architecture and behavior."""
    #Mode Category
    mode: Literal['Supervised', 'Unsupervised'] = 'Unsupervised' # Training mode, affects all aspects of model
    architecture: Literal['cnn', 'fno', 'hybrid', 'stable_hybrid'] = 'cnn'  # Generator architecture selection
    fno_modes: int = 12
    fno_width: int = 32
    fno_blocks: int = 4
    fno_cnn_blocks: int = 2
    fno_input_transform: Literal['none', 'sqrt', 'log1p', 'instancenorm'] = 'none'
    generator_output_mode: Literal['real_imag', 'amp_phase_logits', 'amp_phase'] = 'real_imag'

    #Intensity Parameters
    intensity_scale_trainable: bool = False
    intensity_scale: float = 10000.0 # General intensity scale guess
    max_position_jitter: int = 10 # Random jitter for translation (robustness)
    num_datasets: int = 1 #Number of unique datasets being trained. For instantiating fitting constants

    #Model architecture parameters
    C_model: int = DataConfig.C
    n_filters_scale: int = 2 # Shrinking factor for channels in network layers
    amp_activation: str = 'silu' # Activation function for amplitude part
    batch_norm: bool = False # Whether to use batch normalization
    probe_mask: Optional[TensorType] = None # Optional probe mask tensor

    #Module-specific
    edge_pad: int = 10 #For padding the decoder_last reconstruction
    decoder_last_c_outer_fraction: float = 0.125 #Amount of channels going to higher frequency components in decoder_last
    decoder_last_amp_channels: int = 1

    #Attention
    eca_encoder: bool = False
    cbam_encoder: bool = True #Whether CBAM module is turned on for encoder
    cbam_bottleneck: bool = False #CBAM bottleneck
    cbam_decoder: bool = False #CBAM for decoder
    eca_decoder: bool = False #ECA for decoder
    spatial_decoder: bool = False #Spatial attention for decoder
    decoder_spatial_kernel: int = 7 #Spatial attention kernel for decoder

    #Forward model parameters
    object_big: bool = False # True if object requires patch reassembly
    probe_big: bool = True # True if probe requires patch reassembly
    offset: int = 6 # Offset parameter (for nearest neighbor patches)
    C_forward: int = DataConfig.C # Number of channels

    # Spec-mandated defaults (align with TensorFlow backend)
    pad_object: bool = True # Pad object during forward model
    gaussian_smoothing_sigma: float = 0.0 # Gaussian smoothing sigma for probe

    #Loss
    loss_function: Literal['MAE', 'Poisson'] = 'Poisson' # Loss function to use ('MAE', 'MSE', etc.)
    amp_loss: Literal['Total_Variation', "Mean_Deviation", None] = None
    phase_loss: Literal['Total_Variation', "Mean_Deviation", None] = None
    amp_loss_coeff: float = 1.0
    phase_loss_coeff: float = 1.0
    
    

@dataclass
class TrainingConfig:
    """Configuration parameters related to the training process."""
    # Directories (only used in train_full)
    training_directories: List[str] = field(default_factory=list)

    # Device/Loss
    nll: bool = True # Use Negative Log Likelihood loss component
    device: str = 'cuda' # Device to train on ('cuda', 'cpu')
    strategy: Optional[str] = 'ddp' # Strategy for distributed training (e.g., 'ddp', None)
    n_devices: int = 1 #Number of devices you're training on

    # Framework
    framework: Literal['Default', 'Lightning'] = 'Lightning' #Training framework. Most of work don in PT was done in lightning
    orchestrator: Literal['Mlflow', 'Lightning'] = 'Mlflow'

    # Add other training-specific parameters here as needed, e.g.:
    learning_rate: float = 1e-3
    epochs: int = 50 #Default epochs number, will be overridden if multi-stage training is active at all
    batch_size: int = 16
    epochs_fine_tune: int = 0 #Default 0 fine-tune means no fine-tuning
    fine_tune_gamma: float = 0.1 #Scales base LR for fine-tuning
    scheduler: Literal['Default', 'Exponential', 'MultiStage', 'Adaptive'] = 'Default'
    num_workers: int = 4 #Dataloader workers
    accum_steps: int = 1 #Batch size accumulation, manually implemented for DDP
    gradient_clip_val: Union[float,None] = None #Gradient clip value
    gradient_clip_algorithm: str = 'norm'  # Gradient clipping algorithm: 'norm', 'value', or 'agc'
    log_grad_norm: bool = False
    grad_norm_log_freq: int = 1
    # batch_size: int = 32

    # Meta learning: Schedulers etc.
    stage_1_epochs: int = 0       # Will be set to total epochs if not specified
    stage_2_epochs: int = 0        # Weighted transition (0 = disabled)
    stage_3_epochs: int = 0        # Physics only (0 = disabled)
    physics_weight_schedule: str = 'cosine'  # 'linear', 'cosine', 'exponential'

    # Multi-stage learning rate parameters
    stage_3_lr_factor: float = 0.1    # LR reduction for stage 3 (physics)

    # Backend-specific loss selection
    torch_loss_mode: Literal['poisson', 'mae'] = 'poisson'

    #MLFlow config
    experiment_name: str = "Synthetic_Runs"
    notes: str = ""
    model_name: str = "PtychoPINNv2"

    #Lightning specific configs
    output_dir: str = "lightning_outputs"

    # Spec-mandated lifecycle fields (align with TensorFlow backend)
    train_data_file: Optional[str] = None # Path to training NPZ dataset
    test_data_file: Optional[str] = None # Path to test NPZ dataset
    output_dir: str = "training_outputs" # Output directory for checkpoints/logs
    n_groups: Optional[int] = None # Number of grouped samples

@dataclass
class InferenceConfig:
    """Configuration parameters for inference"""
    middle_trim: int = 32
    batch_size: int = 1000 #Batch size for reconstruction. Lower this due to GPU memory bandwidth
    experiment_number: int = 0  #Experiment number for inference
    pad_eval: bool = True #Pads the evaluation edges, enforced during training for Nyquist frequency. Can turn off for eval
    window: int = 20 #Window padding around reconstruction due to edge errors

@dataclass
class DatagenConfig:
    """Configuration parameters for data generation class"""
    objects_per_probe: int = 4 #Number of unique synthetic objects per probe function
    diff_per_object: int = 7000 #Number of diffraction images per unique object
    object_class: str = 'dead_leaves'
    image_size: Tuple[int, int] = (250,250)
    probe_paths: List[str] = field(default_factory=list) # List of all probe files used
    beamstop_diameter: int = 4 # For simulating beamstop in forward model




# Update the existing instance
def update_existing_config(config_instance, updates_dict, verbose = False):
    for key, value in updates_dict.items():
        if hasattr(config_instance, key):
            setattr(config_instance, key, value)
        else:
            if verbose:
            # Bypassing this for now since this function is being used as a generic updater for payloads
                print(f"Warning: Attribute '{key}' not found in {type(config_instance).__name__}")


# Example Usage (how users would now use these configs):
#
# if __name__ == "__main__":
#     # Instantiate with defaults
#     data_cfg = DataConfig()
#     model_cfg = ModelConfig()
#     train_cfg = TrainingConfig()
#
#     # Override specific defaults
#     model_cfg_custom = ModelConfig(loss_function='MSE', batch_norm=True)
#     train_cfg_cpu = TrainingConfig(device='cpu', strategy=None)
#
#     print(data_cfg)
#     print(model_cfg_custom)
#     print(train_cfg_cpu)
#
#     # Access parameters like attributes
#     print(f"Image size: {data_cfg.N}")
#     print(f"Loss function: {model_cfg_custom.loss_function}")
