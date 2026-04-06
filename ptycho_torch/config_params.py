from dataclasses import dataclass, field
from typing import Tuple, Optional, Literal, Dict, Union, List, TYPE_CHECKING

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

    # ── Important / Frequently Modified ──────────────────────────────
    N: int = 64                # Diffraction pattern size (px). Power of 2. Typical: 64, 128
    C: int = 4                 # Number of channels. Values: 1 (no overlap) or 4 (grid overlap)
    normalize: Literal['Group', 'Batch'] = 'Batch'  # Data normalization mode. Values: 'Group', 'Batch'
    neighbor_function: Literal['Nearest', 'Min_dist', '4_quadrant'] = 'Nearest'  # Neighbor lookup method. Values: 'Nearest', 'Min_dist', '4_quadrant'. 4_quadrant method is best if you have a grid-like scan pattern.
    scan_pattern: Literal['Isotropic', 'Rectangular'] = 'Isotropic'  # Scan pattern, used for 4_quadrant neighbor function. Values: 'Isotropic', 'Rectangular'. Rectangular is needed when step size on one axis (e.g. y) is much different than the other axis.
    probe_normalize: bool = True   # Whether to normalize the probe. With PtychoPINN-S, this is TRUE during training but FALSE during inference.
    x_bounds: Tuple[float, float] = (0.1, 0.9)  # Scan position bounds (x) for dataloader. Range: each element in [0.0, 1.0]. Helps nearest neighbor search when range is away from edges.
    y_bounds: Tuple[float, float] = (0.1, 0.9)  # Scan position bounds (y) for dataloader. Range: each element in [0.0, 1.0]
    min_neighbor_distance: float = 0.0   # Minimum neighbor distance threshold. Range: >= 0. This is for coordinate grouping for overlap constraint.
    max_neighbor_distance: float = 3.0   # Maximum neighbor distance threshold. Range: >= 0
    K_quadrant: int = 30       # Number of nearest neighbors for quadrant lookup. Range: > 0

    # ── Static / Advanced ────────────────────────────────────────────
    nphotons: float = 1e5      # Simulated photon count for Poisson noise. Range: > 0. Typical: 1e4–1e7. Rarely used
    n_subsample: int = 7       # Subsampling factor for coordinates. Range: >= 1. Used if subsampling nearest neighbors more than once
    probe_scale: float = 1.0      # Probe amplitude scaling factor. Range: > 0
    K: int = 6                 # Number of nearest neighbors for lookup. Range: > 0
    grid_size: Tuple[int, int] = (2, 2)  # Grid size for scanning positions
    subsample_seed: Optional[int] = None  # Random seed for reproducible subsampling
    probe_ramp_removal: bool = False  # Remove linear phase ramp from probe
    data_scaling: Literal['Parseval', 'Max'] = 'Parseval'  # Data scaling method. Values: 'Parseval', 'Max'. Almost always parseval
    phase_subtraction: bool = True  # Subtract mean phase from ground truth. Only useful for supervised training

@dataclass
class ModelConfig:
    """Configuration parameters related to the model architecture and behavior."""

    # ── Important / Frequently Modified ──────────────────────────────
    mode: Literal['Supervised', 'Unsupervised'] = 'Unsupervised'  # Training mode. Values: 'Supervised', 'Unsupervised'
    object_big: bool = False   # True if object requires patch reassembly. Auto-set when C > 1
    probe_big: bool = True     # True if probe requires patch reassembly
    loss_function: Literal['MAE', 'Poisson'] = 'Poisson'  # Primary loss function. Values: 'MAE', 'Poisson'
    amp_activation: str = 'silu'  # Activation for amplitude output. Values: 'silu', 'sigmoid'
    cbam_encoder: bool = True  # CBAM attention in encoder. Ablation study parameter
    decoder_last_amp_channels: int = DataConfig.C   # Output channels for amplitude decoder. Values: 1 or 4
    use_shared_decoder: bool = False  # Use single shared decoder instead of dual amp/phase branches

    # ── Static / Advanced ────────────────────────────────────────────
    intensity_scale_trainable: bool = False  # Whether intensity_scale is a learnable parameter
    intensity_scale: float = 10000.0  # Intensity scale guess for physics model. Range: > 0
    max_position_jitter: int = 10  # Random translation jitter for robustness. Range: >= 0
    num_datasets: int = 1      # Number of unique datasets being trained. Range: >= 1
    C_model: int = DataConfig.C  # Channels in model architecture. Typically matches DataConfig.C
    C_forward: int = DataConfig.C  # Channels in forward model. Typically matches DataConfig.C
    amp_loss: Literal['Total_Variation', "Mean_Deviation", None] = None  # Auxiliary amplitude loss. Values: None, 'Total_Variation', 'Mean_Deviation'
    phase_loss: Literal['Total_Variation', "Mean_Deviation", None] = None  # Auxiliary phase loss. Values: None, 'Total_Variation', 'Mean_Deviation'
    amp_loss_coeff: float = 1.0    # Amplitude auxiliary loss weight. Range: >= 0
    phase_loss_coeff: float = 1.0  # Phase auxiliary loss weight. Range: >= 0
    n_filters_scale: int = 2   # Shrinking factor for channels in network layers. Range: >= 1
    probe_mask: Optional[TensorType] = None  # Optional probe mask tensor
    eca_decoder: bool = False  # ECA attention in decoder
    batch_norm: bool = False   # Whether to use batch normalization
    edge_pad: int = 10         # Padding for decoder_last reconstruction. Range: >= 0
    decoder_last_c_outer_fraction: float = 0.125  # Fraction of channels for high-freq components. Range: (0.0, 1.0]
    cbam_bottleneck: bool = False  # CBAM attention in bottleneck
    cbam_decoder: bool = False     # CBAM attention in decoder
    spatial_decoder: bool = False  # Spatial attention in decoder
    decoder_spatial_kernel: int = 7  # Spatial attention kernel size for decoder
    eca_encoder: bool = False  # ECA attention in encoder
    offset: int = 6           # Offset for nearest neighbor patches. Range: >= 0
    pad_object: bool = True    # Pad object during forward model
    gaussian_smoothing_sigma: float = 0.0  # Gaussian smoothing sigma for probe
    probe_reference_coeff: float = 0.0  # Probe reference loss coefficient. Range: >= 0. 0.0 = disabled

@dataclass
class TrainingConfig:
    """Configuration parameters related to the training process."""

    # ── Important / Frequently Modified ──────────────────────────────
    epochs: int = 50           # Total training epochs. Range: >= 1. Overridden if multi-stage is active
    batch_size: int = 16       # Training batch size. Range: >= 1. Typical: power of 2
    learning_rate: float = 1e-3  # Base learning rate. Range: > 0. Typical: 1e-4 to 1e-2
    n_devices: int = 1         # Number of training devices. Range: >= 1
    num_workers: int = 4       # Dataloader workers. Range: >= 0
    accum_steps: int = 1       # Gradient accumulation steps (manual DDP). Range: >= 1
    epochs_fine_tune: int = 0  # Fine-tuning epochs after main training. Range: >= 0. 0 = disabled
    fine_tune_gamma: float = 0.1  # Fine-tuning LR scale factor. Range: (0.0, 1.0]
    gradient_clip_val: Union[float, None] = None  # Max gradient norm. Range: > 0 or None (disabled)
    experiment_name: str = "Synthetic_Runs"  # MLflow experiment name

    # ── Static / Advanced ────────────────────────────────────────────
    training_directories: List[str] = field(default_factory=list)  # Data directories (only used in train_full)
    nll: bool = True           # Use Negative Log Likelihood loss component
    device: str = 'cuda'       # Training device. Values: 'cuda', 'cpu'
    strategy: Optional[str] = 'ddp'  # Distributed training strategy. Values: 'ddp', None
    framework: Literal['Default', 'Lightning'] = 'Lightning'  # Training framework. Values: 'Default', 'Lightning'
    orchestrator: Literal['Mlflow', 'Lightning'] = 'Mlflow'  # Experiment tracking backend. Values: 'Mlflow', 'Lightning'
    scheduler: Literal['Default', 'Exponential', 'MultiStage', 'Adaptive', 'Cosine'] = 'Default'  # LR scheduler type
    warmup_epochs: int = 5     # Warmup epochs for Cosine scheduler. Range: >= 0
    min_lr_ratio: float = 0.1  # Minimum LR ratio for Cosine scheduler. Range: > 0
    physics_weight_schedule: str = 'cosine'  # Multi-stage weight schedule. Values: 'linear', 'cosine', 'exponential'
    torch_loss_mode: Literal['poisson', 'mae'] = 'poisson'  # Backend-specific loss selection. Values: 'poisson', 'mae'
    notes: str = ""            # Free-text run notes for MLflow
    model_name: str = "PtychoPINNv2"  # Model name for MLflow logging

    # Beta: staged fine-tuning (synthetic → experimental transfer)
    enable_staged_finetuning: bool = False  # Master switch for staged fine-tuning
    finetune_stage1_epochs: int = 7    # Decoder-only fine-tuning epochs
    finetune_stage2_epochs: int = 7    # Partial encoder + decoder epochs
    finetune_stage3_epochs: int = 5    # Full network fine-tuning epochs (optional)
    finetune_stage1_lr_decoder: float = 0.1  # Stage 1 decoder LR multiplier
    finetune_stage2_lr_encoder_top: float = 0.01  # Stage 2 top-encoder LR multiplier
    finetune_stage2_lr_decoder: float = 0.05  # Stage 2 decoder LR multiplier
    finetune_stage2_lr_phase_head: float = 0.1  # Stage 2 phase-head LR multiplier
    finetune_stage3_lr_encoder_bottom: float = 0.005  # Stage 3 bottom-encoder LR multiplier
    finetune_stage3_lr_encoder_top: float = 0.01  # Stage 3 top-encoder LR multiplier
    finetune_stage3_lr_decoder: float = 0.02  # Stage 3 decoder LR multiplier
    finetune_stage3_lr_phase_head: float = 0.05  # Stage 3 phase-head LR multiplier
    finetune_skip_stage3: bool = True  # Skip stage 3 (most cases don't need it)
    finetune_early_stop_patience: int = 3  # Per-stage early stopping patience
    finetune_val_split: float = 0.05   # Validation split for fine-tuning. Range: (0.0, 1.0)

    output_dir: str = "lightning_outputs"  # Output directory for Lightning checkpoints
    train_data_file: Optional[str] = None  # Path to training NPZ dataset
    test_data_file: Optional[str] = None   # Path to test NPZ dataset
    n_groups: Optional[int] = None  # Number of grouped samples

@dataclass
class InferenceConfig:
    """Configuration parameters for inference."""

    # ── Important / Frequently Modified ──────────────────────────────
    batch_size: int = 1000     # Reconstruction batch size. Range: >= 1. Lower for GPU memory constraints
    middle_trim: int = 32      # Center-crop size for patch extraction. Range: >= 0, must be <= N
    experiment_number: int = 0  # Experiment index for multi-dataset inference. Range: >= 0
    patch_weighting: Literal['uniform', 'probe'] = 'probe'  # Patch overlap weighting. Values: 'uniform', 'probe'

    # ── Static / Advanced ────────────────────────────────────────────
    pad_eval: bool = True      # Pad evaluation edges (Nyquist). Can disable for eval
    window: int = 20           # Edge padding to discard due to boundary artifacts. Range: >= 0

@dataclass
class DatagenConfig:
    """Configuration parameters for data generation class."""

    # ── Important / Frequently Modified ──────────────────────────────
    objects_per_probe: int = 4  # Unique synthetic objects per probe. Range: >= 1
    diff_per_object: int = 7000  # Diffraction images per object. Range: >= 1
    object_class: str = 'dead_leaves'  # Synthetic object type. Values: 'dead_leaves', 'exp'
    probe_paths: List[str] = field(default_factory=list)  # List of probe file paths
    reim_mode: Literal['gaussian', 'histogram', 'constrained_phase', 'uniform'] = 'histogram'  # Complex value sampling mode

    # ── Static / Advanced ────────────────────────────────────────────
    image_size: Tuple[int, int] = (250, 250)  # Synthetic object canvas size (px)
    beamstop_diameter: int = 4  # Beamstop diameter in forward model (px). Range: >= 0
    histogram_threshold: float = 0.05  # Histogram clipping threshold. Range: [0.0, 1.0]


# Update the existing instance
def update_existing_config(config_instance, updates_dict, verbose = False):
    for key, value in updates_dict.items():
        if hasattr(config_instance, key):
            setattr(config_instance, key, value)
        else:
            if verbose:
            # Bypassing this for now since this function is being used as a generic updater for payloads
                print(f"Warning: Attribute '{key}' not found in {type(config_instance).__name__}")
