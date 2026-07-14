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
    scale_contract_version: Literal['ci_intensity_v2', 'legacy_v1'] = 'ci_intensity_v2'
    measurement_domain: Literal['count_intensity', 'normalized_amplitude'] = 'count_intensity'

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
    normalize: Literal['Group', 'Batch', 'None'] = 'Batch' # Whether to normalize the data
    probe_scale: float = 4.0
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
    architecture: Literal[
        'cnn', 'ffno', 'fno', 'fno_vanilla', 'neuralop_uno'
    ] = 'cnn'  # Generator architecture selection
    fno_modes: int = 12
    fno_width: int = 32
    fno_blocks: int = 4
    fno_cnn_blocks: int = 2
    learned_input_channels: int = 1
    fno_input_transform: Literal['none', 'sqrt', 'log1p', 'instancenorm'] = 'none'
    max_hidden_channels: Optional[int] = None
    resnet_width: Optional[int] = None
    spectral_bottleneck_blocks: int = 6
    spectral_bottleneck_modes: int = 12
    spectral_bottleneck_share_weights: bool = True
    spectral_bottleneck_gate_init: float = 0.1
    spectral_bottleneck_gate_mode: Literal['shared', 'per_block'] = 'shared'
    generator_output_mode: Literal['real_imag', 'amp_phase_logits', 'amp_phase'] = 'real_imag'
    # CNN decoder output parameterization (Task 2.3 / backlog B1). Distinct from
    # ``generator_output_mode`` (that knob targets FNO/Hybrid cores and defaults to
    # 'real_imag'; reusing it here would silently flip the CNN default). Contract in
    # ptycho_torch.model._effective_cnn_output_mode / _predict_complex_patches.
    #
    # - 'amp_phase' (default): fno-stable's current CNN behavior, UNCHANGED. Amplitude
    #   head uses Amplitude_activation (silu/sigmoid); phase head uses pi*tanh. No
    #   representability restriction -- correct for high-phase-contrast objects.
    # - 'real_imag' (opt-in, UNSUPERVISED-ONLY per Amendment #4): the CNN emits
    #   (real, imag) heads combined via torch.complex. HARD REPRESENTABILITY
    #   CONSTRAINT (Amendment #13): the two heads carry main's hardwired ScaledTanh
    #   box -- real via tanh+0.2 (range (-0.8, 1.2)) and imag via 1.2*tanh (range
    #   (-1.2, 1.2)). The real floor -0.8 makes a unit-amplitude object at |phase|->pi
    #   (real ~ -1, imag ~ 0) UNRECONSTRUCTABLE in real_imag mode. Use 'amp_phase'
    #   (pi*tanh phase head, no such box) for high-phase-contrast objects. See
    #   docs/plans/2026-07-01-varpro-ablation-phase1-findings.md (Phase-1 findings)
    #   and plan-amendments-pending.md finding #13.
    cnn_output_mode: Literal['amp_phase', 'real_imag'] = 'amp_phase'

    # Shared decoder (Task 2.4 / backlog B2, opt-in). Default False leaves fno-stable's
    # current CNN architecture (separate Decoder_amp/Decoder_phase) UNCHANGED. When True,
    # ``Autoencoder`` builds a single ``Decoder_shared`` (ported from main; see
    # ptycho_torch.model.Decoder_shared / FeatureRefinementBlock) that emits 2*C_out raw
    # channels, split into two C_out-channel branches. The per-branch activation gating
    # (Amplitude_activation/pi*tanh for 'amp_phase', ScaledTanh box for 'real_imag') is
    # applied AFTER the split so it stays in lockstep with ``cnn_output_mode`` /
    # ``_effective_cnn_output_mode`` exactly as the separate-decoder path does.
    use_shared_decoder: bool = False

    #Intensity Parameters
    intensity_scale_trainable: bool = False
    intensity_scale: float = 10000.0 # General intensity scale guess
    max_position_jitter: int = 10 # Deprecated in Torch path (padded size ignores this)
    num_datasets: int = 1 #Number of unique datasets being trained. For instantiating fitting constants

    #Model architecture parameters
    C_model: int = DataConfig.C
    n_filters_scale: int = 2 # Shrinking factor for channels in network layers
    amp_activation: str = 'silu' # Activation function for amplitude part
    batch_norm: bool = False # Whether to use batch normalization
    # Probe mask controls (Torch default: enabled soft disk mask).
    # - bool toggle: True -> auto soft mask, False -> disabled
    # - tensor value: explicit custom mask (legacy-compatible path)
    probe_mask: Optional[Union[bool, TensorType]] = False
    probe_mask_tensor: Optional[TensorType] = None
    probe_mask_sigma: float = 1.0
    probe_mask_diameter: Optional[float] = None

    #Module-specific
    edge_pad: int = 10 #For padding the decoder_last reconstruction
    decoder_last_c_outer_fraction: float = 0.125 #Amount of channels going to higher frequency components in decoder_last
    # Historical checkpoint-shape compatibility only. Normal CNN heads use the
    # semantic component count derived from object_big and C_model.
    decoder_last_amp_channels: int = 1
    use_legacy_decoder_channel_override: bool = False

    #Attention
    eca_encoder: bool = False
    cbam_encoder: bool = True #Whether CBAM module is turned on for encoder
    cbam_bottleneck: bool = False #CBAM bottleneck
    cbam_decoder: bool = False #CBAM for decoder
    eca_decoder: bool = False #ECA for decoder
    spatial_decoder: bool = False #Spatial attention for decoder
    decoder_spatial_kernel: int = 7 #Spatial attention kernel for decoder

    #Forward model parameters
    object_big: bool = True # True if object requires patch reassembly
    # Normal object-big CNNs require learned full-patch support. False is an
    # explicit historical/diagnostic zero-border opt-out.
    probe_big: bool = True
    offset: int = 6 # Offset parameter (for nearest neighbor patches)
    C_forward: int = DataConfig.C # Number of channels
    # B3 (Task 2.5): reassembly weighting for the object_big forward path.
    # 'central_mask' = unchanged default (binary center-mask via
    # reassemble_patches_position_real); 'probe' = |Probe|^2-weighted assembly;
    # 'uniform' = binary center mask via the merged probe helper
    # (reassemble_patches_position_real_probe, use_probe_weights=False).
    training_patch_weighting: Literal['central_mask', 'probe', 'uniform'] = 'central_mask'
    # B5 (Task 2.6): forward-model physics parameterization. 'amplitude' = the
    # unchanged fno-stable ProbeIllumination -> pad_and_diffract -> inv_scale
    # amplitude chain (default, byte-stable). 'rectangular_scaled' routes the
    # object patches through RectangularScaledDiffraction (main's analytic
    # real/imag intensity model with folded probe/physics scaling); it requires
    # real/imag-derived object patches (see PtychoPINN.__init__ fail-fast).
    physics_forward_mode: Literal['amplitude', 'rectangular_scaled'] = 'amplitude'
    # Whether RectangularScaledDiffraction's s1/s2 scale parameters are trainable
    # (only consulted when physics_forward_mode='rectangular_scaled').
    rect_s1s2_trainable: bool = True
    # PROBE-RANK-001 (design 2026-07-12 §3.3): explicit amplitude physics
    # gain. The banned flat (B, H, W) probe layout used to multiply the
    # predicted amplitude by the BATCH SIZE (an accidental, batch-size-
    # dependent gain that demonstrably conditioned amplitude-mode training).
    # The gain survives as this explicit, batch-size-independent constant:
    # applied ONCE, multiplicatively, to the predicted amplitude in the
    # amplitude-mode training forward (ForwardModel.forward); a training-
    # objective device only — inference/forward_predict never applies it.
    # rectangular_scaled/CI paths ignore it and their scaling contract
    # rejects non-1.0 values fail-closed
    # (ptycho_torch.scaling_contract.validate_amplitude_physics_gain).
    # Must be finite and > 0. Task 26 calibrated 16 only for the locked
    # legacy-amplitude N128/Run1084 dictionary reference regime; set it
    # explicitly there. The general default and rectangular/CI value stay 1.0.
    # Contract: docs/specs/spec-ptycho-torch-probe-layout.md.
    amplitude_physics_gain: float = 1.0

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
    strategy: Optional[str] = 'ddp' # Strategy for distributed training (e.g., 'ddp', 'ddp_spawn', 'auto', None)
    n_devices: Union[int, str] = 1 # Number of devices; "auto" resolves to available GPU count.

    # Framework
    framework: Literal['Default', 'Lightning'] = 'Lightning' #Training framework. Most of work don in PT was done in lightning
    orchestrator: Literal['Mlflow', 'Lightning'] = 'Mlflow'

    # Add other training-specific parameters here as needed, e.g.:
    learning_rate: float = 1e-3
    epochs: int = 50 #Default epochs number, will be overridden if multi-stage training is active at all
    batch_size: int = 16
    epochs_fine_tune: int = 0 #Default 0 fine-tune means no fine-tuning
    fine_tune_gamma: float = 0.1 #Scales base LR for fine-tuning
    scheduler: Literal['Default', 'Exponential', 'MultiStage', 'Adaptive', 'WarmupCosine', 'ReduceLROnPlateau'] = 'Default'
    lr_warmup_epochs: int = 0  # Number of warmup epochs for WarmupCosine scheduler
    lr_min_ratio: float = 0.1  # Minimum LR ratio for WarmupCosine scheduler (eta_min = base_lr * ratio)
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_min_lr: float = 5e-5
    plateau_threshold: float = 0.0
    num_workers: int = 4 #Dataloader workers
    accum_steps: int = 1 #Batch size accumulation, manually implemented for DDP
    gradient_clip_val: Union[float,None] = None #Gradient clip value
    gradient_clip_algorithm: str = 'norm'  # Gradient clipping algorithm: 'norm', 'value', or 'agc'
    optimizer: str = 'adam'  # Optimizer algorithm: 'adam', 'adamw', or 'sgd'
    momentum: float = 0.9  # SGD momentum (ignored for Adam/AdamW)
    weight_decay: float = 0.0  # Weight decay (L2 penalty)
    adam_beta1: float = 0.9  # Adam/AdamW beta1
    adam_beta2: float = 0.999  # Adam/AdamW beta2
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
    # Optional MAE path: scale each prediction sample so ||pred||_2 matches ||target||_2.
    torch_mae_pred_l2_match_target: bool = False

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
    patch_weighting: Literal['uniform', 'probe'] = 'probe'
    varpro_scaling: bool = True
    log_patch_stats: bool = False  # Emit patch stats during training/inference
    patch_stats_limit: Optional[int] = None  # Max number of batches to log

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
