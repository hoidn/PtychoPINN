"""
Canonical TensorFlow configuration baseline fixtures for parity testing.

This module defines explicit, non-default values for every spec-required field in
ModelConfig, TrainingConfig, and InferenceConfig to ensure the config bridge adapter
performs actual translation instead of relying on defaults.

Purpose: Phase A.A1 of B.B4 parity test plan
Spec References:
    - specs/ptychodus_api_spec.md:213-273 (§5.1-5.3 field tables)
    - ptycho/config/config.py:72-154 (dataclass definitions)

Design Rationale:
    - Avoid defaults: Explicitly set every field to catch missing translations
    - Use realistic values: Based on actual training/inference workflows
    - Mark spec section: Each field annotated with spec table row reference
"""

from pathlib import Path
from ptycho.config.config import ModelConfig, TrainingConfig, InferenceConfig


def get_canonical_model_config() -> ModelConfig:
    """
    Create ModelConfig with explicit non-default values for all spec-required fields.

    Field values chosen to differ from defaults to ensure bridge performs translation.

    Spec coverage: §5.1 (ModelConfig fields table)
    """
    return ModelConfig(
        # §5.1 row 1: Diffraction pattern size
        N=128,  # Default=64, using 128 to test non-default

        # §5.1 row 2: Group cardinality (gridsize²)
        gridsize=3,  # Default=2, using 3 to test non-default

        # §5.1 row 3: Filter width scaling
        n_filters_scale=2,  # Default=1, using 2 to test non-default

        # §5.1 row 4: Model type ('pinn' | 'supervised')
        model_type='pinn',  # Explicit choice (tests enum handling)

        # §5.1 row 5: Amplitude activation function
        amp_activation='swish',  # Default='sigmoid', using 'swish' to test mapping

        # §5.1 row 6: Whole-object stitching toggle
        object_big=False,  # Default=True, using False to test boolean translation

        # §5.1 row 7: Large-probe decoding toggle
        probe_big=False,  # Default=True, using False to test boolean translation

        # §5.1 row 8: Probe circular masking toggle
        probe_mask=True,  # Default=False, using True to test boolean translation

        # §5.1 row 9: Padded output toggle
        pad_object=False,  # Default=True, using False to test boolean translation

        # §5.1 row 10: Probe normalization factor
        probe_scale=2.0,  # Default=4.0, using 2.0 to test float translation

        # §5.1 row 11: ProbeIllumination smoothing parameter
        gaussian_smoothing_sigma=0.5,  # Default=0.0, using 0.5 to test non-zero
    )


def get_canonical_training_config(model: ModelConfig) -> TrainingConfig:
    """
    Create TrainingConfig with explicit non-default values for all spec-required fields.

    Args:
        model: ModelConfig instance to nest (required by TrainingConfig)

    Spec coverage: §5.2 (TrainingConfig fields table)
    """
    return TrainingConfig(
        # Nested model config (required)
        model=model,

        # §5.2 row 1: Training data source
        train_data_file=Path('/canonical/baseline/train_data.npz'),

        # §5.2 row 2: Optional test data source
        test_data_file=Path('/canonical/baseline/test_data.npz'),

        # §5.2 row 3: Batch size (legacy compatibility field)
        batch_size=32,  # Default=16, using 32 to test non-default

        # §5.2 row 4: Training epochs
        nepochs=100,  # Default=50, using 100 to test non-default

        # §5.2 row 5: Diffraction MAE loss weight
        mae_weight=0.3,  # Default=0.0, using 0.3 to test non-zero

        # §5.2 row 6: Poisson NLL loss weight
        nll_weight=0.7,  # Default=1.0, using 0.7 to test non-default

        # §5.2 row 7: Real-space MAE alignment weight
        realspace_mae_weight=0.05,  # Default=0.0, using 0.05 to test non-zero

        # §5.2 row 8: Real-space consistency weight
        realspace_weight=0.1,  # Default=0.0, using 0.1 to test non-zero

        # §5.2 row 9: Photon-count prior for scaling
        nphotons=5e8,  # Default=1e9, using 5e8 to test non-default

        # §5.2 row 10: Number of grouped samples (replaces n_images)
        n_groups=1024,  # Explicit (None by default), critical for grouping

        # §5.2 row 12: Independent subsampling count
        n_subsample=2048,  # Optional, using explicit value to test

        # §5.2 row 13: Subsampling RNG seed
        subsample_seed=42,  # Optional, using explicit seed for reproducibility test

        # §5.2 row 14: K-nearest-neighbor search width
        neighbor_count=5,  # Default=4, using 5 to test non-default

        # §5.2 row 15: Legacy simulation flag
        positions_provided=False,  # Default=True, using False to test boolean

        # §5.2 row 16: Joint probe optimization toggle
        probe_trainable=True,  # Default=False, using True to test boolean

        # §5.2 row 17: Learnable intensity scaling toggle
        intensity_scale_trainable=True,  # Default=False, using True to test boolean

        # §5.2 row 18: Output directory path
        output_dir=Path('/canonical/baseline/training_outputs'),

        # §5.2 row 19: Deterministic sequential grouping toggle
        sequential_sampling=True,  # Default=False, using True to test boolean
    )


def get_canonical_inference_config(model: ModelConfig) -> InferenceConfig:
    """
    Create InferenceConfig with explicit non-default values for all spec-required fields.

    Args:
        model: ModelConfig instance to nest (required by InferenceConfig)

    Spec coverage: §5.3 (InferenceConfig fields table)
    """
    return InferenceConfig(
        # Nested model config (required)
        model=model,

        # §5.3 row 1: Trained model directory
        model_path=Path('/canonical/baseline/model_directory'),

        # §5.3 row 2: Inference data source
        test_data_file=Path('/canonical/baseline/inference_data.npz'),

        # §5.3 row 3: Number of grouped samples
        n_groups=512,  # Explicit (None by default), critical for inference

        # §5.3 row 5: Independent subsampling count
        n_subsample=1024,  # Optional, using explicit value to test

        # §5.3 row 6: Subsampling RNG seed
        subsample_seed=99,  # Optional, using different seed than training

        # §5.3 row 7: K-nearest-neighbor search width
        neighbor_count=6,  # Default=4, using 6 to test non-default

        # §5.3 row 8: Verbose debug logging toggle
        debug=True,  # Default=False, using True to test boolean

        # §5.3 row 9: Output directory path
        output_dir=Path('/canonical/baseline/inference_outputs'),
    )


def get_all_canonical_configs():
    """
    Helper to instantiate all three canonical configs with nesting relationships.

    Returns:
        Tuple[ModelConfig, TrainingConfig, InferenceConfig]: Canonical baseline configs
    """
    model = get_canonical_model_config()
    training = get_canonical_training_config(model)
    inference = get_canonical_inference_config(model)
    return model, training, inference


# Field inventory for test parameterization (maps to field_matrix.md)
SPEC_REQUIRED_FIELDS = {
    'ModelConfig': [
        'N', 'gridsize', 'n_filters_scale', 'model_type', 'amp_activation',
        'object_big', 'probe_big', 'probe_mask', 'pad_object', 'probe_scale',
        'gaussian_smoothing_sigma'
    ],
    'TrainingConfig': [
        'train_data_file', 'test_data_file', 'batch_size', 'nepochs',
        'mae_weight', 'nll_weight', 'realspace_mae_weight', 'realspace_weight',
        'nphotons', 'n_groups', 'n_subsample', 'subsample_seed', 'neighbor_count',
        'positions_provided', 'probe_trainable', 'intensity_scale_trainable',
        'output_dir', 'sequential_sampling'
    ],
    'InferenceConfig': [
        'model_path', 'test_data_file', 'n_groups', 'n_subsample', 'subsample_seed',
        'neighbor_count', 'debug', 'output_dir'
    ]
}


if __name__ == '__main__':
    # Smoke test: Verify all configs instantiate successfully
    model, training, inference = get_all_canonical_configs()
    print(f"✓ Canonical ModelConfig: N={model.N}, gridsize={model.gridsize}, model_type={model.model_type}")
    print(f"✓ Canonical TrainingConfig: nepochs={training.nepochs}, n_groups={training.n_groups}")
    print(f"✓ Canonical InferenceConfig: n_groups={inference.n_groups}, debug={inference.debug}")
    print(f"\nTotal spec-required fields covered:")
    for config_name, fields in SPEC_REQUIRED_FIELDS.items():
        print(f"  {config_name}: {len(fields)} fields")
