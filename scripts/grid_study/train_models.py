"""Model training wrappers for grid-based ptychography comparison study.

Provides unified training interface for baseline (supervised U-Net) and
PtychoPINN (physics-informed) models with proper params.cfg configuration.

References:
    - ptycho/baselines.py - Baseline U-Net implementation
    - ptycho/model.py - PtychoPINN model with create_model_with_gridsize()
    - ptycho/train_pinn.py - PINN training workflow
"""
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
from dataclasses import dataclass

import numpy as np
import tensorflow as tf

# Suppress TF warnings for cleaner output
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from ptycho import params as p


@dataclass
class TrainingResult:
    """Container for training results."""
    model: tf.keras.Model
    history: Any
    config: Dict[str, Any]
    model_type: str  # 'baseline' or 'pinn'


def setup_training_params(
    N: int,
    gridsize: int = 1,
    nepochs: int = 30,
    batch_size: int = 16,
    nphotons: float = 1e9,
    intensity_scale: Optional[float] = None,
    **extra_params
) -> Dict[str, Any]:
    """
    Set up params.cfg for training.

    Follows CONFIG-001: params.cfg must be set before calling legacy modules.

    Returns:
        Dictionary of configuration values (also sets params.cfg)
    """
    config = {
        'N': N,
        'gridsize': gridsize,
        'nepochs': nepochs,
        'batch_size': batch_size,
        'nphotons': nphotons,
        'offset': 4,  # default inner offset
        **extra_params
    }

    if intensity_scale is not None:
        config['intensity_scale'] = intensity_scale

    # Set all values in params.cfg
    for key, value in config.items():
        p.cfg[key] = value

    return config


def train_baseline(
    X_train: np.ndarray,
    Y_I_train: np.ndarray,
    Y_phi_train: np.ndarray,
    nepochs: int = 30,
    batch_size: int = 16,
    **config_overrides
) -> TrainingResult:
    """
    Train baseline supervised U-Net model.

    The baseline model directly maps diffraction patterns to amplitude/phase
    without physics constraints. Uses c=1 (single channel).

    Args:
        X_train: Diffraction patterns (batch, N, N, 1)
        Y_I_train: Ground truth amplitude (batch, N, N, 1)
        Y_phi_train: Ground truth phase (batch, N, N, 1)
        nepochs: Number of training epochs
        batch_size: Training batch size
        **config_overrides: Override default config values

    Returns:
        TrainingResult containing trained model and history
    """
    from ptycho import baselines

    N = X_train.shape[1]

    # Ensure single channel for baseline (c=1 hardcoded in baselines.py)
    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]
    if Y_I_train.ndim == 3:
        Y_I_train = Y_I_train[..., np.newaxis]
    if Y_phi_train.ndim == 3:
        Y_phi_train = Y_phi_train[..., np.newaxis]

    # Ensure single channel
    if X_train.shape[-1] > 1:
        print(f"Warning: Baseline expects single channel, got {X_train.shape[-1]}. Using first channel.")
        X_train = X_train[..., :1]
        Y_I_train = Y_I_train[..., :1]
        Y_phi_train = Y_phi_train[..., :1]

    # Setup params.cfg
    config = setup_training_params(
        N=N,
        gridsize=1,  # baseline always uses gridsize=1
        nepochs=nepochs,
        batch_size=batch_size,
        **config_overrides
    )

    print("=" * 60)
    print("Training Baseline Model (Supervised U-Net)")
    print("=" * 60)
    print(f"  Input shape: {X_train.shape}")
    print(f"  Epochs: {nepochs}, Batch size: {batch_size}")

    # Build and train
    model = baselines.build_model(X_train, Y_I_train, Y_phi_train)
    trained_model, history = baselines.train(X_train, Y_I_train, Y_phi_train, model)

    print("Baseline training complete.")

    return TrainingResult(
        model=trained_model,
        history=history,
        config=config,
        model_type='baseline'
    )


def train_pinn(
    X_train: np.ndarray,
    Y_I_train: np.ndarray,
    Y_phi_train: np.ndarray,
    coords_train: np.ndarray,
    probe: np.ndarray,
    intensity_scale: float,
    nepochs: int = 30,
    batch_size: int = 16,
    gridsize: int = 1,
    **config_overrides
) -> TrainingResult:
    """
    Train PtychoPINN physics-informed model.

    PtychoPINN uses differentiable forward modeling with Poisson NLL loss
    to enforce physical consistency without requiring ground truth.

    Args:
        X_train: Diffraction patterns (batch, N, N, gridsize^2)
        Y_I_train: Ground truth amplitude (for validation only)
        Y_phi_train: Ground truth phase (for validation only)
        coords_train: Coordinate information for patches
        probe: Complex probe array (N, N)
        intensity_scale: Intensity normalization factor
        nepochs: Number of training epochs
        batch_size: Training batch size
        gridsize: Grid size for patch grouping
        **config_overrides: Override default config values

    Returns:
        TrainingResult containing trained model and history
    """
    from ptycho import model as pinn_model
    from ptycho import probe as probe_module
    from ptycho.loader import PtychoDataContainer

    N = X_train.shape[1]

    # Ensure correct channel count for gridsize
    expected_channels = gridsize ** 2
    if X_train.shape[-1] != expected_channels:
        if X_train.shape[-1] == 1 and expected_channels == 1:
            pass  # OK
        else:
            raise ValueError(
                f"X_train has {X_train.shape[-1]} channels, "
                f"expected {expected_channels} for gridsize={gridsize}"
            )

    # Setup params.cfg with PINN-specific params
    config = setup_training_params(
        N=N,
        gridsize=gridsize,
        nepochs=nepochs,
        batch_size=batch_size,
        intensity_scale=intensity_scale,
        **config_overrides
    )

    # Set probe in global state (required by PINN model)
    probe_module.set_probe_guess(None, probe)
    p.cfg['probe'] = probe

    print("=" * 60)
    print("Training PtychoPINN Model (Physics-Informed)")
    print("=" * 60)
    print(f"  Input shape: {X_train.shape}")
    print(f"  Gridsize: {gridsize}, N: {N}")
    print(f"  Intensity scale: {intensity_scale:.6f}")
    print(f"  Epochs: {nepochs}, Batch size: {batch_size}")

    # Clear session and create fresh model (MODULE-SINGLETON-001)
    tf.keras.backend.clear_session()

    # Create model with explicit gridsize
    model_instance, diffraction_to_obj = pinn_model.create_model_with_gridsize(
        gridsize=gridsize,
        N=N
    )

    # Compile model
    model_instance = pinn_model.compile_model(model_instance)

    # Create nominal coords if needed
    if coords_train is None:
        # Create zero coords for gridsize=1
        coords_train = np.zeros((X_train.shape[0], 1, 2, gridsize**2), dtype=np.float32)

    # Create PtychoDataContainer for training
    # Convert arrays to TensorFlow tensors
    X_tf = tf.constant(X_train, dtype=tf.float32)
    Y_I_tf = tf.constant(Y_I_train, dtype=tf.float32) if Y_I_train is not None else None
    Y_phi_tf = tf.constant(Y_phi_train, dtype=tf.float32) if Y_phi_train is not None else None
    coords_tf = tf.constant(coords_train, dtype=tf.float32)
    probe_tf = tf.constant(probe, dtype=tf.complex64)

    # Create Y (complex ground truth) if we have amplitude and phase
    if Y_I_tf is not None and Y_phi_tf is not None:
        # Create complex tensor: amplitude * exp(i * phase)
        phase_complex = tf.complex(tf.cos(Y_phi_tf), tf.sin(Y_phi_tf))
        Y_tf = tf.cast(Y_I_tf, tf.complex64) * phase_complex
    else:
        Y_tf = None

    train_data = PtychoDataContainer(
        X=X_tf,
        Y=Y_tf,
        Y_I=Y_I_tf,
        Y_phi=Y_phi_tf,
        coords_nominal=coords_tf,
        coords_true=coords_tf,
        probe=probe_tf,
        norm_Y_I=tf.constant(1.0, dtype=tf.float32),
    )

    # Train using model.train which expects PtychoDataContainer
    history = pinn_model.train(nepochs, train_data, model_instance=model_instance)

    print("PtychoPINN training complete.")

    return TrainingResult(
        model=model_instance,
        history=history,
        config=config,
        model_type='pinn'
    )


def train_pinn_simple(
    X_train: np.ndarray,
    probe: np.ndarray,
    intensity_scale: float,
    nepochs: int = 30,
    batch_size: int = 16,
    Y_I_train: Optional[np.ndarray] = None,
    Y_phi_train: Optional[np.ndarray] = None,
    **config_overrides
) -> TrainingResult:
    """
    Simplified PtychoPINN training for gridsize=1.

    This is a convenience wrapper that handles coordinate creation
    and simplifies the interface for single-patch training.

    Args:
        X_train: Diffraction patterns (batch, N, N, 1)
        probe: Complex probe array (N, N)
        intensity_scale: Intensity normalization factor
        nepochs: Number of training epochs
        batch_size: Training batch size
        Y_I_train: Optional ground truth amplitude (for validation)
        Y_phi_train: Optional ground truth phase (for validation)
        **config_overrides: Override default config values

    Returns:
        TrainingResult containing trained model and history
    """
    from ptycho import model as pinn_model
    from ptycho import probe as probe_module
    from ptycho.loader import PtychoDataContainer

    N = X_train.shape[1]
    gridsize = 1

    # Ensure single channel
    if X_train.ndim == 3:
        X_train = X_train[..., np.newaxis]

    # Setup params.cfg
    config = setup_training_params(
        N=N,
        gridsize=gridsize,
        nepochs=nepochs,
        batch_size=batch_size,
        intensity_scale=intensity_scale,
        **config_overrides
    )

    # Set probe in global state
    probe_module.set_probe_guess(None, probe)
    p.cfg['probe'] = probe

    print("=" * 60)
    print("Training PtychoPINN Model (Simplified, gridsize=1)")
    print("=" * 60)
    print(f"  Input shape: {X_train.shape}")
    print(f"  Intensity scale: {intensity_scale:.6f}")
    print(f"  Epochs: {nepochs}, Batch size: {batch_size}")

    # Clear session and create fresh model
    tf.keras.backend.clear_session()

    # Create and compile model
    model_instance, diffraction_to_obj = pinn_model.create_model_with_gridsize(
        gridsize=gridsize,
        N=N
    )
    model_instance = pinn_model.compile_model(model_instance)

    # Create zero coords for gridsize=1
    coords = np.zeros((X_train.shape[0], 1, 2, 1), dtype=np.float32)

    # Convert to TensorFlow tensors
    X_tf = tf.constant(X_train, dtype=tf.float32)
    coords_tf = tf.constant(coords, dtype=tf.float32)
    probe_tf = tf.constant(probe, dtype=tf.complex64)

    # Handle optional ground truth
    if Y_I_train is not None:
        if Y_I_train.ndim == 3:
            Y_I_train = Y_I_train[..., np.newaxis]
        Y_I_tf = tf.constant(Y_I_train, dtype=tf.float32)
    else:
        Y_I_tf = None

    if Y_phi_train is not None:
        if Y_phi_train.ndim == 3:
            Y_phi_train = Y_phi_train[..., np.newaxis]
        Y_phi_tf = tf.constant(Y_phi_train, dtype=tf.float32)
    else:
        Y_phi_tf = None

    # Create Y (complex ground truth) if we have amplitude and phase
    if Y_I_tf is not None and Y_phi_tf is not None:
        # Create complex tensor: amplitude * exp(i * phase)
        phase_complex = tf.complex(tf.cos(Y_phi_tf), tf.sin(Y_phi_tf))
        Y_tf = tf.cast(Y_I_tf, tf.complex64) * phase_complex
    else:
        Y_tf = None

    # Create PtychoDataContainer
    train_data = PtychoDataContainer(
        X=X_tf,
        Y=Y_tf,
        Y_I=Y_I_tf,
        Y_phi=Y_phi_tf,
        coords_nominal=coords_tf,
        coords_true=coords_tf,
        probe=probe_tf,
        norm_Y_I=tf.constant(1.0, dtype=tf.float32),
    )

    # Train using model.train
    history = pinn_model.train(nepochs, train_data, model_instance=model_instance)

    print("PtychoPINN training complete.")

    return TrainingResult(
        model=model_instance,
        history=history,
        config=config,
        model_type='pinn'
    )


def save_model(result: TrainingResult, output_dir: Path, name: str) -> Path:
    """
    Save trained model to disk.

    Args:
        result: TrainingResult from training
        output_dir: Directory to save model
        name: Model name (e.g., 'baseline_n64', 'pinn_n128')

    Returns:
        Path to saved model
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / f"{name}.keras"
    result.model.save(model_path)
    print(f"Model saved to: {model_path}")

    # Also save config
    import json
    config_path = output_dir / f"{name}_config.json"
    # Convert numpy types to Python types for JSON serialization
    config_json = {}
    for k, v in result.config.items():
        if isinstance(v, np.ndarray):
            continue  # Skip arrays
        elif isinstance(v, (np.integer, np.floating)):
            config_json[k] = float(v)
        else:
            try:
                json.dumps(v)  # Test if serializable
                config_json[k] = v
            except (TypeError, ValueError):
                continue

    with open(config_path, 'w') as f:
        json.dump(config_json, f, indent=2)

    return model_path


if __name__ == "__main__":
    # Test training with small dataset
    from probe_utils import get_probe_for_N
    from grid_data_generator import generate_train_test_data

    print("=== Training Models Test ===\n")

    # Get probe and generate small dataset
    probe_64 = get_probe_for_N(64)

    train_data, test_data = generate_train_test_data(
        probe=probe_64,
        n_train_objects=1,
        n_test_objects=1,
        outer_offset=12,
    )

    # Test baseline training (very short)
    print("\n--- Testing Baseline Training ---")
    baseline_result = train_baseline(
        X_train=train_data.X[:100],
        Y_I_train=train_data.Y_I[:100],
        Y_phi_train=train_data.Y_phi[:100],
        nepochs=2,
        batch_size=16,
    )
    print(f"Baseline model output shapes: {[o.shape for o in baseline_result.model.outputs]}")

    print("\n--- Training test complete ---")
