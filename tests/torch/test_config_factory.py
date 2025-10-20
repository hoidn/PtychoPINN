"""
RED Phase Tests for PyTorch Config Factory Functions

This test module encodes the expected behavior of the configuration factory functions
defined in ptycho_torch/config_factory.py. These tests are written BEFORE implementation
(TDD RED phase) and will fail with NotImplementedError until Phase B3.a implementation.

Test Strategy:
    Phase B2 (RED): All tests fail with NotImplementedError from factory stubs
    Phase B3 (GREEN): Implementation added, tests pass

Test Coverage:
    1. Factory Returns Correct Payload Structure
    2. Config Bridge Integration (TF dataclass translation)
    3. params.cfg Population (CONFIG-001 compliance)
    4. Override Precedence Rules
    5. Validation Errors (missing n_groups, invalid paths)

Design Reference:
    plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/factory_design.md §5

Override Matrix Reference:
    plans/active/ADR-003-BACKEND-API/reports/2025-10-19T232336Z/phase_b_factories/override_matrix.md
"""

import pytest
from pathlib import Path
from dataclasses import is_dataclass
import tempfile
import shutil

# Factory functions under test (stubs in Phase B2)
from ptycho_torch.config_factory import (
    create_training_payload,
    create_inference_payload,
    infer_probe_size,
    populate_legacy_params,
    TrainingPayload,
    InferencePayload,
)

# Config dataclasses for assertions
from ptycho.config.config import (
    ModelConfig as TFModelConfig,
    TrainingConfig as TFTrainingConfig,
    InferenceConfig as TFInferenceConfig,
)
from ptycho_torch.config_params import (
    DataConfig as PTDataConfig,
    ModelConfig as PTModelConfig,
    TrainingConfig as PTTrainingConfig,
    InferenceConfig as PTInferenceConfig,
)

# For params.cfg validation
import ptycho.params


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def temp_output_dir():
    """Temporary directory for factory outputs (cleaned up after test)."""
    tmpdir = Path(tempfile.mkdtemp(prefix="factory_test_"))
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def mock_train_npz(tmp_path):
    """
    Mock training NPZ file with minimal DATA-001 compliant fields.

    Creates a temporary NPZ with required keys (diffraction, probeGuess, xcoords, ycoords)
    for factory tests. Probe size N=64 for deterministic inference.
    """
    import numpy as np

    N = 64
    n_images = 100
    M = 256  # Object size (larger than probe)

    npz_path = tmp_path / "mock_train.npz"
    np.savez(
        npz_path,
        diffraction=np.random.rand(n_images, N, N).astype(np.float32),  # Amplitude, not intensity
        probeGuess=np.random.rand(N, N).astype(np.complex64),  # Square probe
        objectGuess=np.random.rand(M, M).astype(np.complex64),  # Larger than probe
        xcoords=np.linspace(0, 1, n_images).astype(np.float64),
        ycoords=np.linspace(0, 1, n_images).astype(np.float64),
        scan_index=np.arange(n_images).astype(np.int32),
    )
    return npz_path


@pytest.fixture
def mock_test_npz(tmp_path):
    """Mock test NPZ file (smaller than training for faster tests)."""
    import numpy as np

    N = 64
    n_images = 20  # Smaller test set
    M = 256

    npz_path = tmp_path / "mock_test.npz"
    np.savez(
        npz_path,
        diffraction=np.random.rand(n_images, N, N).astype(np.float32),
        probeGuess=np.random.rand(N, N).astype(np.complex64),
        objectGuess=np.random.rand(M, M).astype(np.complex64),
        xcoords=np.linspace(0, 1, n_images).astype(np.float64),
        ycoords=np.linspace(0, 1, n_images).astype(np.float64),
        scan_index=np.arange(n_images).astype(np.int32),
    )
    return npz_path


@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    """Mock model checkpoint directory with wts.h5.zip."""
    checkpoint_dir = tmp_path / "mock_checkpoint"
    checkpoint_dir.mkdir()

    # Create empty wts.h5.zip (factory only checks existence for validation)
    checkpoint_file = checkpoint_dir / "wts.h5.zip"
    checkpoint_file.touch()

    return checkpoint_dir


# ============================================================================
# Test Category 1: Factory Returns Correct Payload Structure
# ============================================================================

class TestTrainingPayloadStructure:
    """
    Verify create_training_payload() returns TrainingPayload with all required fields.

    Expected behavior (Phase B3):
        - Returns TrainingPayload dataclass instance
        - Contains tf_training_config (TFTrainingConfig)
        - Contains pt_data_config (PTDataConfig)
        - Contains pt_model_config (PTModelConfig)
        - Contains pt_training_config (PTTrainingConfig)
        - Contains execution_config (PyTorchExecutionConfig or None)
        - Contains overrides_applied dict (audit trail)

    RED phase behavior:
        - Raises NotImplementedError from factory stub
    """

    def test_training_payload_returns_dataclass(self, mock_train_npz, temp_output_dir):
        """Factory returns TrainingPayload dataclass instance."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'batch_size': 4},
        )
        # GREEN phase assertions (will run after implementation):
        assert is_dataclass(payload)
        assert isinstance(payload, TrainingPayload)

    def test_training_payload_contains_tf_config(self, mock_train_npz, temp_output_dir):
        """Payload contains TensorFlow TrainingConfig instance."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
        )
        # GREEN phase:
        assert isinstance(payload.tf_training_config, TFTrainingConfig)

    def test_training_payload_contains_pytorch_configs(self, mock_train_npz, temp_output_dir):
        """Payload contains all three PyTorch singleton config instances."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
        )
        # GREEN phase assertions:
        assert isinstance(payload.pt_data_config, PTDataConfig)
        assert isinstance(payload.pt_model_config, PTModelConfig)
        assert isinstance(payload.pt_training_config, PTTrainingConfig)

    def test_training_payload_contains_overrides_dict(self, mock_train_npz, temp_output_dir):
        """Payload includes audit trail of applied overrides."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'batch_size': 8},
        )
        # GREEN phase assertions:
        assert 'n_groups' in payload.overrides_applied
        assert payload.overrides_applied['n_groups'] == 512
        assert payload.overrides_applied['batch_size'] == 8


class TestInferencePayloadStructure:
    """Verify create_inference_payload() returns InferencePayload with all required fields."""

    def test_inference_payload_returns_dataclass(self, mock_checkpoint_dir, mock_test_npz, temp_output_dir):
        """Factory returns InferencePayload dataclass instance."""
        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 128},
        )
        # GREEN phase:
        assert isinstance(payload, InferencePayload)

    def test_inference_payload_contains_tf_config(self, mock_checkpoint_dir, mock_test_npz, temp_output_dir):
        """Payload contains TensorFlow InferenceConfig instance."""
        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 128},
        )
        # GREEN phase:
        assert isinstance(payload.tf_inference_config, TFInferenceConfig)

    def test_inference_payload_contains_pytorch_configs(self, mock_checkpoint_dir, mock_test_npz, temp_output_dir):
        """Payload contains PyTorch inference config instances."""
        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 128},
        )
        # GREEN phase assertions:
        assert isinstance(payload.pt_data_config, PTDataConfig)
        assert isinstance(payload.pt_inference_config, PTInferenceConfig)


# ============================================================================
# Test Category 2: Config Bridge Integration
# ============================================================================

class TestConfigBridgeTranslation:
    """
    Verify factory delegates TensorFlow dataclass translation to config_bridge.

    Expected behavior (Phase B3):
        - PyTorch grid_size tuple → TensorFlow gridsize int conversion
        - PyTorch epochs → TensorFlow nepochs conversion
        - PyTorch K → TensorFlow neighbor_count conversion
        - Activation name normalization (silu → swish)
        - All config_bridge.py transformations applied correctly
    """

    def test_grid_size_tuple_to_gridsize_int(self, mock_train_npz, temp_output_dir):
        """Factory converts grid_size (2, 2) → gridsize 2 via bridge."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'gridsize': 2},
        )
        # GREEN phase assertions:
        assert payload.pt_data_config.grid_size == (2, 2)  # PyTorch tuple
        assert payload.tf_training_config.model.gridsize == 2  # TensorFlow int

    def test_epochs_to_nepochs_conversion(self, mock_train_npz, temp_output_dir):
        """Factory maps epochs → nepochs via bridge."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'max_epochs': 20},
        )
        # GREEN phase assertions:
        assert payload.pt_training_config.epochs == 20  # PyTorch naming
        assert payload.tf_training_config.nepochs == 20  # TensorFlow naming

    def test_k_to_neighbor_count_conversion(self, mock_train_npz, temp_output_dir):
        """Factory maps K → neighbor_count via bridge."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'neighbor_count': 7},
        )
        # GREEN phase assertions:
        assert payload.pt_data_config.K == 7  # PyTorch K
        assert payload.tf_training_config.neighbor_count == 7  # TensorFlow naming


# ============================================================================
# Test Category 3: params.cfg Population (CONFIG-001 Compliance)
# ============================================================================

class TestLegacyParamsPopulation:
    """
    Verify factory calls update_legacy_dict to populate params.cfg.

    Critical requirement (CONFIG-001): params.cfg MUST be populated before
    any data loading or model construction. Factory is responsible for this
    checkpoint via populate_legacy_params() helper.
    """

    def test_factory_populates_params_cfg(self, mock_train_npz, temp_output_dir):
        """Factory updates ptycho.params.cfg with config values."""
        # Clear params.cfg before test
        ptycho.params.cfg.clear()

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'gridsize': 2},
        )

        # GREEN phase assertions:
        assert ptycho.params.cfg['gridsize'] == 2
        assert ptycho.params.cfg['N'] == 64  # Inferred from NPZ
        assert ptycho.params.cfg['n_groups'] == 512

    def test_populate_legacy_params_helper(self, mock_train_npz, temp_output_dir):
        """populate_legacy_params() wrapper calls update_legacy_dict."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Construct minimal TF config
        tf_config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2),
            train_data_file=mock_train_npz,
            n_groups=512,
        )

        # Clear params.cfg
        ptycho.params.cfg.clear()

        # Call factory helper
        populate_legacy_params(tf_config)

        # GREEN phase:
        assert ptycho.params.cfg['gridsize'] == 2


# ============================================================================
# Test Category 4: Override Precedence Rules
# ============================================================================

class TestOverridePrecedence:
    """
    Verify override precedence rules per override_matrix.md §4.

    Priority order (highest to lowest):
        1. Explicit overrides dict
        2. Execution config fields
        3. CLI argument defaults
        4. PyTorch config defaults
        5. TensorFlow config defaults
    """

    def test_override_dict_wins_over_defaults(self, mock_train_npz, temp_output_dir):
        """Overrides dict has highest precedence."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 1024, 'batch_size': 16},
        )
        # GREEN phase assertions:
        assert payload.tf_training_config.n_groups == 1024  # Override wins
        assert payload.tf_training_config.batch_size == 16

    def test_probe_size_override_wins_over_inference(self, mock_train_npz, temp_output_dir):
        """Explicit N override takes precedence over inferred probe size."""
        # NPZ has N=64 probe, but override specifies N=128
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'N': 128},
        )
        # GREEN phase:
        assert payload.tf_training_config.model.N == 128


# ============================================================================
# Test Category 5: Validation Errors
# ============================================================================

class TestFactoryValidation:
    """
    Verify factory raises appropriate errors for invalid inputs.

    Critical validations:
        - train_data_file / test_data_file path existence
        - n_groups required in overrides (no default)
        - model_path must contain wts.h5.zip
        - NPZ field validation (diffraction, probeGuess present)
    """

    def test_missing_n_groups_raises_error(self, mock_train_npz, temp_output_dir):
        """Factory raises ValueError if n_groups missing from overrides."""
        # Omit n_groups (required field)
        with pytest.raises(ValueError, match="n_groups is required"):
            payload = create_training_payload(
                train_data_file=mock_train_npz,
                output_dir=temp_output_dir,
                overrides={},  # Missing n_groups!
            )

    def test_nonexistent_train_data_file_raises_error(self, temp_output_dir):
        """Factory raises FileNotFoundError for missing train_data_file."""
        with pytest.raises(FileNotFoundError, match="Training data file not found"):
            payload = create_training_payload(
                train_data_file=Path("/nonexistent/train.npz"),
                output_dir=temp_output_dir,
                overrides={'n_groups': 512},
            )

    def test_missing_checkpoint_raises_error(self, mock_test_npz, temp_output_dir):
        """Factory raises ValueError if model_path missing wts.h5.zip."""
        bad_checkpoint_dir = temp_output_dir / "no_checkpoint"
        bad_checkpoint_dir.mkdir()

        with pytest.raises(ValueError, match="Model archive not found"):
            payload = create_inference_payload(
                model_path=bad_checkpoint_dir,
                test_data_file=mock_test_npz,
                output_dir=temp_output_dir,
                overrides={'n_groups': 128},
            )


# ============================================================================
# Test Category 6: Probe Size Inference Helper
# ============================================================================

class TestProbeSizeInference:
    """Verify infer_probe_size() extracts N from NPZ probeGuess."""

    def test_infer_probe_size_from_npz(self, mock_train_npz):
        """Helper extracts probe size from NPZ metadata."""
        N = infer_probe_size(mock_train_npz)
        # GREEN phase:
        assert N == 64  # Mock fixture has 64x64 probe

    def test_infer_probe_size_missing_file_fallback(self):
        """Helper returns fallback N=64 for missing NPZ file."""
        N = infer_probe_size(Path("/nonexistent/data.npz"))
        # GREEN phase:
        assert N == 64  # Fallback per design decision


# ============================================================================
# Test Execution Summary (for RED log capture)
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
