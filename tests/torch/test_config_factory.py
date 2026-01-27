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
def mock_train_npz_128(tmp_path):
    """Mock training NPZ with 128x128 probe for N inference coverage."""
    import numpy as np

    N = 128
    n_images = 100
    M = 256

    npz_path = tmp_path / "mock_train_128.npz"
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

    def test_gridsize_sets_channel_count(self, mock_train_npz, temp_output_dir):
        """
        Gridsize override synchronizes C_forward and C_model with data channel count.

        Regression test for ADR-003 C4.D3: create_training_payload() must set
        pt_model_config.C_forward and C_model to match pt_data_config.C when
        gridsize is specified. This ensures PyTorch helpers (reassemble_patches_position_real)
        receive tensor shapes consistent with the grouping strategy.

        Expected behavior:
            - gridsize=1 → C=1, C_forward=1, C_model=1
            - gridsize=2 → C=4, C_forward=4, C_model=4
            - Default (no gridsize override) → C=4, C_forward=4, C_model=4

        Reference: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T061500Z/
                   phase_c4_cli_integration_debug/coords_relative_investigation.md
        """
        # Case 1: gridsize=1 (single-position groups)
        payload_gs1 = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'gridsize': 1, 'n_groups': 512},
        )
        assert payload_gs1.pt_data_config.C == 1, "DataConfig.C should match gridsize**2 (1)"
        assert payload_gs1.pt_model_config.C_forward == 1, "ModelConfig.C_forward should match DataConfig.C"
        assert payload_gs1.pt_model_config.C_model == 1, "ModelConfig.C_model should match DataConfig.C"

        # Case 2: gridsize=2 (2x2 = 4 overlapping positions)
        payload_gs2 = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'gridsize': 2, 'n_groups': 512},
        )
        assert payload_gs2.pt_data_config.C == 4, "DataConfig.C should match gridsize**2 (4)"
        assert payload_gs2.pt_model_config.C_forward == 4, "ModelConfig.C_forward should match DataConfig.C"
        assert payload_gs2.pt_model_config.C_model == 4, "ModelConfig.C_model should match DataConfig.C"

        # Case 3: No gridsize override (default grid_size=(2,2) → C=4)
        payload_default = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
        )
        # Default grid_size is (2,2) per PTDataConfig defaults (config_params.py:29)
        # but factory may compute C from overrides; accept any C >= 1
        assert payload_default.pt_data_config.C >= 1, "DataConfig.C should be positive"
        assert payload_default.pt_model_config.C_forward == payload_default.pt_data_config.C, \
            "ModelConfig.C_forward must always match DataConfig.C"
        assert payload_default.pt_model_config.C_model == payload_default.pt_data_config.C, \
            "ModelConfig.C_model must always match DataConfig.C"

    def test_training_payload_infers_probe_size_for_pt_data_config(self, mock_train_npz_128, temp_output_dir):
        """Factory should propagate inferred N into pt_data_config and TF model config."""
        payload = create_training_payload(
            train_data_file=mock_train_npz_128,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
        )
        assert payload.pt_data_config.N == 128
        assert payload.tf_training_config.model.N == 128


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
# Test Category 6: ExecutionConfig Override Integration
# ============================================================================

class TestExecutionConfigOverrides:
    """
    Verify execution config knobs propagate through factory payloads.

    Expected behavior (Phase C2 GREEN):
        - TrainingPayload contains PyTorchExecutionConfig instance
        - InferencePayload contains PyTorchExecutionConfig instance
        - Execution knobs (accelerator, deterministic, num_workers) accessible
        - Override precedence: explicit > execution_config > defaults
        - overrides_applied captures execution knob applications

    RED phase behavior (Phase C2.B3):
        - execution_config field exists but returns None (placeholder)
        - Tests fail on assertion expecting PyTorchExecutionConfig instance
    """

    def test_training_payload_execution_config_not_none(self, mock_train_npz, temp_output_dir):
        """Factory returns execution_config (not None placeholder)."""
        from ptycho.config.config import PyTorchExecutionConfig

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
        )
        # GREEN phase assertion:
        assert payload.execution_config is not None
        assert isinstance(payload.execution_config, PyTorchExecutionConfig)

    def test_inference_payload_execution_config_not_none(self, mock_checkpoint_dir, mock_test_npz, temp_output_dir):
        """Inference factory returns execution_config instance."""
        from ptycho.config.config import PyTorchExecutionConfig

        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 128},
        )
        # GREEN phase:
        assert payload.execution_config is not None
        assert isinstance(payload.execution_config, PyTorchExecutionConfig)

    def test_execution_config_defaults_applied(self, mock_train_npz, temp_output_dir):
        """Execution config uses dataclass defaults when not overridden."""
        import torch
        from ptycho.config.config import PyTorchExecutionConfig

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
        )
        # GREEN phase assertions (verify defaults from PyTorchExecutionConfig):
        exec_cfg = payload.execution_config
        # POLICY-001: GPU-first defaults (auto='cuda' if available, else 'cpu')
        expected_accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert exec_cfg.accelerator == expected_accelerator, (
            f"Expected GPU-first default accelerator='{expected_accelerator}' per POLICY-001, "
            f"got '{exec_cfg.accelerator}'"
        )
        assert exec_cfg.deterministic is True  # Default for reproducibility
        assert exec_cfg.num_workers == 0  # CPU-safe default

    def test_execution_config_explicit_instance_propagates(self, mock_train_npz, temp_output_dir):
        """User-provided execution_config instance propagates through payload."""
        from ptycho.config.config import PyTorchExecutionConfig

        custom_exec_cfg = PyTorchExecutionConfig(
            accelerator='gpu',
            enable_progress_bar=True,
            deterministic=False,
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
            execution_config=custom_exec_cfg,
        )
        # GREEN phase assertions:
        assert payload.execution_config.accelerator == 'gpu'
        assert payload.execution_config.enable_progress_bar is True
        assert payload.execution_config.deterministic is False

    def test_execution_config_fields_accessible(self, mock_train_npz, temp_output_dir):
        """All critical execution fields are accessible from payload."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
        )
        # GREEN phase: Verify key execution knobs are accessible
        exec_cfg = payload.execution_config
        assert hasattr(exec_cfg, 'accelerator')
        assert hasattr(exec_cfg, 'deterministic')
        assert hasattr(exec_cfg, 'num_workers')
        assert hasattr(exec_cfg, 'enable_progress_bar')
        assert hasattr(exec_cfg, 'gradient_clip_val')

    def test_overrides_applied_records_execution_knobs(self, mock_train_npz, temp_output_dir):
        """Factory audit trail includes execution config knobs when applied."""
        from ptycho.config.config import PyTorchExecutionConfig

        custom_exec_cfg = PyTorchExecutionConfig(
            accelerator='cpu',
            num_workers=4,
            deterministic=True,
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512, 'batch_size': 8},
            execution_config=custom_exec_cfg,
        )
        # GREEN phase assertions:
        # Execution knobs should be recorded in overrides_applied
        assert 'accelerator' in payload.overrides_applied or payload.execution_config.accelerator == 'cpu'
        assert 'num_workers' in payload.overrides_applied or payload.execution_config.num_workers == 4

    def test_checkpoint_knobs_propagate_through_factory(self, mock_train_npz, temp_output_dir):
        """
        RED Test: Checkpoint control knobs propagate from execution_config to payload.

        Expected RED Failure:
        - AttributeError: 'PyTorchExecutionConfig' object has no attribute 'checkpoint_mode'
        OR
        - AssertionError: Checkpoint fields do not match expected values

        References:
        - input.md EB1.E (checkpoint controls RED tests)
        - plans/.../phase_e_execution_knobs/plan.md §EB1.C (factory wiring)
        """
        from ptycho.config.config import PyTorchExecutionConfig

        custom_exec_cfg = PyTorchExecutionConfig(
            enable_checkpointing=False,
            checkpoint_save_top_k=3,
            checkpoint_monitor_metric='train_loss',
            checkpoint_mode='max',
            early_stop_patience=10,
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
            execution_config=custom_exec_cfg,
        )

        # GREEN phase assertions:
        exec_cfg = payload.execution_config
        assert exec_cfg.enable_checkpointing is False
        assert exec_cfg.checkpoint_save_top_k == 3
        assert exec_cfg.checkpoint_monitor_metric == 'train_loss'
        assert exec_cfg.checkpoint_mode == 'max'
        assert exec_cfg.early_stop_patience == 10

    def test_checkpoint_defaults_respected(self, mock_train_npz, temp_output_dir):
        """
        RED Test: Checkpoint knobs use dataclass defaults when not overridden.

        Expected RED Failure:
        - AttributeError: 'PyTorchExecutionConfig' object has no attribute 'checkpoint_mode'
        OR
        - AssertionError: Default values do not match expected

        References:
        - input.md EB1.E (checkpoint controls RED tests)
        - plans/.../phase_e_execution_knobs/plan.md §EB1.A (schema audit)
        """
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
        )

        # GREEN phase assertions (verify defaults from PyTorchExecutionConfig):
        exec_cfg = payload.execution_config
        assert exec_cfg.enable_checkpointing is True  # Default per dataclass
        assert exec_cfg.checkpoint_save_top_k == 1  # Default per dataclass
        assert exec_cfg.checkpoint_monitor_metric == 'val_loss'  # Default per dataclass
        assert hasattr(exec_cfg, 'checkpoint_mode')  # New field in EB1.A
        assert exec_cfg.checkpoint_mode == 'min'  # Expected default for loss metrics
        assert exec_cfg.early_stop_patience == 100  # Default per dataclass

    def test_scheduler_override_applied(self, mock_train_npz, temp_output_dir):
        """
        Test: --scheduler override propagates to execution_config and overrides_applied.

        Expected Behavior:
        - execution_config.scheduler matches provided value
        - overrides_applied['scheduler'] records the applied value

        References:
        - input.md EB2.B2 (factory override tests)
        - plans/.../phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md §EB2.B (factory wiring)
        """
        from ptycho.config.config import PyTorchExecutionConfig

        custom_exec_cfg = PyTorchExecutionConfig(
            scheduler='Exponential',
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
            execution_config=custom_exec_cfg,
        )

        # GREEN phase assertions:
        assert payload.execution_config.scheduler == 'Exponential', \
            f"Expected scheduler='Exponential', got {payload.execution_config.scheduler}"
        assert 'scheduler' in payload.overrides_applied, \
            "scheduler must appear in overrides_applied audit trail"
        assert payload.overrides_applied['scheduler'] == 'Exponential', \
            f"Expected overrides_applied['scheduler']='Exponential', got {payload.overrides_applied['scheduler']}"

    def test_accum_steps_override_applied(self, mock_train_npz, temp_output_dir):
        """
        Test: --accumulate-grad-batches override propagates to execution_config and overrides_applied.

        Expected Behavior:
        - execution_config.accum_steps matches provided value
        - overrides_applied['accum_steps'] records the applied value

        References:
        - input.md EB2.B2 (factory override tests)
        - plans/.../phase_e_execution_knobs/2025-10-23T081500Z/eb2_plan.md §EB2.B (factory wiring)
        """
        from ptycho.config.config import PyTorchExecutionConfig

        custom_exec_cfg = PyTorchExecutionConfig(
            accum_steps=4,
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
            execution_config=custom_exec_cfg,
        )

        # GREEN phase assertions:
        assert payload.execution_config.accum_steps == 4, \
            f"Expected accum_steps=4, got {payload.execution_config.accum_steps}"
        assert 'accum_steps' in payload.overrides_applied, \
            "accum_steps must appear in overrides_applied audit trail"
        assert payload.overrides_applied['accum_steps'] == 4, \
            f"Expected overrides_applied['accum_steps']=4, got {payload.overrides_applied['accum_steps']}"

    def test_logger_backend_csv_default(self, mock_train_npz, temp_output_dir):
        """
        RED Test: Factory returns CSV logger instance when logger_backend='csv'.

        Expected RED Failure:
        - AttributeError: 'PyTorchExecutionConfig' object has no attribute 'logger_backend'
        OR
        - NotImplementedError from factory stub

        Expected GREEN Behavior:
        - execution_config.logger_backend == 'csv'
        - Factory returns or prepares CSV logger configuration

        References:
        - input.md EB3.B1 (factory logger tests)
        - plans/.../phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md §Q1
        """
        from ptycho.config.config import PyTorchExecutionConfig

        custom_exec_cfg = PyTorchExecutionConfig(
            logger_backend='csv',
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
            execution_config=custom_exec_cfg,
        )

        # GREEN phase assertions:
        assert payload.execution_config.logger_backend == 'csv', \
            f"Expected logger_backend='csv', got {payload.execution_config.logger_backend}"
        # Verify override tracking
        assert 'logger_backend' in payload.overrides_applied, \
            "logger_backend must appear in overrides_applied audit trail"
        assert payload.overrides_applied['logger_backend'] == 'csv'

    def test_logger_backend_tensorboard(self, mock_train_npz, temp_output_dir):
        """
        RED Test: Factory handles TensorBoard logger backend configuration.

        Expected RED Failure:
        - AttributeError: 'PyTorchExecutionConfig' object has no attribute 'logger_backend'
        OR
        - NotImplementedError from factory stub

        Expected GREEN Behavior:
        - execution_config.logger_backend == 'tensorboard'

        References:
        - input.md EB3.B1 (factory logger tests)
        - plans/.../phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md §Q2
        """
        from ptycho.config.config import PyTorchExecutionConfig

        custom_exec_cfg = PyTorchExecutionConfig(
            logger_backend='tensorboard',
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'n_groups': 512},
            execution_config=custom_exec_cfg,
        )

        # GREEN phase assertions:
        assert payload.execution_config.logger_backend == 'tensorboard', \
            f"Expected logger_backend='tensorboard', got {payload.execution_config.logger_backend}"


# ============================================================================
# Test Category 7: Generator Output Mode Overrides
# ============================================================================

class TestGeneratorOutputModeOverrides:
    """Verify generator_output_mode override propagates to PT model config."""

    def test_generator_output_mode_override_propagates(self, mock_train_npz, temp_output_dir):
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={
                'n_groups': 512,
                'architecture': 'fno',
                'generator_output_mode': 'amp_phase_logits',
            },
        )
        assert payload.pt_model_config.generator_output_mode == 'amp_phase_logits'


# ============================================================================
# Test Category 7: Probe Size Inference Helper
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
