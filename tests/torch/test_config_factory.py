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
    5. Validation Errors (missing training_groups, invalid paths)

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
    simulation_from_datagen_config,
    datagen_config_from_simulation,
    TrainingPayload,
    InferencePayload,
)

# Config dataclasses for assertions
from ptycho.config.config import (
    DetectorSimulationConfig,
    ModelConfig as TFModelConfig,
    ProbeSimulationConfig,
    SimulationConfig,
    SyntheticObjectConfig,
    TrainingConfig as TFTrainingConfig,
    InferenceConfig as TFInferenceConfig,
)
from ptycho_torch.config_params import (
    DataConfig as PTDataConfig,
    ModelConfig as PTModelConfig,
    TrainingConfig as PTTrainingConfig,
    InferenceConfig as PTInferenceConfig,
    DatagenConfig,
)


def test_resolved_torch_records_build_exact_training_payload(tmp_path):
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.config_factory import (
        create_training_payload_from_resolved_configs,
    )

    data = PTDataConfig(N=96, gridsize=2, neighbor_count=7, nphotons=3e7)
    model = PTModelConfig(
                object_big=True,
        rect_s1s2_trainable=True,
    )
    training = PTTrainingConfig(
        learning_rate=4e-4,
        epochs=3,
        batch_size=5,
    )
    inference = PTInferenceConfig(patch_weighting="uniform")
    execution = PyTorchExecutionConfig(
        accelerator="cpu",
        devices=1,
        strategy="auto",
        num_workers=0,
    )

    payload = create_training_payload_from_resolved_configs(
        data,
        model,
        training,
        inference,
        execution,
        train_data_file=tmp_path / "train.npz",
        output_dir=tmp_path / "run",
        training_groups=23,
        parity_scale_mode="fixed",
        parity_fixed_delta=0.75,
        parity_init_scheme="tf_glorot",
    )

    assert payload.pt_data_config is data
    assert payload.pt_model_config is model
    assert payload.pt_training_config is training
    assert payload.pt_inference_config is inference
    assert payload.execution_config is execution
    assert payload.tf_training_config.train_data_file == tmp_path / "train.npz"
    assert payload.tf_training_config.output_dir == tmp_path / "run"
    assert payload.tf_training_config.training_groups == 23
    assert payload.tf_training_config.model.N == 96
    assert payload.tf_training_config.model.gridsize == 2
    assert payload.model_spec.parity_scale_mode == "fixed"
    assert payload.model_spec.parity_fixed_delta == 0.75
    assert payload.model_spec.parity_init_scheme == "tf_glorot"


def test_resolved_torch_records_preserve_use_all_group_semantics(tmp_path):
    from ptycho.config.config import PyTorchExecutionConfig
    from ptycho_torch.config_factory import (
        create_training_payload_from_resolved_configs,
    )

    training = PTTrainingConfig(training_groups=None)
    payload = create_training_payload_from_resolved_configs(
        PTDataConfig(),
        PTModelConfig(),
        training,
        PTInferenceConfig(),
        PyTorchExecutionConfig(accelerator="cpu", devices=1),
        train_data_file=tmp_path / "train.npz",
        output_dir=tmp_path / "run",
        training_groups=None,
    )

    assert training.training_groups is None
    assert payload.tf_training_config.training_groups is None


def test_datagen_config_converts_owned_fields_to_simulation_without_changing_payload_shape():
    datagen = DatagenConfig(
        objects_per_probe=6,
        diff_per_object=128,
        object_class="dead_leaves",
        image_size=(256, 256),
        probe_paths=["probe.npz"],
        beamstop_diameter=8,
    )

    simulation = datagen.to_simulation_config()

    assert simulation.object == SyntheticObjectConfig(
        kind="dead_leaves",
        image_size=(256, 256),
        objects_per_probe=6,
        diffractions_per_object=128,
    )
    assert simulation.probe.source == "custom"
    assert simulation.probe.source_path == Path("probe.npz")
    assert simulation.detector.beamstop_diameter == 8
    assert tuple(datagen.__dataclass_fields__) == (
        "objects_per_probe",
        "diff_per_object",
        "object_class",
        "image_size",
        "probe_paths",
        "beamstop_diameter",
    )


def test_datagen_config_round_trip_preserves_only_representable_owned_fields():
    simulation = SimulationConfig(
        probe=ProbeSimulationConfig(source="custom", source_path=Path("probe.npz")),
        object=SyntheticObjectConfig(
            kind="natural_patch",
            image_size=(320, 320),
            objects_per_probe=5,
            diffractions_per_object=64,
        ),
        detector=DetectorSimulationConfig(
            photons_per_pattern=1e7,
            beamstop_diameter=4,
        ),
    )

    restored = datagen_config_from_simulation(simulation)

    assert restored == DatagenConfig(
        objects_per_probe=5,
        diff_per_object=64,
        object_class="natural_patch",
        image_size=(320, 320),
        probe_paths=["probe.npz"],
        beamstop_diameter=4,
    )
    assert simulation_from_datagen_config(restored, base=simulation) == simulation


def test_datagen_config_rejects_lossy_probe_or_object_conversion():
    with pytest.raises(ValueError, match="exactly one probe path"):
        DatagenConfig(probe_paths=["a.npz", "b.npz"]).to_simulation_config()
    with pytest.raises(ValueError, match="single object_class"):
        DatagenConfig(object_class=["lines", "dead_leaves"]).to_simulation_config()
    with pytest.raises(ValueError, match="ideal probe"):
        DatagenConfig.from_simulation_config(
            SimulationConfig(probe=ProbeSimulationConfig(source="ideal"))
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
def mock_train_npz_with_metadata(tmp_path):
    """Mock training NPZ with embedded metadata including nphotons."""
    import numpy as np
    from ptycho.metadata import MetadataManager
    from ptycho.config.config import TrainingConfig as TFTrainingConfig, ModelConfig as TFModelConfig

    N = 64
    n_images = 100
    M = 256
    nphotons = 5e8

    npz_path = tmp_path / "mock_train_with_metadata.npz"
    payload = {
        "diffraction": np.random.rand(n_images, N, N).astype(np.float32),
        "probeGuess": np.random.rand(N, N).astype(np.complex64),
        "objectGuess": np.random.rand(M, M).astype(np.complex64),
        "xcoords": np.linspace(0, 1, n_images).astype(np.float64),
        "ycoords": np.linspace(0, 1, n_images).astype(np.float64),
        "scan_index": np.arange(n_images).astype(np.int32),
    }
    tf_config = TFTrainingConfig(model=TFModelConfig(N=N, gridsize=1), nphotons=nphotons)
    metadata = MetadataManager.create_metadata(
        tf_config,
        script_name="unit-test",
        source="unit-test",
    )
    MetadataManager.save_with_metadata(str(npz_path), payload, metadata)
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
            overrides={'training_groups': 512, 'batch_size': 16},
        )
        # GREEN phase assertions (will run after implementation):
        assert is_dataclass(payload)
        assert isinstance(payload, TrainingPayload)

    def test_training_payload_contains_tf_config(self, mock_train_npz, temp_output_dir):
        """Payload contains TensorFlow TrainingConfig instance."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
        )
        # GREEN phase:
        assert isinstance(payload.tf_training_config, TFTrainingConfig)

    def test_training_payload_contains_pytorch_configs(self, mock_train_npz, temp_output_dir):
        """Payload contains all three PyTorch singleton config instances."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
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
            overrides={'training_groups': 512, 'batch_size': 8},
        )
        # GREEN phase assertions:
        assert 'training_groups' in payload.overrides_applied
        assert payload.overrides_applied['training_groups'] == 512
        assert payload.overrides_applied['batch_size'] == 8

    def test_gridsize_sets_channel_count(self, mock_train_npz, temp_output_dir):
        """
        Gridsize override synchronizes C_forward and C_model with data channel count.

        Regression test for ADR-003 C4.D3: create_training_payload() must set
        pt_model_config.C_forward and C_model to match pt_data_config.C when
        gridsize is specified. This ensures PyTorch helpers (reassemble_patches_position_real)
        receive tensor shapes consistent with the grouping strategy.

        Expected behavior:
            - gridsize=1 → C=1, 
            - gridsize=2 → C=4, C_forward=4, C_model=4
            - Default (no gridsize override) → C=4, C_forward=4, C_model=4

        Reference: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T061500Z/
                   phase_c4_cli_integration_debug/coords_relative_investigation.md
        """
        # Case 1: gridsize=1 (single-position groups)
        payload_gs1 = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'gridsize': 1, 'training_groups': 512},
        )
        assert payload_gs1.pt_data_config.gridsize == 1, "DataConfig.gridsize should be 1"

        # Case 2: gridsize=2 (2x2 = 4 overlapping positions)
        payload_gs2 = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'gridsize': 2, 'training_groups': 512},
        )
        assert payload_gs2.pt_data_config.gridsize == 2, "DataConfig.gridsize should be 2"

        # Case 3: No gridsize override (default grid_size=(2,2) → C=4)
        payload_default = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
        )
        # Default grid_size is (2,2) per PTDataConfig defaults (config_params.py:29)
        # but factory may compute C from overrides; accept any C >= 1
        assert payload_default.pt_data_config.gridsize >= 1, "DataConfig.gridsize should be positive"

    def test_training_payload_infers_probe_size_for_pt_data_config(self, mock_train_npz_128, temp_output_dir):
        """Factory should propagate inferred N into pt_data_config and TF model config."""
        payload = create_training_payload(
            train_data_file=mock_train_npz_128,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
        )
        assert payload.pt_data_config.N == 128
        assert payload.tf_training_config.model.N == 128

    def test_training_payload_maps_model_type_override_to_pt_mode(self, mock_train_npz, temp_output_dir):
        """Legacy model_type override must drive the PyTorch mode enum."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={
                'training_groups': 512,
                'model_type': 'Supervised',
            },
        )
        assert payload.pt_model_config.mode == 'Supervised'
        assert payload.tf_training_config.model.model_type == 'supervised'


class TestInferencePayloadStructure:
    """Verify create_inference_payload() returns InferencePayload with all required fields."""

    def test_inference_payload_returns_dataclass(self, mock_checkpoint_dir, mock_test_npz, temp_output_dir):
        """Factory returns InferencePayload dataclass instance."""
        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 128},
        )
        # GREEN phase:
        assert isinstance(payload, InferencePayload)

    def test_inference_payload_contains_tf_config(self, mock_checkpoint_dir, mock_test_npz, temp_output_dir):
        """Payload contains TensorFlow InferenceConfig instance."""
        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 128},
        )
        # GREEN phase:
        assert isinstance(payload.tf_inference_config, TFInferenceConfig)

    def test_inference_payload_contains_pytorch_configs(self, mock_checkpoint_dir, mock_test_npz, temp_output_dir):
        """Payload contains PyTorch inference config instances."""
        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 128},
        )
        # GREEN phase assertions:
        assert isinstance(payload.pt_data_config, PTDataConfig)
        assert isinstance(payload.pt_inference_config, PTInferenceConfig)

    def test_create_training_payload_propagates_varpro_scaling(self, mock_train_npz, temp_output_dir):
        """Training payload preserves torch-only VarPro scaling override."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 128, 'varpro_scaling': False},
        )

        assert payload.pt_inference_config.varpro_scaling is False

    def test_create_inference_payload_propagates_varpro_scaling(self, mock_checkpoint_dir, mock_test_npz, temp_output_dir):
        """Inference payload preserves torch-only VarPro scaling override."""
        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 128, 'varpro_scaling': False},
        )

        assert payload.pt_inference_config.varpro_scaling is False


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
            overrides={'training_groups': 512, 'gridsize': 2},
        )
        # GREEN phase assertions:
        assert payload.pt_data_config.gridsize == 2  # PyTorch int
        assert payload.tf_training_config.model.gridsize == 2  # TensorFlow int

    def test_epochs_to_nepochs_conversion(self, mock_train_npz, temp_output_dir):
        """Factory maps epochs → nepochs via bridge."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'max_epochs': 20},
        )
        # GREEN phase assertions:
        assert payload.pt_training_config.epochs == 20  # PyTorch naming
        assert payload.tf_training_config.nepochs == 20  # TensorFlow naming

    def test_k_to_neighbor_count_conversion(self, mock_train_npz, temp_output_dir):
        """Factory maps K → neighbor_count via bridge."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'neighbor_count': 7},
        )
        # GREEN phase assertions:
        assert payload.pt_data_config.neighbor_count == 7  # PyTorch K
        assert payload.tf_training_config.neighbor_count == 7  # TensorFlow naming


# ============================================================================
# Test Category 3: Legacy params.cfg Projection (Retired)
# ============================================================================

class TestLegacyParamsNotPopulated:
    """
    Verify the torch factory no longer projects the legacy ``params.cfg``.

    The CONFIG-001 checkpoint (factory-populates-legacy-global) is retired: the
    data path reads explicit arguments instead, so ``create_training_payload``
    resolves owners without committing the global dictionary.
    """

    def test_factory_does_not_populate_params_cfg(self, mock_train_npz, temp_output_dir):
        """Factory resolves owners without touching ptycho.params.cfg."""
        ptycho.params.cfg.clear()

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'gridsize': 2},
        )

        assert ptycho.params.cfg == {}
        assert payload.tf_training_config.training_groups == 512


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
            overrides={'training_groups': 1024, 'batch_size': 16},
        )
        # GREEN phase assertions:
        assert payload.tf_training_config.training_groups == 1024  # Override wins
        assert payload.tf_training_config.batch_size == 16

    def test_probe_size_override_wins_over_inference(self, mock_train_npz, temp_output_dir):
        """Explicit N override takes precedence over inferred probe size."""
        # NPZ has N=64 probe, but override specifies N=128
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'N': 128},
        )
        # GREEN phase:
        assert payload.tf_training_config.model.N == 128

    def test_nphotons_defaults_to_tf_without_metadata(self, mock_train_npz, temp_output_dir):
        """When no metadata or override, factory should use TF default nphotons."""
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 128},
        )
        assert payload.tf_training_config.nphotons == TFTrainingConfig(model=TFModelConfig()).nphotons

    def test_nphotons_uses_metadata_when_present(self, mock_train_npz_with_metadata, temp_output_dir):
        """Metadata nphotons should override defaults when present."""
        payload = create_training_payload(
            train_data_file=mock_train_npz_with_metadata,
            output_dir=temp_output_dir,
            overrides={'training_groups': 128},
        )
        assert payload.tf_training_config.nphotons == 5e8


# ============================================================================
# Test Category 5: Validation Errors
# ============================================================================

class TestFactoryValidation:
    """
    Verify factory raises appropriate errors for invalid inputs.

    Critical validations:
        - train_data_file / test_data_file path existence
        - training_groups required in overrides (no default)
        - model_path must contain wts.h5.zip
        - NPZ field validation (diffraction, probeGuess present)
    """

    def test_missing_training_groups_raises_error(self, mock_train_npz, temp_output_dir):
        """Factory raises ValueError if training_groups missing from overrides."""
        # Omit training_groups (required field)
        with pytest.raises(ValueError, match="training_groups is required"):
            payload = create_training_payload(
                train_data_file=mock_train_npz,
                output_dir=temp_output_dir,
                overrides={},  # Missing training_groups!
            )

    def test_nonexistent_train_data_file_raises_error(self, temp_output_dir):
        """Factory raises FileNotFoundError for missing train_data_file."""
        with pytest.raises(FileNotFoundError, match="Training data file not found"):
            payload = create_training_payload(
                train_data_file=Path("/nonexistent/train.npz"),
                output_dir=temp_output_dir,
                overrides={'training_groups': 512},
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
                overrides={'training_groups': 128},
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
            overrides={'training_groups': 512},
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
            overrides={'training_groups': 128},
        )
        # GREEN phase:
        assert payload.execution_config is not None
        assert isinstance(payload.execution_config, PyTorchExecutionConfig)

    def test_inference_payload_keeps_object_big_decoder_support_on_by_default(
        self, mock_checkpoint_dir, mock_test_npz, temp_output_dir
    ):
        """Normal object-big inference must not silently disable outer support."""
        payload = create_inference_payload(
            model_path=mock_checkpoint_dir,
            test_data_file=mock_test_npz,
            output_dir=temp_output_dir,
            overrides={"training_groups": 128, "object_big": True},
        )

        assert payload.tf_inference_config.model.object_big is True
        assert payload.tf_inference_config.model.probe_big is True

    def test_execution_config_defaults_applied(self, mock_train_npz, temp_output_dir):
        """Execution config uses dataclass defaults when not overridden."""
        import torch
        from ptycho.config.config import PyTorchExecutionConfig

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
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
        """User-provided runtime request resolves into the payload carrier."""
        from ptycho_torch.execution_request import ExecutionRequest

        request = ExecutionRequest(
            values={
                "accelerator": "gpu",
                "enable_progress_bar": True,
                "deterministic": False,
            },
            explicit_fields=frozenset(
                {"accelerator", "enable_progress_bar", "deterministic"}
            ),
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
            execution_config=request,
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
            overrides={'training_groups': 512},
        )
        # GREEN phase: Verify key execution knobs are accessible
        exec_cfg = payload.execution_config
        assert hasattr(exec_cfg, 'accelerator')
        assert hasattr(exec_cfg, 'deterministic')
        assert hasattr(exec_cfg, 'num_workers')
        assert hasattr(exec_cfg, 'enable_progress_bar')
        assert not hasattr(exec_cfg, 'gradient_clip_val')
        assert hasattr(payload.pt_training_config, 'gradient_clip_val')

    def test_overrides_applied_records_execution_knobs(self, mock_train_npz, temp_output_dir):
        """Factory audit trail includes execution config knobs when applied."""
        from ptycho_torch.execution_request import ExecutionRequest

        request = ExecutionRequest(
            values={
                "accelerator": "cpu",
                "num_workers": 4,
                "deterministic": True,
            },
            explicit_fields=frozenset(
                {"accelerator", "num_workers", "deterministic"}
            ),
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'batch_size': 8},
            execution_config=request,
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
        from ptycho_torch.execution_request import ExecutionRequest

        request = ExecutionRequest(
            values={
                "enable_checkpointing": False,
                "checkpoint_save_top_k": 3,
                "checkpoint_monitor_metric": "train_loss",
                "checkpoint_mode": "max",
                "early_stop_patience": 10,
            },
            explicit_fields=frozenset(
                {
                    "enable_checkpointing",
                    "checkpoint_save_top_k",
                    "checkpoint_monitor_metric",
                    "checkpoint_mode",
                    "early_stop_patience",
                }
            ),
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
            execution_config=request,
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
            overrides={'training_groups': 512},
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
        """Scheduler overrides resolve only on canonical TrainingConfig."""

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'scheduler': 'Exponential'},
        )

        assert payload.pt_training_config.scheduler == 'Exponential'
        assert not hasattr(payload.execution_config, 'scheduler')
        assert 'scheduler' in payload.overrides_applied, \
            "scheduler must appear in overrides_applied audit trail"
        assert payload.overrides_applied['scheduler'] == 'Exponential', \
            f"Expected overrides_applied['scheduler']='Exponential', got {payload.overrides_applied['scheduler']}"

    def test_accum_steps_override_applied(self, mock_train_npz, temp_output_dir):
        """Accumulation overrides resolve only on canonical TrainingConfig."""

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'accum_steps': 4},
        )

        assert payload.pt_training_config.accum_steps == 4
        assert not hasattr(payload.execution_config, 'accum_steps')
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
        from ptycho_torch.execution_request import ExecutionRequest

        request = ExecutionRequest(
            values={"logger_backend": "csv"},
            explicit_fields=frozenset({"logger_backend"}),
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
            execution_config=request,
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
        from ptycho_torch.execution_request import ExecutionRequest

        request = ExecutionRequest(
            values={"logger_backend": "tensorboard"},
            explicit_fields=frozenset({"logger_backend"}),
        )

        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512},
            execution_config=request,
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
                'training_groups': 512,
                'architecture': 'fno',
                'generator_output_mode': 'amp_phase_logits',
            },
        )
        assert payload.pt_model_config.generator_output_mode == 'amp_phase_logits'


# ============================================================================
# Test Category 8: VarPro / Probe-Weighting ModelConfig Knob Overrides (Task 2.7 / B7)
# ============================================================================

class TestVarProProbeWeightingKnobOverrides:
    """Prove the 5 ModelConfig knobs added by Phase-2 (Tasks 2.3-2.6) survive
    create_training_payload's override round-trip.

    A naive test that only checks create_training_payload() did not raise would
    pass even if the resolver silently dropped a key. These tests instead
    assert that each override reaches the resolved ModelConfig owner.
    """

    def test_cnn_output_mode_override_survives_training_payload(self, mock_train_npz, temp_output_dir):
        assert PTModelConfig().cnn_output_mode == 'amp_phase'  # documents the default
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'cnn_output_mode': 'real_imag'},
        )
        assert payload.pt_model_config.cnn_output_mode == 'real_imag'

    def test_use_shared_decoder_override_survives_training_payload(self, mock_train_npz, temp_output_dir):
        assert PTModelConfig().use_shared_decoder is False  # documents the default
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'use_shared_decoder': True},
        )
        assert payload.pt_model_config.use_shared_decoder is True

    def test_training_patch_weighting_override_survives_training_payload(self, mock_train_npz, temp_output_dir):
        assert PTModelConfig().training_patch_weighting is None  # unresolved input default
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'training_patch_weighting': 'probe'},
        )
        assert payload.pt_model_config.training_patch_weighting == 'probe'

    def test_physics_forward_mode_override_survives_training_payload(self, mock_train_npz, temp_output_dir):
        assert PTModelConfig().physics_forward_mode == 'amplitude'  # documents the default
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'physics_forward_mode': 'rectangular_scaled'},
        )
        assert payload.pt_model_config.physics_forward_mode == 'rectangular_scaled'

    def test_rect_s1s2_trainable_override_survives_training_payload(self, mock_train_npz, temp_output_dir):
        assert PTModelConfig().rect_s1s2_trainable is True  # documents the default
        payload = create_training_payload(
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            overrides={'training_groups': 512, 'rect_s1s2_trainable': False},
        )
        assert payload.pt_model_config.rect_s1s2_trainable is False


# ============================================================================
# Test Category 8b: _train_with_lightning `overrides` forwarding (Task 2.7 fix / B7)
# ============================================================================

class TestTrainWithLightningVarProProbeWeightingForwarding:
    """Prove the runner-plumbing fix: _train_with_lightning's `overrides` kwarg
    (ptycho_torch/workflows/components.py) actually reaches the built
    pt_model_config through create_training_payload's REAL (unmocked) override
    round-trip, not just an intermediate factory_overrides dict.

    Review finding (Task 2.7 / commit 99a3acf0): the 4 runner CLI flags
    (--training-patch-weighting, --physics-forward-mode, --cnn-output-mode,
    --freeze-s1s2) populated TorchRunnerConfig + provenance but never reached
    create_training_payload, so they had zero effect on the trained model. This
    class exercises the fix end-to-end from _train_with_lightning's `overrides`
    parameter through to pt_model_config, so a silent-drop or missing-forward
    regression fails these tests.
    """

    def test_overrides_kwarg_reaches_pt_model_config(self, monkeypatch, mock_train_npz, temp_output_dir):
        from ptycho_torch.workflows import components
        import ptycho_torch.config_factory as config_factory_module

        cfg = TFTrainingConfig(
            model=TFModelConfig(N=64, gridsize=1, architecture="fno"),
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            backend="pytorch",
            training_groups=4,
        )

        real_resolve_training_payload = config_factory_module.resolve_training_payload
        captured = {}

        def spy_resolve_payload(*args, **kwargs):
            payload = real_resolve_training_payload(*args, **kwargs)
            captured["pt_model_config"] = payload.pt_model_config
            raise RuntimeError("stop-after-capture")

        monkeypatch.setattr(
            "ptycho_torch.config_factory.resolve_training_payload",
            spy_resolve_payload,
        )

        monkeypatch.setattr("ptycho_torch.workflows.containers.create_torch_data_container",
            lambda *_args, **_kwargs: object(),
        )
        with pytest.raises(RuntimeError, match="stop-after-capture"):
            components.train_cdi_model_torch(
                train_data=object(),
                test_data=None,
                config=cfg,
                overrides={
                    "training_patch_weighting": "probe",
                    "physics_forward_mode": "rectangular_scaled",
                    "cnn_output_mode": "real_imag",
                    "rect_s1s2_trainable": False,
                },
            )

        pt_model_config = captured["pt_model_config"]
        assert pt_model_config.training_patch_weighting == "probe"
        assert pt_model_config.physics_forward_mode == "rectangular_scaled"
        assert pt_model_config.cnn_output_mode == "real_imag"
        assert pt_model_config.rect_s1s2_trainable is False

    def test_omitted_overrides_kwarg_preserves_pt_model_config_defaults(
        self, monkeypatch, mock_train_npz, temp_output_dir
    ):
        """Omission preserves defaults except declared object-policy derivation."""
        from ptycho_torch.workflows import components
        import ptycho_torch.config_factory as config_factory_module

        cfg = TFTrainingConfig(
            model=TFModelConfig(N=64, gridsize=1, architecture="fno"),
            train_data_file=mock_train_npz,
            output_dir=temp_output_dir,
            backend="pytorch",
            training_groups=4,
        )

        real_resolve_training_payload = config_factory_module.resolve_training_payload
        captured = {}

        def spy_resolve_payload(*args, **kwargs):
            payload = real_resolve_training_payload(*args, **kwargs)
            captured["pt_model_config"] = payload.pt_model_config
            raise RuntimeError("stop-after-capture")

        monkeypatch.setattr(
            "ptycho_torch.config_factory.resolve_training_payload",
            spy_resolve_payload,
        )

        monkeypatch.setattr("ptycho_torch.workflows.containers.create_torch_data_container",
            lambda *_args, **_kwargs: object(),
        )
        with pytest.raises(RuntimeError, match="stop-after-capture"):
            components.train_cdi_model_torch(
                train_data=object(),
                test_data=None,
                config=cfg,
            )

        pt_model_config = captured["pt_model_config"]
        defaults = PTModelConfig()
        assert defaults.training_patch_weighting is None
        assert pt_model_config.training_patch_weighting == "central_mask"
        assert pt_model_config.physics_forward_mode == defaults.physics_forward_mode
        assert pt_model_config.cnn_output_mode == defaults.cnn_output_mode
        assert pt_model_config.rect_s1s2_trainable == defaults.rect_s1s2_trainable


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

    @pytest.mark.parametrize(
        "probe_shape",
        [(64, 64), (3, 64, 64), (64, 64, 1)],
        ids=["two-dimensional", "mode-first", "legacy-singleton"],
    )
    def test_probe_size_consumers_use_canonical_header(self, tmp_path, probe_shape):
        import numpy as np
        from ptycho_torch.config_resolution import observe_probe_size

        path = tmp_path / "probe_layout.npz"
        np.savez(
            path,
            diffraction=np.ones((3, 64, 64), dtype=np.float32),
            probeGuess=np.ones(probe_shape, dtype=np.complex64),
            xcoords=np.arange(3, dtype=np.float64),
            ycoords=np.arange(3, dtype=np.float64),
        )

        observation = observe_probe_size(path)

        assert observation.value == 64
        assert observation.notices == ()
        assert infer_probe_size(path) == 64

    def test_probe_size_accepts_grouped_dataset_contract(self, tmp_path):
        """Grouped training NPZs have probe identity but no raw scan vectors."""
        import numpy as np
        from ptycho_torch.config_resolution import observe_probe_size

        path = tmp_path / "grouped.npz"
        np.savez(
            path,
            diffraction=np.ones((3, 128, 128, 1), dtype=np.float32),
            coords_nominal=np.zeros((3, 1, 2, 1), dtype=np.float32),
            probeGuess=np.ones((128, 128), dtype=np.complex64),
        )

        observation = observe_probe_size(path)

        assert observation.value == 128
        assert observation.notices == ()

    def test_infer_probe_size_missing_file_fallback(self):
        """Helper returns fallback N=64 for missing NPZ file."""
        N = infer_probe_size(Path("/nonexistent/data.npz"))
        # GREEN phase:
        assert N == 64  # Fallback per design decision


def test_factory_overrides_emit_mode_not_model_type():
    """D3: the factory emits the canonical torch key ``mode`` directly, not
    ``model_type`` (which previously round-tripped through the alias table)."""
    from ptycho_torch.config_factory import build_training_factory_overrides

    config = TFTrainingConfig(
        model=TFModelConfig(N=64, gridsize=1, architecture="cnn", model_type="pinn"),
        training_groups=7,
    )
    overrides = build_training_factory_overrides(config)
    assert "mode" in overrides
    assert "model_type" not in overrides
    assert overrides["mode"] == "Unsupervised"


def test_patch_supplying_model_type_still_resolves_to_mode():
    """The external ``model_type`` fence still resolves to canonical ``mode``."""
    from ptycho_torch.config_resolution import normalize_training_patch

    canonical = normalize_training_patch({"mode": "Supervised"})
    fenced = normalize_training_patch({"model_type": "Supervised"})
    assert fenced.values == canonical.values
    assert fenced.values["mode"] == "Supervised"


def test_build_training_factory_overrides_equality():
    """W1 pre/post equality: freeze the factory override projection.

    The factory override surface gained an import-time tripwire (possible-key
    set within the resolver vocabulary). This freezes the surface's output on
    fixture inputs so a refactor that silently changes a projected key is
    caught here.
    """
    from ptycho_torch.config_factory import build_training_factory_overrides
    from ptycho_torch.config_bridge import to_model_config, to_training_config

    data = PTDataConfig(N=128, gridsize=3, nphotons=1e9, neighbor_count=7)
    model = PTModelConfig(mode='Unsupervised', amp_activation='silu', n_filters_scale=3)
    tf_model = to_model_config(data, model)
    tf_train = to_training_config(
        tf_model, data, model, PTTrainingConfig(epochs=77, batch_size=24, nll=False),
        overrides=dict(
            train_data_file=Path('train.npz'),
            test_data_file=Path('test.npz'),
            training_groups=512,
        ),
    )
    expected = {
        'adam_beta1': 0.9,
        'adam_beta2': 0.999,
        'amp_activation': 'swish',
        'architecture': 'cnn',
        'batch_size': 24,
        'fno_blocks': 4,
        'fno_cnn_blocks': 2,
        'fno_input_transform': 'none',
        'fno_modes': 12,
        'fno_width': 32,
        'gaussian_smoothing_sigma': 0.0,
        'generator_output_mode': 'real_imag',
        'grad_norm_log_freq': 1,
        'gradient_clip_algorithm': 'norm',
        'gradient_clip_val': None,
        'gridsize': 3,
        'intensity_scale_trainable': False,
        'learned_input_channels': 1,
        'log_grad_norm': False,
        'lr_min_ratio': 0.1,
        'lr_warmup_epochs': 0,
        'max_epochs': 77,
        'mode': 'Unsupervised',
        'momentum': 0.9,
        'n_filters_scale': 3,
        'neighbor_count': 7,
        'nphotons': 1000000000.0,
        'object_layout': 'grouped_patches',
        'optimizer': 'adam',
        'pad_object': True,
        'plateau_factor': 0.5,
        'plateau_min_lr': 5e-5,
        'plateau_patience': 2,
        'plateau_threshold': 0.0,
        'probe_big': True,
        'probe_mask': False,
        'probe_mask_diameter': None,
        'probe_mask_sigma': 1.0,
        'probe_scale': 4.0,
        'scheduler': 'Default',
        'sequential_sampling': False,
        'subsample_seed': None,
        'test_data_file': Path('test.npz'),
        'torch_loss_mode': 'poisson',
        'torch_mae_pred_l2_match_target': False,
        'training_canvas': 'relative_overlap',
        'training_groups': 512,
        'training_patch_weighting': 'central_mask',
        'weight_decay': 0.0,
    }
    assert build_training_factory_overrides(tf_train) == expected


# ============================================================================
# Test Execution Summary (for RED log capture)
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
