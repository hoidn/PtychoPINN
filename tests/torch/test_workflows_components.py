"""
Tests for ptycho_torch/workflows/components.py — PyTorch workflow orchestration module.

This module validates that the PyTorch workflow functions (`run_cdi_example_torch`,
`train_cdi_model_torch`, `load_inference_bundle_torch`) satisfy the reconstructor
contract defined in specs/ptychodus_api_spec.md §4 and maintain parity with
ptycho/workflows/components.py.

Critical Behavioral Requirements (from CONFIG-001 + spec §4.5):
1. Entry points MUST call `update_legacy_dict(params.cfg, config)` before delegating
   to core modules (prevents silent params.cfg drift).
2. All workflow functions must be torch-optional (importable when PyTorch unavailable).
3. Signatures must match TensorFlow equivalents to enable transparent backend selection.

Test Strategy:
- Red-phase: document required API via failing tests using monkeypatch spies
- Green-phase: implement stubs that invoke update_legacy_dict and raise NotImplementedError
- torch-optional: module structure follows test_config_bridge.py pattern (guarded imports)

Artifacts (Phase D2.A):
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T091450Z/phase_d2_scaffold.md
- plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T091450Z/pytest_scaffold.log
"""

import pytest
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    torch = None

# Add to conftest.py TORCH_OPTIONAL_MODULES if not already present
# This test must run without torch runtime


class TestWorkflowsComponentsScaffold:
    """
    Phase D2.A scaffold tests — verify update_legacy_dict parity guard.

    These tests validate that PyTorch workflow entry points follow the critical
    CONFIG-001 pattern: call update_legacy_dict before delegating to core modules.
    """

    @pytest.fixture
    def params_cfg_snapshot(self):
        """Snapshot and restore params.cfg state across tests."""
        from ptycho import params
        original = params.cfg.copy()
        yield params.cfg
        params.cfg.clear()
        params.cfg.update(original)

    @pytest.fixture
    def minimal_training_config(self):
        """Create minimal TrainingConfig fixture for parity tests."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
        )

        return training_config

    def test_run_cdi_example_calls_update_legacy_dict(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config
    ):
        """
        CRITICAL PARITY TEST: run_cdi_example_torch must call update_legacy_dict.

        Requirement: specs/ptychodus_api_spec.md:187 mandates that PyTorch entry
        points must synchronize params.cfg via update_legacy_dict() to prevent
        silent CONFIG-001 violations (shape mismatch errors from empty params.cfg).

        Red-phase contract:
        - Entry signature: run_cdi_example_torch(train_data, test_data, config, ...)
        - MUST call ptycho.config.config.update_legacy_dict(params.cfg, config)
        - Stub implementation may raise NotImplementedError for paths Phase D2.B/C fill

        Test mechanism:
        - Use monkeypatch to spy on update_legacy_dict calls
        - Pass minimal dummy data (no actual training execution required)
        - Assert update_legacy_dict was invoked with correct params.cfg + config args
        """
        # Import the module under test
        # This import must succeed even when torch unavailable (torch-optional)
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import update_legacy_dict
        from ptycho.raw_data import RawData

        # Spy flag to track update_legacy_dict invocation
        update_legacy_dict_called = {"called": False, "args": None}

        def mock_update_legacy_dict(cfg_dict, config_obj):
            """Spy that records invocation and delegates to real function."""
            update_legacy_dict_called["called"] = True
            update_legacy_dict_called["args"] = (cfg_dict, config_obj)
            # Call the real function to populate params.cfg for validation
            update_legacy_dict(cfg_dict, config_obj)

        # Patch update_legacy_dict with spy
        monkeypatch.setattr(
            "ptycho.config.config.update_legacy_dict",
            mock_update_legacy_dict
        )

        # Create minimal dummy train_data (RawData-compatible stub)
        # For scaffold test, we don't need valid NPZ data — just RawData structure
        dummy_coords = np.array([0.0, 1.0, 2.0])
        dummy_diff = np.random.rand(3, 64, 64).astype(np.float32)
        dummy_probe = np.ones((64, 64), dtype=np.complex64)
        dummy_scan_index = np.array([0, 1, 2], dtype=int)

        train_data = RawData(
            xcoords=dummy_coords,
            ycoords=dummy_coords,
            xcoords_start=dummy_coords,
            ycoords_start=dummy_coords,
            diff3d=dummy_diff,
            probeGuess=dummy_probe,
            scan_index=dummy_scan_index,
        )

        # Attempt to call run_cdi_example_torch
        # Phase D2.A: expects NotImplementedError (scaffold only) — COMPLETED
        # Phase D2.B/C: IMPLEMENTED — now returns results tuple
        # Test validates update_legacy_dict was called, but doesn't fully exercise training
        # (training path tested separately in TestWorkflowsComponentsTraining)

        # Monkeypatch train_cdi_model_torch to prevent full training execution in this test
        def mock_train_cdi_model_torch(train_data, test_data, config, execution_config=None):
            """Minimal stub to prevent full training in scaffold test."""
            return {"history": {"train_loss": [0.5]}, "train_container": None, "test_container": None}

        monkeypatch.setattr(
            "ptycho_torch.workflows.components.train_cdi_model_torch",
            mock_train_cdi_model_torch
        )

        # Call should now succeed (Phase D2.C implemented)
        recon_amp, recon_phase, results = torch_components.run_cdi_example_torch(
            train_data=train_data,
            test_data=None,  # Optional
            config=minimal_training_config,
            flip_x=False,
            flip_y=False,
            transpose=False,
            M=20,
            do_stitching=False,
        )

        # Validate update_legacy_dict was called
        assert update_legacy_dict_called["called"], (
            "run_cdi_example_torch MUST call update_legacy_dict before delegating "
            "to prevent CONFIG-001 violations (params.cfg empty → shape mismatches)"
        )

        # Validate correct arguments passed
        cfg_dict_arg, config_obj_arg = update_legacy_dict_called["args"]
        assert cfg_dict_arg is params_cfg_snapshot, (
            "First argument to update_legacy_dict must be ptycho.params.cfg"
        )
        assert config_obj_arg is minimal_training_config, (
            "Second argument to update_legacy_dict must be the TrainingConfig instance"
        )

        # Validate params.cfg was actually populated (side effect check)
        assert params_cfg_snapshot['N'] == 64, "params.cfg should be populated with N=64"
        assert params_cfg_snapshot['gridsize'] == 2, "params.cfg should be populated with gridsize=2"
        assert params_cfg_snapshot['model_type'] == 'pinn', "params.cfg should be populated with model_type='pinn'"


class TestWorkflowsComponentsTraining:
    """
    Phase D2.B training path tests — validate Lightning orchestration.

    These tests validate that train_cdi_model_torch properly normalizes input data
    via Phase C adapters and delegates to Lightning trainer while maintaining
    torch-optional behavior.
    """

    @pytest.fixture
    def params_cfg_snapshot(self):
        """Snapshot and restore params.cfg state across tests."""
        from ptycho import params
        original = params.cfg.copy()
        yield params.cfg
        params.cfg.clear()
        params.cfg.update(original)

    @pytest.fixture
    def minimal_training_config(self):
        """Create minimal TrainingConfig fixture for training tests."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(
            N=64,
            gridsize=1,  # No neighbor sampling for simpler test
            model_type='pinn',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            neighbor_count=1,  # No neighbors
            nphotons=1e9,
            nepochs=2,  # Small number for testing
        )

        return training_config

    @pytest.fixture
    def dummy_raw_data(self):
        """Create minimal RawData fixture for testing."""
        from ptycho.raw_data import RawData

        # Create minimal synthetic data
        nsamples = 10
        N = 64

        dummy_coords = np.linspace(0, 9, nsamples)
        dummy_diff = np.random.rand(nsamples, N, N).astype(np.float32)
        dummy_probe = np.ones((N, N), dtype=np.complex64)
        dummy_scan_index = np.arange(nsamples, dtype=int)

        return RawData(
            xcoords=dummy_coords,
            ycoords=dummy_coords,
            xcoords_start=dummy_coords,
            ycoords_start=dummy_coords,
            diff3d=dummy_diff,
            probeGuess=dummy_probe,
            scan_index=dummy_scan_index,
        )

    def test_train_with_lightning_passes_fno_input_transform(
        self,
        monkeypatch,
        params_cfg_snapshot,
    ):
        """_train_with_lightning must forward fno_input_transform to factory overrides."""
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho_torch.workflows import components as torch_components

        captured = {}

        def fake_create_training_payload(*, train_data_file, output_dir, execution_config=None, overrides=None):
            captured["overrides"] = overrides or {}
            raise RuntimeError("stop after overrides capture")

        monkeypatch.setattr(
            "ptycho_torch.config_factory.create_training_payload",
            fake_create_training_payload
        )

        model_config = ModelConfig(
            N=64,
            gridsize=1,
            model_type='pinn',
            architecture='fno',
            fno_input_transform='log1p',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            neighbor_count=1,
            nphotons=1e9,
            nepochs=1,
        )

        with pytest.raises(RuntimeError, match="stop after overrides capture"):
            torch_components._train_with_lightning(
                train_container={"X": np.ones((1, 64, 64)), "Y": np.ones((1, 64, 64))},
                test_container=None,
                config=training_config,
            )

        assert captured["overrides"].get("fno_input_transform") == "log1p"

    def test_train_cdi_model_torch_invokes_lightning(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        CRITICAL TRAINING PATH TEST: train_cdi_model_torch must delegate to Lightning.

        Requirement: Phase D2.B must implement training orchestration following the
        pattern in plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T093500Z/
        phase_d2_training_analysis.md.

        Red-phase contract:
        - Entry signature: train_cdi_model_torch(train_data, test_data, config)
        - MUST call _ensure_container(data, config) for train/test inputs
        - MUST delegate to Lightning trainer with normalized config
        - MUST return dict with keys: history, train_container, test_container
        - Stub implementation may raise NotImplementedError initially

        Test mechanism:
        - Use monkeypatch to spy on _ensure_container and Lightning orchestration calls
        - Pass minimal RawData (no actual training execution required)
        - Assert expected orchestration order without running full training
        """
        # Import the module under test
        from ptycho_torch.workflows import components as torch_components

        # Spy flags to track internal calls
        ensure_container_calls = []
        lightning_trainer_called = {"called": False, "config": None}

        def mock_ensure_container(data, config):
            """Spy that records _ensure_container invocations."""
            ensure_container_calls.append({
                "data": data,
                "config": config
            })
            # Return a sentinel PtychoDataContainerTorch-like object
            # In Phase D2.B implementation, this would be a real container
            return {"X": np.ones((2, 64, 64)), "Y": np.ones((2, 64, 64), dtype=np.complex64)}

        def mock_lightning_orchestrator(train_container, test_container, config, execution_config=None):
            """Spy that records Lightning trainer invocation."""
            lightning_trainer_called["called"] = True
            lightning_trainer_called["config"] = config
            # Return minimal training results dict
            return {
                "history": {"train_loss": [0.5, 0.3], "val_loss": [0.6, 0.4]},
                "train_container": train_container,
                "test_container": test_container,
            }

        # Patch internal helpers (Phase D2.B implemented)
        monkeypatch.setattr(
            "ptycho_torch.workflows.components._ensure_container",
            mock_ensure_container
        )
        monkeypatch.setattr(
            "ptycho_torch.workflows.components._train_with_lightning",
            mock_lightning_orchestrator
        )

        # Call train_cdi_model_torch (Phase D2.B green phase)
        results = torch_components.train_cdi_model_torch(
            train_data=dummy_raw_data,
            test_data=None,  # Optional
            config=minimal_training_config
        )

        # Phase D2.B green phase assertions - validate orchestration

        # Validate _ensure_container was called for train_data
        assert len(ensure_container_calls) >= 1, (
            "train_cdi_model_torch MUST call _ensure_container to normalize train_data"
        )
        assert ensure_container_calls[0]["data"] is dummy_raw_data
        assert ensure_container_calls[0]["config"] is minimal_training_config

        # Validate Lightning orchestrator was invoked
        assert lightning_trainer_called["called"], (
            "train_cdi_model_torch MUST delegate to Lightning trainer"
        )
        assert lightning_trainer_called["config"] is minimal_training_config

        # Validate results dict structure
        assert "history" in results
        assert "train_container" in results
        assert "test_container" in results

    def test_lightning_dataloader_tensor_dict_structure(
        self,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        CRITICAL PARITY TEST: Lightning dataloaders must yield TensorDict-style batches.

        Requirement: ADR-003-BACKEND-API Phase C4.D3 — _build_lightning_dataloaders must
        produce batches matching the contract expected by PtychoPINN_Lightning.compute_loss
        (ptycho_torch/model.py:1118-1128).

        Expected batch structure:
        - batch[0]: dict-like with keys ['images', 'coords_relative',
                    'rms_scaling_constant', 'physics_scaling_constant']
        - batch[1]: probe tensor
        - batch[2]: scaling constant tensor

        Red-phase contract:
        - _build_lightning_dataloaders currently wraps tensors in TensorDataset
        - This yields (Tensor, Tensor) tuples, causing IndexError in compute_loss
        - Must refactor to use TensorDictDataLoader + Collate_Lightning pattern
        - Reference: ptycho_torch/dataloader.py:771-856 (TensorDictDataLoader + Collate_Lightning)

        Test mechanism:
        - Call _build_lightning_dataloaders with minimal container fixture
        - Extract first batch from train_loader
        - Assert batch structure matches compute_loss expectations
        - Validate presence of required keys in batch[0] dict
        """
        # Import the module under test
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import update_legacy_dict
        from ptycho import params
        import torch

        # Initialize params.cfg via CONFIG-001
        update_legacy_dict(params.cfg, minimal_training_config)

        # Create minimal container fixture matching _ensure_container output
        # (duck-typed dict for testing; production uses PtychoDataContainerTorch)
        N = minimal_training_config.model.N
        gridsize = minimal_training_config.model.gridsize
        n_samples = 8  # Small batch for testing
        n_channels = gridsize * gridsize

        train_container = {
            "X": torch.randn(n_samples, n_channels, N, N, dtype=torch.float32),
            "coords_nominal": torch.randn(n_samples, n_channels, 1, 2, dtype=torch.float32),
            "coords_relative": torch.randn(n_samples, n_channels, 1, 2, dtype=torch.float32),
            "rms_scaling_constant": torch.ones(n_samples, 1, 1, 1, dtype=torch.float32),
            "physics_scaling_constant": torch.ones(n_samples, 1, 1, 1, dtype=torch.float32),
            "probe": torch.randn(N, N, dtype=torch.complex64),
            "scaling_constant": torch.ones(1, dtype=torch.float32),
        }

        # Call _build_lightning_dataloaders
        train_loader, _ = torch_components._build_lightning_dataloaders(
            train_container=train_container,
            test_container=None,
            config=minimal_training_config
        )

        # Extract first batch
        batch = next(iter(train_loader))

        # Validate batch is a tuple with 3 elements
        assert isinstance(batch, (tuple, list)), (
            f"Batch must be tuple/list, got {type(batch)}"
        )
        assert len(batch) == 3, (
            f"Batch must have 3 elements (tensor_dict, probe, scaling), got {len(batch)}"
        )

        # Validate batch[0] is dict-like with required keys
        tensor_dict = batch[0]
        assert hasattr(tensor_dict, '__getitem__'), (
            "batch[0] must support dict-like indexing (TensorDict or dict)"
        )

        required_keys = ['images', 'coords_relative', 'rms_scaling_constant', 'physics_scaling_constant']
        for key in required_keys:
            assert key in tensor_dict or hasattr(tensor_dict, key), (
                f"batch[0] must contain key '{key}' for compute_loss compatibility. "
                f"Available keys: {list(tensor_dict.keys()) if hasattr(tensor_dict, 'keys') else 'N/A'}"
            )

        # Validate batch[1] and batch[2] are tensors
        assert isinstance(batch[1], torch.Tensor), (
            f"batch[1] (probe) must be torch.Tensor, got {type(batch[1])}"
        )
        assert isinstance(batch[2], torch.Tensor), (
            f"batch[2] (scaling) must be torch.Tensor, got {type(batch[2])}"
        )

        # Validate tensor shapes are reasonable
        images = tensor_dict['images']
        assert images.ndim == 4, (
            f"batch[0]['images'] must be 4D (batch, channels, H, W), got shape {images.shape}"
        )

    def test_lightning_poisson_count_contract(
        self,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data,
        tmp_path
    ):
        """
        CRITICAL PARITY TEST: Poisson loss must accept photon counts, not raw amplitudes.

        Requirement: ADR-003-BACKEND-API Phase C4.D3 — PyTorch Poisson loss path must
        replicate TensorFlow behavior: square amplitudes and apply nphotons scaling
        before computing log-likelihood. Current implementation passes normalized
        amplitude floats directly to torch.distributions.Poisson, violating the
        integer support constraint.

        TensorFlow behavior (ptycho/model.py:541-561):
        - Squares amplitudes: pred², raw²
        - Applies intensity_scale rescaling
        - Passes squared values to tf.nn.log_poisson_loss

        PyTorch current behavior (ptycho_torch/model.py:714-720):
        - PoissonIntensityLayer squares predictions: Lambda = amplitudes²
        - forward() receives raw amplitudes (float32, 0.0-0.07 range)
        - Poisson.log_prob(raw) raises ValueError: "expected within support IntegerGreaterThan(0)"

        Red-phase contract:
        - Test constructs minimal Lightning module + dataloader
        - Executes one forward + compute_loss pass
        - Current implementation MUST fail with Poisson support violation
        - Captures the exact error message for documentation

        Fix requirement:
        - PoissonIntensityLayer.forward() must convert both pred and raw to counts
        - Apply nphotons scaling via rms_scaling_constant (available in batch dict)
        - Square amplitudes before Poisson.log_prob()
        - Reference: ptycho/loader.py:355-363 (TF normalization pattern)

        Test mechanism:
        - Initialize params.cfg via CONFIG-001
        - Build minimal container + Lightning module
        - Call compute_loss with one batch (no full trainer needed)
        - Assert ValueError is raised with "support" in message (RED phase)
        - After fix: Assert loss is finite scalar (GREEN phase)
        """
        # Import required modules
        from ptycho_torch.workflows import components as torch_components
        from ptycho_torch.model import PtychoPINN_Lightning
        from ptycho.config.config import update_legacy_dict
        from ptycho import params
        import torch

        # CONFIG-001: Initialize params.cfg before data operations
        update_legacy_dict(params.cfg, minimal_training_config)

        # Set output_dir for checkpoint path resolution
        minimal_training_config.output_dir = tmp_path / "lightning_poisson_test"
        minimal_training_config.output_dir.mkdir(parents=True, exist_ok=True)

        # Build minimal container (uses _ensure_container internally)
        train_container = torch_components._ensure_container(
            dummy_raw_data,
            minimal_training_config
        )

        # Build Lightning dataloaders
        train_loader, _ = torch_components._build_lightning_dataloaders(
            train_container=train_container,
            test_container=None,
            config=minimal_training_config
        )

        # Instantiate Lightning module with PyTorch config objects
        from ptycho_torch.config_params import DataConfig, ModelConfig as PTModelConfig, TrainingConfig as PTTrainingConfig, InferenceConfig

        # Create PyTorch config objects matching the TensorFlow configuration
        pt_data_config = DataConfig(
            N=minimal_training_config.model.N,
            grid_size=(minimal_training_config.model.gridsize, minimal_training_config.model.gridsize),
        )

        pt_model_config = PTModelConfig(
            mode='Unsupervised',  # 'pinn' in TF maps to 'Unsupervised' in PyTorch
            n_filters_scale=minimal_training_config.model.n_filters_scale,
        )

        pt_training_config = PTTrainingConfig(
            epochs=minimal_training_config.nepochs,
            batch_size=minimal_training_config.batch_size,
            learning_rate=1e-4,
            nll=True,  # Enable Poisson loss (this is the key test requirement)
        )

        pt_inference_config = InferenceConfig()

        lightning_module = PtychoPINN_Lightning(
            model_config=pt_model_config,
            data_config=pt_data_config,
            training_config=pt_training_config,
            inference_config=pt_inference_config
        )

        # Extract one batch from dataloader
        batch_iter = iter(train_loader)
        batch = next(batch_iter)

        # GREEN PHASE: After fixing PoissonIntensityLayer to convert amplitudes→intensities,
        # this test should pass without raising ValueError
        total_loss = lightning_module.compute_loss(batch)

        # Validate loss computation succeeded (returns scalar Tensor, not dict)
        assert isinstance(total_loss, torch.Tensor), \
            f"compute_loss must return Tensor, got {type(total_loss)}"

        assert torch.isfinite(total_loss), \
            f"Poisson loss must be finite after amplitude→intensity conversion, " \
            f"got: {total_loss}"

        # Validate loss is a positive scalar (negative log-likelihood should be positive)
        assert total_loss > 0, \
            f"Negative log-likelihood should be positive, got: {total_loss}"

    def test_lightning_training_respects_gridsize(
        self,
        monkeypatch,
        params_cfg_snapshot,
        dummy_raw_data
    ):
        """
        Gridsize Channel Parity Test — _train_with_lightning MUST propagate gridsize to model channels.

        Requirement: docs/findings.md#BUG-TF-001 — gridsize > 1 yields channel mismatches
        unless params.cfg['gridsize'] AND PyTorch model configs synchronize C = gridsize**2.

        Design contract (phase_c4d_blockers/plan.md §B1-B2):
        - When config.model.gridsize=2, PyTorch DataConfig MUST set C=4 (2×2)
        - ModelConfig MUST set C_model=4 and C_forward=4 to match grouping
        - PtychoPINN_Lightning first conv layer MUST expect 4 input channels (not 1)

        Test mechanism:
        - Create TrainingConfig with gridsize=2
        - Monkeypatch PtychoPINN_Lightning to inspect first conv layer input channels
        - Invoke _train_with_lightning
        - Assert first conv layer has in_channels == gridsize**2 == 4

        Expected failure mode (RED phase):
        - _train_with_lightning manually builds PTDataConfig with default C=1
        - Lightning module created with C_model=1 → first conv expects 1 channel
        - Assertion fails: in_channels=1 != expected 4

        GREEN phase fix:
        - Refactor _train_with_lightning to reuse config_factory.create_training_payload
        - Factory propagates gridsize → C via grid_size tuple → C = grid_size[0]*grid_size[1]
        - ModelConfig receives C_model=4, Lightning module conv layers match
        """
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho_torch.workflows import components as torch_components

        # Spy to track Lightning module instantiation and inspect model structure
        lightning_init_spy = {"called": False, "first_conv_in_channels": None}

        # Store original PtychoPINN_Lightning before patching
        from ptycho_torch.model import PtychoPINN_Lightning as OriginalLightningModule

        def mock_lightning_init(model_config, data_config, training_config, inference_config):
            """Spy that captures PtychoPINN_Lightning model structure."""
            import torch
            import lightning.pytorch as L

            # Record that init was called
            lightning_init_spy["called"] = True

            # Use the ORIGINAL Lightning module (not the patched version) to inspect architecture
            module = OriginalLightningModule(
                model_config=model_config,
                data_config=data_config,
                training_config=training_config,
                inference_config=inference_config
            )

            # Extract first conv layer input channels from the model
            # The PtychoPINN architecture: model.autoencoder.encoder contains Conv2d layers
            # Find the first Conv2d layer and record its in_channels
            for layer in module.model.autoencoder.encoder.modules():
                if isinstance(layer, torch.nn.Conv2d):
                    lightning_init_spy["first_conv_in_channels"] = layer.in_channels
                    break

            # Return a stub module to avoid full training execution
            class StubLightningModule(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    self.dummy_param = torch.nn.Parameter(torch.zeros(1))

                def training_step(self, batch, batch_idx):
                    return torch.tensor(0.0, requires_grad=True)

                def configure_optimizers(self):
                    return torch.optim.Adam(self.parameters(), lr=1e-3)

            return StubLightningModule()

        # Monkeypatch Lightning module constructor
        monkeypatch.setattr(
            "ptycho_torch.model.PtychoPINN_Lightning",
            mock_lightning_init
        )

        # Create minimal dummy NPZ file for factory validation
        # (Factory validates file exists before proceeding with config construction)
        import tempfile
        tmpdir = Path(tempfile.mkdtemp())
        dummy_npz = tmpdir / "dummy_train.npz"

        # Create minimal NPZ with required keys for DATA-001 compliance
        np.savez(
            dummy_npz,
            diffraction=np.ones((10, 64, 64), dtype=np.float32),
            xcoords=np.linspace(0, 9, 10),
            ycoords=np.linspace(0, 9, 10),
            probeGuess=np.ones((64, 64), dtype=np.complex64),
            objectGuess=np.ones((128, 128), dtype=np.complex64),
        )

        # Create TrainingConfig with gridsize=2 (requires 4 input channels)
        model_config = ModelConfig(
            N=64,
            gridsize=2,  # CRITICAL: 2×2 grouping → 4 channels expected
            model_type='pinn',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=dummy_npz,  # Use temp file for factory validation
            test_data_file=dummy_npz,   # Reuse for test data
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            nepochs=2,
        )

        # Create minimal train_container
        train_container = {
            "X": np.ones((10, 64, 64)),
            "Y": np.ones((10, 64, 64), dtype=np.complex64),
        }

        # Call _train_with_lightning with gridsize=2 config
        results = torch_components._train_with_lightning(
            train_container=train_container,
            test_container=None,
            config=training_config
        )

        # Assert Lightning module was instantiated
        assert lightning_init_spy["called"], \
            "_train_with_lightning must instantiate PtychoPINN_Lightning module"

        # RED PHASE ASSERTION: Channel count must match gridsize**2
        expected_channels = training_config.model.gridsize ** 2
        actual_channels = lightning_init_spy["first_conv_in_channels"]

        assert actual_channels == expected_channels, (
            f"Lightning model first conv layer MUST have in_channels={expected_channels} "
            f"when gridsize={training_config.model.gridsize}, but got in_channels={actual_channels}. "
            f"This indicates _train_with_lightning is not propagating gridsize to PyTorch configs. "
            f"See docs/findings.md#BUG-TF-001 for channel mismatch pattern."
        )

    def test_coords_relative_layout(
        self,
        monkeypatch,
        params_cfg_snapshot,
        dummy_raw_data
    ):
        """
        Coords Relative Tensor Layout Test — dataloader MUST provide (batch, C, 1, 2) shaped coords_relative.

        Requirement: input.md Do Now #1 (2025-10-20 Phase C4.D.B2) — The current axis ordering bug
        causes coords_relative to arrive as (batch, 1, 2, C) which triggers Translation reshape crash
        when reassemble_patches_position_real broadcasts with norm_factor.

        Design contract (phase_c4d_coords_debug/summary.md):
        - _build_lightning_dataloaders MUST permute coords_relative from raw (batch, 1, 2, C)
          to expected (batch, C, 1, 2) before batching
        - Tensors MUST be .contiguous() before .view() operations
        - This layout matches the TensorFlow tf_helper contract for Translation inputs

        Test mechanism:
        - Create TrainingConfig with gridsize=2 (C=4)
        - Build Lightning dataloaders via _build_lightning_dataloaders
        - Extract a batch from train_loader
        - Assert batch['coords_relative'].shape == (batch_size, 4, 1, 2)

        Expected failure mode (RED phase):
        - Current implementation hands coords_relative shaped (batch, 1, 2, 4)
        - Assertion fails: shape[-3:] == (1, 2, 4) != expected (4, 1, 2)

        GREEN phase fix:
        - Refactor _build_lightning_dataloaders to permute coords_relative axes
        - Apply .contiguous() before batching to keep view() happy
        - Rerun test → assertion passes with (batch, 4, 1, 2) shape
        """
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import update_legacy_dict
        from ptycho import params
        import torch

        # Create TrainingConfig with gridsize=2 (C = 2×2 = 4)
        model_config = ModelConfig(
            N=64,
            gridsize=2,  # C = 4 channels
            model_type='pinn',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            nepochs=2,
            batch_size=4,  # Explicit batch size for shape check
        )

        # Populate params.cfg (CONFIG-001 requirement)
        update_legacy_dict(params.cfg, training_config)

        # Convert RawData to PtychoDataContainerTorch via _ensure_container
        # This will call generate_grouped_data internally
        train_container = torch_components._ensure_container(
            data=dummy_raw_data,
            config=training_config
        )

        # Build Lightning dataloaders (the function under test)
        train_loader, _ = torch_components._build_lightning_dataloaders(
            train_container=train_container,
            test_container=None,
            config=training_config
        )

        # Extract one batch from the dataloader
        # Batch structure: (tensor_dict, probe, scaling) per Phase D2.C contract
        batch = next(iter(train_loader))
        tensor_dict = batch[0]  # Extract the first element (dict with coords_relative, etc.)

        # RED PHASE ASSERTION: coords_relative must be shaped (batch, C, 1, 2)
        expected_C = training_config.model.gridsize ** 2  # 2×2 = 4
        expected_shape_suffix = (expected_C, 1, 2)

        actual_shape = tensor_dict['coords_relative'].shape
        actual_shape_suffix = actual_shape[-3:]  # Extract last 3 dims (C, 1, 2)

        assert actual_shape_suffix == expected_shape_suffix, (
            f"coords_relative tensor MUST have shape suffix (C, 1, 2) where C={expected_C}, "
            f"but got shape={actual_shape} with suffix={actual_shape_suffix}. "
            f"Current axis order causes Translation.view() to fail in reassemble_patches_position_real. "
            f"See plans/active/ADR-003-BACKEND-API/reports/2025-10-20T103200Z/phase_c4d_coords_debug/summary.md"
        )


class TestWorkflowsComponentsRun:
    """
    Phase D2.C inference + stitching tests — validate run_cdi_example_torch orchestration.

    These tests validate that run_cdi_example_torch properly orchestrates training
    and optional stitching workflows per specs/ptychodus_api_spec.md §4.5 and TF
    baseline ptycho/workflows/components.py:676-723.
    """

    @pytest.fixture
    def params_cfg_snapshot(self):
        """Snapshot and restore params.cfg state across tests."""
        from ptycho import params
        original = params.cfg.copy()
        yield params.cfg
        params.cfg.clear()
        params.cfg.update(original)

    @pytest.fixture
    def minimal_training_config(self):
        """Create minimal TrainingConfig fixture for inference tests."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            nepochs=2,
        )

        return training_config

    @pytest.fixture
    def dummy_raw_data(self):
        """Create minimal RawData fixture for testing."""
        from ptycho.raw_data import RawData

        # Create minimal synthetic data
        nsamples = 10
        N = 64

        dummy_coords = np.linspace(0, 9, nsamples)
        dummy_diff = np.random.rand(nsamples, N, N).astype(np.float32)
        dummy_probe = np.ones((N, N), dtype=np.complex64)
        dummy_scan_index = np.arange(nsamples, dtype=int)

        return RawData(
            xcoords=dummy_coords,
            ycoords=dummy_coords,
            xcoords_start=dummy_coords,
            ycoords_start=dummy_coords,
            diff3d=dummy_diff,
            probeGuess=dummy_probe,
            scan_index=dummy_scan_index,
        )

    def test_run_cdi_example_invokes_training(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        CRITICAL PARITY TEST: run_cdi_example_torch must invoke training orchestration.

        Requirement: Phase D2.C must implement full workflow orchestration following
        TensorFlow baseline ptycho/workflows/components.py:676-723 and mirroring
        the reconstructor lifecycle per specs/ptychodus_api_spec.md §4.5.

        Red-phase contract:
        - Entry signature: run_cdi_example_torch(train_data, test_data, config, do_stitching=False, ...)
        - MUST call train_cdi_model_torch(train_data, test_data, config) first
        - When do_stitching=False: return (None, None, results_dict)
        - When do_stitching=True + test_data: invoke reassemble helper, return (amp, phase, results)
        - results dict MUST contain keys from training (history, containers)

        Test mechanism:
        - Use monkeypatch to spy on train_cdi_model_torch call
        - Pass minimal RawData + do_stitching=False (no inference path required)
        - Assert train_cdi_model_torch was invoked with correct args
        - Validate return signature matches TensorFlow baseline
        """
        # Import the module under test
        from ptycho_torch.workflows import components as torch_components

        # Spy flag to track train_cdi_model_torch invocation
        train_cdi_model_torch_called = {"called": False, "args": None}

        def mock_train_cdi_model_torch(train_data, test_data, config, execution_config=None):
            """Spy that records train_cdi_model_torch invocation."""
            train_cdi_model_torch_called["called"] = True
            train_cdi_model_torch_called["args"] = (train_data, test_data, config, execution_config)
            # Return minimal training results dict
            return {
                "history": {"train_loss": [0.5, 0.3]},
                "train_container": {"sentinel": "train"},
                "test_container": None,
            }

        # Patch train_cdi_model_torch
        monkeypatch.setattr(
            "ptycho_torch.workflows.components.train_cdi_model_torch",
            mock_train_cdi_model_torch
        )

        # Call run_cdi_example_torch with do_stitching=False (Phase D2.C red phase)
        recon_amp, recon_phase, results = torch_components.run_cdi_example_torch(
            train_data=dummy_raw_data,
            test_data=None,
            config=minimal_training_config,
            flip_x=False,
            flip_y=False,
            transpose=False,
            M=20,
            do_stitching=False,
        )

        # Validate train_cdi_model_torch was called
        assert train_cdi_model_torch_called["called"], (
            "run_cdi_example_torch MUST invoke train_cdi_model_torch"
        )

        # Validate correct arguments passed
        train_data_arg, test_data_arg, config_arg, exec_cfg_arg = train_cdi_model_torch_called["args"]
        assert train_data_arg is dummy_raw_data
        assert test_data_arg is None
        assert config_arg is minimal_training_config

        # Validate return signature (do_stitching=False → None, None, results)
        assert recon_amp is None, (
            "When do_stitching=False, recon_amp should be None"
        )
        assert recon_phase is None, (
            "When do_stitching=False, recon_phase should be None"
        )
        assert "history" in results, "results dict must contain training history"

    def test_run_cdi_example_persists_models(
        self,
        monkeypatch,
        tmp_path,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        REGRESSION TEST: run_cdi_example_torch must persist models when config.output_dir set.

        Requirement: Phase D4.B2 — validate PyTorch orchestration maintains persistence
        parity with TensorFlow baseline per specs/ptychodus_api_spec.md:§4.6.

        TensorFlow baseline (ptycho/workflows/components.py:709-723):
        - When config.output_dir is provided, calls save_model() or ModelManager.save()
        - Produces wts.h5.zip archive in output_dir with dual-model bundle
        - Persistence happens after training completes successfully

        Red-phase expectation:
        - run_cdi_example_torch currently does NOT call save_torch_bundle
        - Once Phase D4.C1 complete, SHOULD invoke save_torch_bundle when output_dir set
        - Test will FAIL until orchestration wiring is complete

        Test mechanism:
        - Monkeypatch save_torch_bundle to spy on invocation
        - Set config.output_dir to tmp_path
        - Call run_cdi_example_torch
        - Validate save_torch_bundle was called with correct models dict + base_path
        """
        # Import the module under test
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Spy flag to track save_torch_bundle invocation
        save_torch_bundle_called = {"called": False, "args": None, "kwargs": None}

        def mock_save_torch_bundle(models_dict, base_path, config, **kwargs):
            """Spy that records save_torch_bundle invocation."""
            save_torch_bundle_called["called"] = True
            save_torch_bundle_called["args"] = (models_dict, base_path, config)
            save_torch_bundle_called["kwargs"] = kwargs

        # Monkeypatch save_torch_bundle
        monkeypatch.setattr(
            "ptycho_torch.workflows.components.save_torch_bundle",
            mock_save_torch_bundle
        )

        # Monkeypatch train_cdi_model_torch to return minimal results with models
        def mock_train_cdi_model_torch(train_data, test_data, config, execution_config=None):
            """Return stub results including models dict for persistence."""
            return {
                "history": {"train_loss": [0.5, 0.3]},
                "train_container": {"sentinel": "train"},
                "test_container": None,
                "models": {
                    'autoencoder': {'_sentinel': 'trained_autoencoder'},
                    'diffraction_to_obj': {'_sentinel': 'trained_diffraction'},
                },
            }

        monkeypatch.setattr(
            "ptycho_torch.workflows.components.train_cdi_model_torch",
            mock_train_cdi_model_torch
        )

        # Create config with output_dir set
        model_config = ModelConfig(N=64, gridsize=2, model_type='pinn')
        config_with_output = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            output_dir=tmp_path,  # Enable persistence
        )

        # Call run_cdi_example_torch
        recon_amp, recon_phase, results = torch_components.run_cdi_example_torch(
            train_data=dummy_raw_data,
            test_data=None,
            config=config_with_output,
            flip_x=False,
            flip_y=False,
            transpose=False,
            M=20,
            do_stitching=False,
        )

        # Validate save_torch_bundle was called
        # Red phase: this assertion WILL FAIL because persistence wiring not yet implemented
        assert save_torch_bundle_called["called"], (
            "run_cdi_example_torch MUST call save_torch_bundle when config.output_dir is set. "
            "Phase D4.B2 red phase: persistence wiring not yet implemented. This test documents "
            "the expected behavior and will pass once Phase D4.C1 adds persistence call."
        )

        # Validate correct arguments passed (green phase validation)
        if save_torch_bundle_called["called"]:
            models_dict_arg, base_path_arg, config_arg = save_torch_bundle_called["args"]

            assert 'autoencoder' in models_dict_arg, (
                "save_torch_bundle MUST receive models dict with 'autoencoder' key"
            )
            assert 'diffraction_to_obj' in models_dict_arg, (
                "save_torch_bundle MUST receive models dict with 'diffraction_to_obj' key"
            )

            assert str(tmp_path) in str(base_path_arg), (
                "save_torch_bundle base_path MUST be within config.output_dir"
            )

            assert config_arg is config_with_output, (
                "save_torch_bundle MUST receive the TrainingConfig instance"
            )

    def test_load_inference_bundle_handles_bundle(
        self,
        monkeypatch,
        tmp_path,
        params_cfg_snapshot,
        minimal_training_config
    ):
        """
        REGRESSION TEST: load_inference_bundle_torch must delegate to load_torch_bundle.

        Requirement: Phase D4.B2 — validate PyTorch inference loading maintains parity
        with TensorFlow baseline per specs/ptychodus_api_spec.md:§4.5.

        TensorFlow baseline (ptycho/workflows/components.py:94-174):
        - load_inference_bundle unpacks wts.h5.zip via ModelManager.load_multiple_models
        - Restores params.cfg before model reconstruction (CONFIG-001)
        - Returns (models_dict, params_dict) tuple

        Red-phase expectation:
        - load_inference_bundle_torch currently raises NotImplementedError
        - Once Phase D4.C2 complete, SHOULD invoke load_torch_bundle shim
        - Test will FAIL until loader delegation is wired

        Test mechanism:
        - Monkeypatch load_torch_bundle to spy on invocation
        - Call load_inference_bundle_torch with bundle path
        - Validate load_torch_bundle was called with correct base_path
        - Validate params.cfg was updated via CONFIG-001 gate
        """
        # Import the module under test
        from ptycho_torch.workflows import components as torch_components
        from ptycho import params

        # Spy flag to track load_torch_bundle invocation
        load_torch_bundle_called = {"called": False, "args": None}

        def mock_load_torch_bundle(base_path, model_name='diffraction_to_obj'):
            """Spy that records load_torch_bundle invocation."""
            load_torch_bundle_called["called"] = True
            load_torch_bundle_called["args"] = (base_path, model_name)

            # Simulate params.cfg restoration (CONFIG-001 requirement)
            restored_params = {
                'N': 64,
                'gridsize': 2,
                'model_type': 'pinn',
                'nphotons': 1e9,
            }
            params.cfg.update(restored_params)

            # Return sentinel model + params
            return (
                {'_sentinel': 'loaded_model'},
                restored_params
            )

        # Monkeypatch load_torch_bundle
        monkeypatch.setattr(
            "ptycho_torch.workflows.components.load_torch_bundle",
            mock_load_torch_bundle
        )

        # Clear params.cfg to simulate fresh process
        params.cfg.clear()
        assert params.cfg.get('N') is None, "Sanity check: params.cfg should be empty"

        # Call load_inference_bundle_torch
        # Red phase: expect NotImplementedError
        # Green phase: expect successful return
        try:
            models_dict, params_dict = torch_components.load_inference_bundle_torch(
                bundle_dir=str(tmp_path / "test_bundle")
            )

            # Green phase assertions (once Phase D4.C2 complete)
            assert load_torch_bundle_called["called"], (
                "load_inference_bundle_torch MUST delegate to load_torch_bundle"
            )

            base_path_arg, model_name_arg = load_torch_bundle_called["args"]
            assert str(tmp_path / "test_bundle") in str(base_path_arg), (
                "load_torch_bundle MUST receive correct bundle_dir path"
            )

            # Validate params.cfg was updated (CONFIG-001 requirement)
            assert params.cfg.get('N') == 64, (
                "CONFIG-001: params.cfg['N'] must be restored during load"
            )
            assert params.cfg.get('gridsize') == 2, (
                "CONFIG-001: params.cfg['gridsize'] must be restored"
            )

            # Validate return values
            assert models_dict is not None, "load_inference_bundle_torch MUST return models_dict"
            assert params_dict is not None, "load_inference_bundle_torch MUST return params_dict"

        except NotImplementedError as e:
            # Red phase: document expected failure
            if 'load_inference_bundle_torch' in str(e) or 'not yet implemented' in str(e):
                pytest.xfail(
                    "Phase D4.B2 red phase: load_inference_bundle_torch delegation not yet wired. "
                    "Expected NotImplementedError raised. This test will pass once Phase D4.C2 "
                    "implements loader delegation."
                )
            else:
                raise  # Unexpected NotImplementedError, re-raise


class TestTrainWithLightningRed:
    """
    Phase B.B1 Lightning orchestration tests (RED phase).

    These tests encode the Lightning orchestration contract before implementation,
    per plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-18T000606Z/phase_d2_completion/phase_b_test_design.md.

    Design requirements:
    1. _train_with_lightning MUST instantiate PtychoPINN_Lightning with all four config objects
    2. _train_with_lightning MUST invoke Trainer.fit with dataloaders from containers
    3. _train_with_lightning MUST return results dict with 'models' key for persistence

    Red-phase expectation:
    - All three tests WILL FAIL because _train_with_lightning is currently a stub
    - Stub returns minimal dict without Lightning module instantiation or training
    - Once Phase B2 implements Lightning orchestration, tests will turn green
    """

    @pytest.fixture
    def params_cfg_snapshot(self):
        """Snapshot and restore params.cfg state across tests."""
        from ptycho import params
        original = params.cfg.copy()
        yield params.cfg
        params.cfg.clear()
        params.cfg.update(original)

    @pytest.fixture
    def minimal_training_config(self):
        """Create minimal TrainingConfig fixture for Lightning tests."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            nepochs=2,
        )

        return training_config

    @pytest.fixture
    def dummy_raw_data(self):
        """Create minimal RawData fixture for testing."""
        from ptycho.raw_data import RawData

        nsamples = 10
        N = 64

        dummy_coords = np.linspace(0, 9, nsamples)
        dummy_diff = np.random.rand(nsamples, N, N).astype(np.float32)
        dummy_probe = np.ones((N, N), dtype=np.complex64)
        dummy_scan_index = np.arange(nsamples, dtype=int)

        return RawData(
            xcoords=dummy_coords,
            ycoords=dummy_coords,
            xcoords_start=dummy_coords,
            ycoords_start=dummy_coords,
            diff3d=dummy_diff,
            probeGuess=dummy_probe,
            scan_index=dummy_scan_index,
        )

    def test_train_with_lightning_instantiates_module(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        RED TEST 1: _train_with_lightning MUST instantiate PtychoPINN_Lightning with four configs.

        Requirement: specs/ptychodus_api_spec.md:187 reconstructor lifecycle requires
        trained module handles with serialized config for checkpoint reload.

        Design contract (phase_b_test_design.md §1):
        - _train_with_lightning receives (train_container, test_container, config)
        - MUST construct ptycho_torch.model.PtychoPINN_Lightning(__init__)
        - Constructor MUST receive exactly (model_config, data_config, training_config, inference_config)
        - This ensures checkpoint.load can reconstruct module without external state

        Test mechanism:
        - Monkeypatch PtychoPINN_Lightning to spy on __init__ args
        - Create minimal containers (dicts acceptable for red phase)
        - Invoke _train_with_lightning
        - Assert spy recorded all four config objects

        Expected red-phase failure:
        - Stub never instantiates Lightning module
        - Spy not called → assertion fails
        """
        from ptycho_torch.workflows import components as torch_components

        # Spy to track Lightning module instantiation
        lightning_init_called = {"called": False, "args": None}

        def mock_lightning_init(model_config, data_config, training_config, inference_config):
            """Spy that records PtychoPINN_Lightning.__init__ args."""
            import torch
            import lightning.pytorch as L

            lightning_init_called["called"] = True
            lightning_init_called["args"] = (model_config, data_config, training_config, inference_config)

            # Return minimal stub module inheriting from LightningModule
            # to satisfy Lightning's isinstance check during trainer.fit
            class StubLightningModule(L.LightningModule):
                def __init__(self):
                    super().__init__()
                    # Minimal parameter to satisfy Lightning requirements
                    self.dummy_param = torch.nn.Parameter(torch.zeros(1))

                def training_step(self, batch, batch_idx):
                    """Minimal training step required by Lightning."""
                    # Return deterministic zero loss to make Trainer complete immediately
                    return torch.tensor(0.0, requires_grad=True)

                def configure_optimizers(self):
                    """Minimal optimizer required by Lightning."""
                    return torch.optim.Adam(self.parameters(), lr=1e-3)

            return StubLightningModule()

        # Monkeypatch Lightning module constructor
        # Note: actual import path will be ptycho_torch.model.PtychoPINN_Lightning
        # but we monkeypatch at the call site within _train_with_lightning
        monkeypatch.setattr(
            "ptycho_torch.model.PtychoPINN_Lightning",
            mock_lightning_init
        )

        # Create minimal train_container (dict placeholder for red phase)
        # Phase C adapters will produce actual PtychoDataContainerTorch
        train_container = {
            "X": np.ones((10, 64, 64)),
            "Y": np.ones((10, 64, 64), dtype=np.complex64),
        }

        # Call _train_with_lightning
        results = torch_components._train_with_lightning(
            train_container=train_container,
            test_container=None,
            config=minimal_training_config
        )

        # RED PHASE ASSERTION (will fail until Phase B2 implements)
        assert lightning_init_called["called"], (
            "_train_with_lightning MUST instantiate PtychoPINN_Lightning module. "
            "Phase B.B1 red phase: stub does not create module yet. "
            "This test documents the expected behavior and will pass once Phase B2 "
            "implements Lightning orchestration."
        )

        # Green phase validation (once Phase B2 complete)
        if lightning_init_called["called"]:
            model_cfg, data_cfg, train_cfg, infer_cfg = lightning_init_called["args"]

            # Validate all four config objects were passed
            # (Exact types/values validated in green phase; red phase just checks arity)
            assert model_cfg is not None, "model_config must be provided to Lightning module"
            assert data_cfg is not None, "data_config must be provided to Lightning module"
            assert train_cfg is not None, "training_config must be provided to Lightning module"
            assert infer_cfg is not None, "inference_config must be provided to Lightning module"

    def test_train_with_lightning_runs_trainer_fit(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        RED TEST 2: _train_with_lightning MUST invoke Trainer.fit with dataloaders.

        Requirement: docs/workflows/pytorch.md §5 Lightning trainer expectations
        require Trainer.fit orchestration with train/val dataloaders.

        Design contract (phase_b_test_design.md §2):
        - _train_with_lightning MUST construct lightning.pytorch.Trainer
        - MUST invoke trainer.fit(module, train_dataloader, val_dataloader)
        - Dataloaders MUST be derived from provided train/test containers
        - Validation dataloader is None when test_container is None

        Test mechanism:
        - Monkeypatch Trainer constructor to return stub exposing fit_called flag
        - Monkeypatch dataloader builders (future helpers) with sentinels
        - Invoke _train_with_lightning
        - Assert Trainer.fit was called with correct dataloaders

        Expected red-phase failure:
        - Stub never constructs Trainer or calls fit
        - fit_called flag remains False → assertion fails
        """
        from ptycho_torch.workflows import components as torch_components

        # Spy to track Trainer.fit invocation
        trainer_fit_called = {"called": False, "args": None, "kwargs": None}

        class MockTrainer:
            """Stub Trainer that records fit() calls."""
            def fit(self, module, train_dataloaders=None, val_dataloaders=None, **kwargs):
                trainer_fit_called["called"] = True
                trainer_fit_called["args"] = (module, train_dataloaders, val_dataloaders)
                trainer_fit_called["kwargs"] = kwargs

        # Monkeypatch Lightning Trainer
        monkeypatch.setattr(
            "lightning.pytorch.Trainer",
            lambda **kwargs: MockTrainer()
        )

        # Monkeypatch Lightning module to prevent import errors
        class StubLightningModule:
            automatic_optimization = True

            def save_hyperparameters(self):
                pass

        monkeypatch.setattr(
            "ptycho_torch.model.PtychoPINN_Lightning",
            lambda *args, **kwargs: StubLightningModule()
        )

        # Create sentinel dataloaders (Phase B2 will wire real loader builders)
        sentinel_train_loader = {"_sentinel": "train_dataloader"}
        sentinel_val_loader = None  # test_container is None

        # Monkeypatch future dataloader builder helper
        # (Phase B2 will add _build_lightning_dataloaders or similar)
        def mock_build_dataloaders(container, config, shuffle=True):
            """Sentinel that returns mock dataloader."""
            if container is not None:
                return sentinel_train_loader
            return None

        # For red phase, assume _train_with_lightning will eventually call helper
        # For now, test just validates fit() invocation pattern

        # Create minimal containers
        train_container = {"X": np.ones((10, 64, 64))}

        # Call _train_with_lightning
        results = torch_components._train_with_lightning(
            train_container=train_container,
            test_container=None,
            config=minimal_training_config
        )

        # RED PHASE ASSERTION (will fail until Phase B2 implements)
        assert trainer_fit_called["called"], (
            "_train_with_lightning MUST invoke Trainer.fit with dataloaders. "
            "Phase B.B1 red phase: stub does not construct Trainer or call fit. "
            "This test documents the expected behavior and will pass once Phase B2 "
            "implements Lightning orchestration."
        )

        # Green phase validation (once Phase B2 complete)
        if trainer_fit_called["called"]:
            module_arg, train_loader_arg, val_loader_arg = trainer_fit_called["args"]

            assert module_arg is not None, "Trainer.fit must receive Lightning module"
            # Dataloaders validation deferred to integration tests (requires real containers)

    def test_train_with_lightning_returns_models_dict(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        RED TEST 3: _train_with_lightning MUST return results dict with 'models' key.

        Requirement: Phase D4 persistence tests require trained module handles
        for save_torch_bundle orchestration (mirrors TensorFlow train_cdi_model).

        Design contract (phase_b_test_design.md §3):
        - _train_with_lightning returns Dict[str, Any]
        - Results dict MUST contain 'models' key
        - models['lightning_module'] (or models['diffraction_to_obj']) MUST point to trained module
        - This enables downstream save_torch_bundle to persist checkpoint

        Test mechanism:
        - Monkeypatch Lightning components to return stub module
        - Invoke _train_with_lightning
        - Assert results dict contains 'models' key with module handle

        Expected red-phase failure:
        - Stub returns only history/containers → missing 'models' key
        - Assertion fails
        """
        from ptycho_torch.workflows import components as torch_components

        # Stub Lightning module with sentinel identity
        class StubLightningModule:
            automatic_optimization = True

            def save_hyperparameters(self):
                pass

            _sentinel = "trained_lightning_module"

        stub_module = StubLightningModule()

        # Monkeypatch Lightning module constructor
        monkeypatch.setattr(
            "ptycho_torch.model.PtychoPINN_Lightning",
            lambda *args, **kwargs: stub_module
        )

        # Monkeypatch Trainer to skip actual training
        class MockTrainer:
            def fit(self, module, train_dataloaders=None, val_dataloaders=None, **kwargs):
                pass

        monkeypatch.setattr(
            "lightning.pytorch.Trainer",
            lambda **kwargs: MockTrainer()
        )

        # Create minimal containers
        train_container = {"X": np.ones((10, 64, 64))}

        # Call _train_with_lightning
        results = torch_components._train_with_lightning(
            train_container=train_container,
            test_container=None,
            config=minimal_training_config
        )

        # RED PHASE ASSERTION (will fail until Phase B2 implements)
        assert "models" in results, (
            "_train_with_lightning MUST return results dict with 'models' key. "
            "Phase B.B1 red phase: stub returns only history/containers without models. "
            "This test documents the expected behavior and will pass once Phase B2 "
            "implements Lightning orchestration with module persistence."
        )

        # Green phase validation (once Phase B2 complete)
        if "models" in results:
            models_dict = results["models"]

            # Accept either 'lightning_module' or 'diffraction_to_obj' as key
            # (naming decision deferred to Phase B2 implementation)
            assert any(
                key in models_dict for key in ['lightning_module', 'diffraction_to_obj']
            ), (
                "models dict MUST contain trained module under 'lightning_module' or "
                "'diffraction_to_obj' key for persistence"
            )

            # Validate module handle is not None
            module_handle = models_dict.get('lightning_module') or models_dict.get('diffraction_to_obj')
            assert module_handle is not None, "Module handle must not be None"


class TestReassembleCdiImageTorchGreen:
    """
    Phase D2.C4 green tests — PyTorch stitching path validation.

    These tests validate the implemented behavior of `_reassemble_cdi_image_torch`
    and `run_cdi_example_torch(..., do_stitching=True)` after Phase C3 implementation.

    Requirements (from inference_design.md + specs/ptychodus_api_spec.md §4.5):
    1. _reassemble_cdi_image_torch MUST run Lightning inference on test data
    2. MUST apply flip_x, flip_y, transpose coordinate transforms per TF parity
    3. MUST return (recon_amp, recon_phase, results) tuple
    4. run_cdi_example_torch(..., do_stitching=True) MUST delegate to stitching path

    Test Strategy:
    - Use mock Lightning module returning deterministic complex outputs
    - Validate amplitude/phase outputs are numpy arrays with expected shapes
    - Parametrize flip/transpose combinations to verify coordinate transforms
    - Keep focused guard test for train_results=None backward compatibility

    Artifacts:
    - Green log: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T084016Z/phase_d2_completion/pytest_stitch_green.log
    """

    @pytest.fixture
    def params_cfg_snapshot(self):
        """Snapshot and restore params.cfg state across tests."""
        from ptycho import params
        original = params.cfg.copy()
        yield params.cfg
        params.cfg.clear()
        params.cfg.update(original)

    @pytest.fixture
    def minimal_training_config(self, tmp_path):
        """Create minimal TrainingConfig for stitching tests."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
            amp_activation='silu',
            n_filters_scale=1,
        )

        return TrainingConfig(
            model=model_config,
            train_data_file=Path("dummy_train.npz"),
            test_data_file=Path("dummy_test.npz"),
            n_groups=10,
            batch_size=2,
            nepochs=1,
            nphotons=1e9,
            neighbor_count=4,
            output_dir=tmp_path,
        )

    @pytest.fixture
    def dummy_raw_data(self):
        """
        Create deterministic RawData fixture for stitching tests.

        Data characteristics:
        - 20 scan positions in 4x5 grid
        - Complex probe (64x64)
        - Complex object (128x128, significantly larger than probe per DATA-001)
        - Diffraction amplitudes (diff3d) normalized to [0, 1] range
        - Seed=0 for reproducibility
        """
        from ptycho.raw_data import RawData

        np.random.seed(0)
        n_scan = 20
        N = 64

        # Grid positions
        xcoords = np.tile(np.arange(5) * 10.0, 4)
        ycoords = np.repeat(np.arange(4) * 10.0, 5)

        # Normalized diffraction amplitudes (not intensity)
        diff3d = np.random.rand(n_scan, N, N).astype(np.float32) * 0.5

        # Complex probe and object
        probe = (np.random.rand(N, N) + 1j * np.random.rand(N, N)).astype(np.complex64)
        obj = (np.random.rand(128, 128) + 1j * np.random.rand(128, 128)).astype(np.complex64)

        # Scan index
        scan_index = np.arange(n_scan, dtype=int)

        return RawData(
            xcoords=xcoords,
            ycoords=ycoords,
            xcoords_start=xcoords,
            ycoords_start=ycoords,
            diff3d=diff3d,
            probeGuess=probe,
            scan_index=scan_index,
            objectGuess=obj,
        )

    @pytest.fixture
    def mock_lightning_module(self, minimal_training_config):
        """
        Create minimal Lightning module stub for stitching tests.

        Returns a lightweight mock that:
        - Inherits from lightning.LightningModule (satisfies isinstance checks)
        - Implements eval() method (no-op but required for eval mode)
        - Implements __call__(X) returning deterministic complex64 tensor
        - Output shape: (batch, gridsize**2, N, N) complex CHANNEL-FIRST to match real PyTorch models
        - Uses torch.ones for deterministic, finite outputs

        Implementation note:
        - Real PtychoPINN_Lightning models output (batch, gridsize**2, N, N) in channel-first layout
        - This mock mirrors that contract to exercise the channel-order fix in _reassemble_cdi_image_torch
        - See debug_shape_triage.md (2025-10-19T092448Z) for channel axis triage
        """
        import torch
        import lightning.pytorch as pl

        class MockLightningModule(pl.LightningModule):
            def __init__(self, N=64, gridsize=2):
                super().__init__()
                self.N = N
                self.gridsize = gridsize

            def forward(self, X):
                """
                Return deterministic complex tensor for testing.
                X shape: (batch, gridsize**2, N, N) channel-last input
                Output: (batch, gridsize**2, N, N) CHANNEL-FIRST complex to match real models

                This mimics the real PyTorch model output layout, forcing the stitching
                path to convert from channel-first to channel-last before TensorFlow reassembly.
                """
                batch_size = X.shape[0]
                C = self.gridsize ** 2
                # Create deterministic complex output (amplitude=1, phase=0.5 rad)
                # Shape: (batch, C, N, N) CHANNEL-FIRST
                real = torch.ones(batch_size, C, self.N, self.N, dtype=torch.float32)
                imag = torch.ones(batch_size, C, self.N, self.N, dtype=torch.float32) * 0.5
                return torch.complex(real, imag)

            def forward_predict(self, X, *args, **kwargs):
                return self.forward(X)

        return MockLightningModule(
            N=64,
            gridsize=minimal_training_config.model.gridsize
        )

    @pytest.fixture
    def stitch_train_results(self, mock_lightning_module):
        """
        Create train_results dict fixture for stitching tests.

        Provides minimal structure required by _reassemble_cdi_image_torch:
        - models['diffraction_to_obj']: Trained (mocked) Lightning module (Phase C4.D3 update)
        - models['autoencoder']: Sentinel for dual-model bundle compliance
        - history: Training loss (placeholder)

        Note: trainer key omitted from models dict to avoid model_manager validation errors
        when run_cdi_example_torch attempts persistence (trainer=None is invalid for save).
        """
        return {
            "models": {
                "diffraction_to_obj": mock_lightning_module,  # Phase C4.D3: new key structure
                "autoencoder": {'_sentinel': 'autoencoder'},  # Dual-model bundle requirement
            },
            "history": {"train_loss": [0.1, 0.05]},
        }

    def test_reassemble_cdi_image_torch_guard_without_train_results(
        self,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        REGRESSION TEST: _reassemble_cdi_image_torch raises NotImplementedError when train_results=None.

        Requirement: Preserve backward-compatible guard for tests expecting NotImplementedError.

        Expected behavior:
        - Function signature exists in ptycho_torch/workflows/components.py
        - Calling without train_results parameter (or train_results=None) raises NotImplementedError
        - Error message indicates "not yet fully implemented without train_results"

        Implementation note (from ptycho_torch/workflows/components.py:659-666):
        - train_results parameter is required for Lightning inference
        - Guard exists at lines 659-666 to maintain RED test expectations
        - This test documents that omitting train_results is intentionally unsupported

        Test mechanism:
        - Call _reassemble_cdi_image_torch WITHOUT train_results parameter
        - Assert NotImplementedError with descriptive match pattern
        """
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import update_legacy_dict
        from ptycho import params

        # Bridge config to params.cfg (CONFIG-001 gate)
        update_legacy_dict(params.cfg, minimal_training_config)

        # GUARD ASSERTION: expect NotImplementedError when train_results omitted
        with pytest.raises(NotImplementedError, match="not yet fully implemented without train_results"):
            torch_components._reassemble_cdi_image_torch(
                test_data=dummy_raw_data,
                config=minimal_training_config,
                flip_x=False,
                flip_y=False,
                transpose=False,
                M=128,  # Canvas size for reassembly
                train_results=None  # CRITICAL: explicit None to trigger guard
            )

    @pytest.mark.parametrize("flip_x,flip_y,transpose", [
        (False, False, False),  # No transforms
        (True, False, False),   # Flip X only
        (False, True, False),   # Flip Y only
        (False, False, True),   # Transpose only
        (True, True, True),     # All transforms
    ])
    def test_reassemble_cdi_image_torch_flip_transpose_contract(
        self,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data,
        stitch_train_results,
        flip_x,
        flip_y,
        transpose
    ):
        """
        GREEN TEST: _reassemble_cdi_image_torch honors flip/transpose parameters.

        Requirement: TensorFlow parity per specs/ptychodus_api_spec.md §4.5.
        TensorFlow baseline (ptycho/workflows/components.py:582-666) applies coordinate
        transforms via flip_x, flip_y, transpose parameters before reassembly.

        Expected behavior (all parameter combinations):
        - Function accepts flip_x, flip_y, transpose parameters
        - Returns (recon_amp, recon_phase, results) tuple
        - Amplitude/phase are 2D numpy arrays with finite values
        - Output shape invariant under flip/transpose (same canvas size)
        - global_offsets in results dict reflect coordinate transforms

        Test mechanism:
        - Parametrize over 5 representative flag combinations
        - Supply train_results with mock Lightning module
        - Assert successful execution (no exceptions)
        - Validate output structure and finiteness

        Rationale for parametrization:
        - Documents TF parity requirement explicitly in test corpus
        - Ensures implementation handles all transform combinations
        - Provides clear failure message surfacing which transform broke
        """
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import update_legacy_dict
        from ptycho import params

        # Bridge config (CONFIG-001)
        update_legacy_dict(params.cfg, minimal_training_config)

        # GREEN PHASE VALIDATION: expect successful stitching with all transform combos
        recon_amp, recon_phase, results = torch_components._reassemble_cdi_image_torch(
            test_data=dummy_raw_data,
            config=minimal_training_config,
            flip_x=flip_x,
            flip_y=flip_y,
            transpose=transpose,
            M=128,
            train_results=stitch_train_results  # CRITICAL: provide trained model
        )

        # Validate return structure
        assert isinstance(recon_amp, np.ndarray), "recon_amp must be numpy array"
        assert isinstance(recon_phase, np.ndarray), "recon_phase must be numpy array"
        assert recon_amp.ndim == 2, "recon_amp must be 2D (stitched image)"
        assert recon_phase.ndim == 2, "recon_phase must be 2D (stitched image)"
        assert isinstance(results, dict), "results must be dict"

        # Validate finite outputs (no NaN/Inf from stitching)
        assert np.all(np.isfinite(recon_amp)), "recon_amp must have finite values"
        assert np.all(np.isfinite(recon_phase)), "recon_phase must have finite values"

        # Validate results payload
        assert "global_offsets" in results, "results must contain global_offsets"
        assert "obj_tensor_full" in results, "results must contain obj_tensor_full"

        # Validate channel-last layout for obj_tensor_full (debug_shape_triage.md requirement)
        obj_tensor_full = results["obj_tensor_full"]
        assert obj_tensor_full.ndim == 4, "obj_tensor_full must be 4D (n, H, W, C)"
        # After channel reduction for TensorFlow reassembly, should have single channel
        assert obj_tensor_full.shape[-1] == 1, \
            "obj_tensor_full must be channel-last with shape[-1]=1 after reduction for TF reassembly"

    def test_run_cdi_example_torch_do_stitching_delegates_to_reassemble(
        self,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data,
        stitch_train_results,
        monkeypatch
    ):
        """
        GREEN TEST: run_cdi_example_torch(do_stitching=True) delegates to stitching path.

        Requirement: Phase D2.C workflow integration — ensure orchestration calls reassembly.

        TensorFlow baseline (ptycho/workflows/components.py:676-732):
        - run_cdi_example(..., do_stitching=True) invokes reassemble_cdi_image
        - Stitching runs AFTER training completes
        - Returns (recon_amp, recon_phase, results) when stitching enabled
        - Returns (None, None, results) when do_stitching=False

        Expected behavior:
        - Runs training (mocked), then calls _reassemble_cdi_image_torch with test_data
        - Stitching results populate amplitude/phase return values
        - Returns (recon_amp, recon_phase, results) when do_stitching=True

        Test mechanism:
        - Monkeypatch training path to return stitch_train_results fixture (avoid GPU)
        - Call run_cdi_example_torch with do_stitching=True
        - Assert amplitude/phase are returned (not None)
        - Validate outputs are numpy arrays

        Validation coverage:
        - Confirms orchestration wiring exists
        - Ensures stitching path is reachable from public API
        - Documents return value contract for downstream consumers (e.g., ptychodus)
        """
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import update_legacy_dict
        from ptycho import params

        # Bridge config (CONFIG-001)
        update_legacy_dict(params.cfg, minimal_training_config)

        # Monkeypatch _train_with_lightning to return mock results with Lightning module
        def mock_train_with_lightning(train_container, test_container, config, execution_config=None):
            """Stub that returns train_results with mock Lightning module."""
            # Return the stitch_train_results fixture enriched with containers
            results = stitch_train_results.copy()
            results["containers"] = {"train": train_container, "test": test_container}
            return results

        monkeypatch.setattr(
            torch_components,
            "_train_with_lightning",
            mock_train_with_lightning
        )

        # GREEN PHASE VALIDATION: expect successful stitching
        recon_amp, recon_phase, results = torch_components.run_cdi_example_torch(
            train_data=dummy_raw_data,
            test_data=dummy_raw_data,  # Use same data for test (deterministic)
            config=minimal_training_config,
            flip_x=False,
            flip_y=False,
            transpose=False,
            M=128,
            do_stitching=True,  # CRITICAL: enable stitching path
        )

        # Validate stitching outputs were populated
        assert recon_amp is not None, "recon_amp must not be None when do_stitching=True"
        assert recon_phase is not None, "recon_phase must not be None when do_stitching=True"
        assert isinstance(recon_amp, np.ndarray), "recon_amp must be numpy array"
        assert isinstance(recon_phase, np.ndarray), "recon_phase must be numpy array"
        assert recon_amp.ndim == 2, "recon_amp must be 2D stitched image"
        assert recon_phase.ndim == 2, "recon_phase must be 2D stitched image"

    def test_reassemble_cdi_image_torch_return_contract(
        self,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data,
        stitch_train_results
    ):
        """
        GREEN TEST: _reassemble_cdi_image_torch returns (recon_amp, recon_phase, results).

        Requirement: API parity with TensorFlow reassemble_cdi_image per spec §4.5.

        TensorFlow baseline signature (ptycho/workflows/components.py:582):
        ```python
        def reassemble_cdi_image(test_data, config, flip_x=False, flip_y=False,
                                  transpose=False, coord_scale=1, M=None):
            ...
            return recon_amp, recon_phase, results
        ```

        Implemented PyTorch signature (ptycho_torch/workflows/components.py:606-614):
        ```python
        def _reassemble_cdi_image_torch(test_data, config, flip_x=False, flip_y=False,
                                         transpose=False, M=None, train_results=None):
            ...
            return recon_amp, recon_phase, results
        ```

        Return value contract:
        - recon_amp: 2D numpy array (float), stitched amplitude image
        - recon_phase: 2D numpy array (float), stitched phase image
        - results: dict containing {"obj_tensor_full", "global_offsets", "containers"}

        Test mechanism:
        - Supply train_results with mock Lightning module
        - Validate return tuple structure and types
        - Ensure all required keys present in results dict

        Rationale:
        - Ensures signature compatibility with TensorFlow baseline
        - Prevents silent breaking changes to return format
        - Codifies downstream consumer expectations (ptychodus integration)
        """
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import update_legacy_dict
        from ptycho import params

        # Bridge config (CONFIG-001)
        update_legacy_dict(params.cfg, minimal_training_config)

        # GREEN PHASE VALIDATION: validate return contract
        recon_amp, recon_phase, results = torch_components._reassemble_cdi_image_torch(
            test_data=dummy_raw_data,
            config=minimal_training_config,
            flip_x=False,
            flip_y=False,
            transpose=False,
            M=128,
            train_results=stitch_train_results  # CRITICAL: provide trained model
        )

        # Validate return tuple structure
        assert isinstance(recon_amp, np.ndarray), "recon_amp must be numpy array"
        assert isinstance(recon_phase, np.ndarray), "recon_phase must be numpy array"
        assert recon_amp.ndim == 2, "recon_amp must be 2D (stitched image)"
        assert recon_phase.ndim == 2, "recon_phase must be 2D (stitched image)"

        # Validate results dict structure
        assert isinstance(results, dict), "results must be dict"
        assert "obj_tensor_full" in results, "results must contain obj_tensor_full"
        assert "global_offsets" in results, "results must contain global_offsets"
        assert "containers" in results, "results must contain containers"

        # Validate nested structures
        assert isinstance(results["obj_tensor_full"], np.ndarray), "obj_tensor_full must be array"
        assert isinstance(results["global_offsets"], np.ndarray), "global_offsets must be array"
        assert isinstance(results["containers"], dict), "containers must be dict"


class TestReassembleCdiImageTorchFloat32:
    """
    Phase D1d dtype enforcement tests — verify float32 tensor preservation.

    Context:
    - Integration test fails with RuntimeError: Input type (double) and bias type (float)
    - Root cause: Inference dataloader yields float64 tensors to Lightning module
    - Spec requirement: specs/data_contracts.md §1 mandates diffraction arrays be float32
    - TensorFlow parity: TF pipeline maintains float32 throughout inference

    Test Strategy:
    - RED: Assert _build_inference_dataloader preserves float32 from container
    - GREEN: Cast infer_X to torch.float32 before TensorDataset construction
    - Regression: Verify integration test completes without dtype mismatch

    Artifacts:
    - plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/dtype_triage.md
    - plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/pytest_dtype_red.log
    - plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/pytest_dtype_green.log
    """

    @pytest.fixture
    def minimal_training_config(self):
        """Create minimal TrainingConfig fixture for dtype tests."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            batch_size=4,
        )

        return training_config

    @pytest.fixture
    def float32_container_fixture(self):
        """
        Create mock container dict with explicit float32 tensors.

        This fixture emulates the output of _ensure_container after Phase C,
        where X and coords_nominal should be float32 per data contract.
        Uses dict interface for duck-typing compatibility with _build_inference_dataloader.
        """
        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        # Create deterministic float32 tensors matching data contract
        n_samples = 20
        N = 64

        X_full = torch.randn(n_samples, N, N, dtype=torch.float32)
        coords_nominal = torch.randn(n_samples, 2, dtype=torch.float32)
        local_offsets = torch.randn(n_samples, 1, 2, 1, dtype=torch.float64)  # Can be float64
        global_offsets = torch.randn(n_samples, 1, 2, 1, dtype=torch.float64)  # Can be float64

        # Return dict-like container compatible with _get_tensor helper
        container = {
            'X': X_full,
            'coords_nominal': coords_nominal,
            'local_offsets': local_offsets,
            'global_offsets': global_offsets,
        }

        return container

    def test_batches_remain_float32(
        self,
        minimal_training_config,
        float32_container_fixture
    ):
        """
        RED TEST: _build_inference_dataloader must preserve float32 dtype.

        Requirement:
        - specs/data_contracts.md §1: diffraction arrays MUST be float32
        - docs/workflows/pytorch.md §7: PyTorch parity must maintain dtype consistency

        Expected behavior (after GREEN implementation):
        - infer_X from container is torch.float32
        - After _build_inference_dataloader construction, batches remain float32
        - Lightning module receives float32 tensors (avoids float64 bias mismatch)

        Current behavior (RED phase):
        - _build_inference_dataloader uses torch.from_numpy without dtype enforcement
        - If numpy array is float64 (from legacy code), tensor is also float64
        - Batches yielded as float64 trigger RuntimeError in Conv2d layers

        Test mechanism:
        - Supply PtychoDataContainerTorch with float32 X and coords
        - Build inference dataloader via _build_inference_dataloader
        - Iterate loader and assert X_batch.dtype is torch.float32
        - Expect FAILURE because current implementation lacks dtype cast

        Rationale:
        - Codifies DATA-001 float32 contract for PyTorch inference path
        - Prevents regression after dtype enforcement fix
        - Mirrors TensorFlow baseline behavior (maintains float32 throughout)
        """
        from ptycho_torch.workflows.components import _build_inference_dataloader

        # Verify fixture provides float32 container (precondition)
        assert float32_container_fixture['X'].dtype == torch.float32, \
            "Test fixture must provide float32 X tensor"
        assert float32_container_fixture['coords_nominal'].dtype == torch.float32, \
            "Test fixture must provide float32 coords tensor"

        # Build inference dataloader
        infer_loader = _build_inference_dataloader(
            float32_container_fixture,
            minimal_training_config
        )

        # Iterate loader and assert dtype preservation
        batch_count = 0
        for batch in infer_loader:
            X_batch, coords_batch = batch

            # CRITICAL ASSERTION: X_batch must remain float32
            assert X_batch.dtype == torch.float32, \
                f"X_batch dtype is {X_batch.dtype}, expected torch.float32. " \
                f"Inference dataloader must cast tensors to float32 before yielding batches " \
                f"to prevent 'Input type (double) and bias type (float)' runtime errors. " \
                f"See plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/dtype_triage.md"

            # coords can remain float32 (they are already float32 from container)
            assert coords_batch.dtype == torch.float32, \
                f"coords_batch dtype is {coords_batch.dtype}, expected torch.float32"

            batch_count += 1

        # Ensure we actually iterated batches
        assert batch_count > 0, "Dataloader must yield at least one batch"

    def test_dataloader_casts_float64_to_float32(
        self,
        minimal_training_config
    ):
        """
        RED TEST: _build_inference_dataloader must cast float64 numpy arrays to float32.

        Context:
        - Integration test shows `RuntimeError: Input type (double) and bias type (float)`
        - Root cause: numpy arrays may be float64 from legacy code or checkpoint reload
        - When `torch.from_numpy` converts float64 array, tensor remains float64
        - Lightning Conv2d layers expect float32, causing dtype mismatch

        Requirement:
        - specs/data_contracts.md §1: diffraction arrays MUST be float32
        - Inference path must enforce float32 regardless of input dtype

        Test mechanism:
        - Create mock container with float64 numpy arrays (simulates legacy data)
        - Build inference dataloader
        - Assert batches are cast to float32 despite float64 input
        - Expect FAILURE because current implementation lacks dtype enforcement

        Rationale:
        - Catches the actual bug causing integration test failure
        - Validates dtype enforcement handles worst-case (float64 input)
        - Once GREEN, guarantees robustness against legacy data sources
        """
        try:
            import torch
            import numpy as np
        except ImportError:
            pytest.skip("PyTorch not available")

        from ptycho_torch.workflows.components import _build_inference_dataloader

        # Create mock container with float64 arrays (simulates legacy/checkpoint data)
        n_samples = 20
        N = 64

        # CRITICAL: Use float64 to simulate the actual bug
        X_float64 = np.random.randn(n_samples, N, N).astype(np.float64)
        coords_float64 = np.random.randn(n_samples, 2).astype(np.float64)

        container = {
            'X': X_float64,  # Will be converted to torch.float64 by torch.from_numpy
            'coords_nominal': coords_float64,
        }

        # Build inference dataloader
        infer_loader = _build_inference_dataloader(
            container,
            minimal_training_config
        )

        # Iterate loader and assert dtype enforcement
        batch_count = 0
        for batch in infer_loader:
            X_batch, coords_batch = batch

            # CRITICAL ASSERTION: Must cast to float32 despite float64 input
            assert X_batch.dtype == torch.float32, \
                f"X_batch dtype is {X_batch.dtype}, expected torch.float32. " \
                f"Inference dataloader MUST cast float64 arrays to float32 before yielding batches. " \
                f"Current failure: 'Input type (double) and bias type (float)' in Conv2d forward. " \
                f"Fix: Add `infer_X = infer_X.to(torch.float32, copy=False)` before TensorDataset construction. " \
                f"See plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T110500Z/phase_d2_completion/dtype_triage.md"

            assert coords_batch.dtype == torch.float32, \
                f"coords_batch dtype is {coords_batch.dtype}, expected torch.float32"

            batch_count += 1

        # Ensure we actually iterated batches
        assert batch_count > 0, "Dataloader must yield at least one batch"


class TestDecoderLastShapeParity:
    """
    Phase D1e.B1 — Decoder shape alignment regression tests.

    Tests the PyTorch `Decoder_last` spatial dimension parity with TensorFlow baseline
    when `probe_big=True`, ensuring x1 and x2 paths produce compatible shapes for addition.

    Background:
        Integration test fails with "The size of tensor a (572) must match the size of tensor b (1080)"
        at `ptycho_torch/model.py:366` (`outputs = x1 + x2`). Root cause: Path 1 (x1) uses
        padding (540 → 572) while Path 2 (x2) uses 2× upsampling (540 → 1080), creating
        asymmetric spatial dimensions.

    Evidence:
        plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/shape_trace.md
        plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/shape_mismatch_triage.md

    TDD Phase: RED — This test documents the failure before implementing the fix (D1e.B2).

    References:
        - Phase plan: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-19T105248Z/phase_d2_completion/d1e_shape_plan.md
        - TensorFlow baseline: ptycho/model.py:360-410 (Decoder_last with trim_and_pad_output)
        - PyTorch implementation: ptycho_torch/model.py:299-393 (Decoder_last)
    """

    def test_probe_big_shape_alignment(self):
        """
        Test that Decoder_last.forward() returns spatially consistent outputs when probe_big=True.

        Expected Behavior (TensorFlow parity):
            When probe_big=True, the decoder combines two paths (x1 and x2) via element-wise addition.
            Both paths MUST produce tensors with identical spatial dimensions (height, width) to enable
            addition without RuntimeError.

        Implementation Fix (Phase D1e.B2):
            PyTorch decoder center-crops x2 to match x1 spatial dimensions before addition,
            mirroring TensorFlow's trim_and_pad_output logic. This resolves the shape mismatch
            where x1 padding (540 → 572) produced smaller dims than x2 upsampling (540 → 1080).

        Test Strategy:
            Construct representative decoder input with realistic gridsize=1, N=64 dimensions,
            instantiate Decoder_last with probe_big=True, execute forward(), and assert:
            (1) No RuntimeError (successful addition after spatial alignment)
            (2) Output spatial dimensions match x1 path dimensions (height, width padded by N/4)
        """
        import torch
        from ptycho_torch.config_params import ModelConfig, DataConfig
        from ptycho_torch.model import Decoder_last

        # --- 1. Configuration (realistic gridsize=1, N=64 case) ---
        model_config = ModelConfig(
            mode='Unsupervised',
            probe_big=True,  # Enable x2 branch (requires spatial alignment fix)
            n_filters_scale=2,
            decoder_last_c_outer_fraction=0.25
        )
        data_config = DataConfig(
            N=64,
            grid_size=(1, 1)  # gridsize=1
        )

        # --- 2. Construct representative decoder input ---
        # Based on shape trace evidence:
        # Input to Decoder_last: (batch=8, in_channels=64, height=32, width=540)
        # After encoder stack, spatial dims are typically H=N//2, W varies (e.g., 540 for probe_big cases)
        batch_size = 8
        in_channels = 64
        out_channels = 1
        height = 32  # Typical encoder output height (N=64 → 32 after pooling)
        width = 540  # Width observed in shape trace (varies based on probe handling)

        x_input = torch.randn(batch_size, in_channels, height, width)

        # --- 3. Instantiate Decoder_last ---
        decoder = Decoder_last(
            model_config=model_config,
            data_config=data_config,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=torch.sigmoid,
            name='decoder_shape_parity_test',
            batch_norm=False
        )
        decoder.eval()  # Inference mode (no dropout)

        # --- 4. Execute forward and assert successful completion (GREEN phase) ---
        output = decoder.forward(x_input)

        # --- 5. Validate output shape matches x1 path dimensions ---
        # Expected after fix:
        #   x1: (batch, out_channels, height_upsampled, width + 2*(N/4))
        #   x2 center-cropped to match x1 spatial dims
        #   output: same as x1 (after addition)
        #
        # For N=64, height is upsampled via ConvTranspose2d inside conv_up_block (32 → 64),
        # and width is padded (540 → 540 + 2*16 = 572)
        expected_height = height * 2  # Upsampled by ConvTranspose2d in conv_up_block (32 → 64)
        expected_width = width + 2 * (data_config.N // 4)  # Padding applied (540 + 32 = 572)

        assert output.ndim == 4, f"Expected 4D output, got {output.ndim}D: {output.shape}"
        assert output.shape[0] == batch_size, f"Batch size mismatch: {output.shape[0]} != {batch_size}"
        assert output.shape[1] == out_channels, f"Channel count mismatch: {output.shape[1]} != {out_channels}"
        assert output.shape[2] == expected_height, \
            f"Height mismatch: {output.shape[2]} != {expected_height}. " \
            f"Expected: input_height ({height}) * 2 (upsample) = {expected_height}"
        assert output.shape[3] == expected_width, \
            f"Width mismatch: {output.shape[3]} != {expected_width}. " \
            f"Expected: input_width ({width}) + 2*padding ({data_config.N // 4} per side) = {expected_width}"

    def test_probe_big_false_no_mismatch(self):
        """
        Test that Decoder_last.forward() succeeds when probe_big=False (x2 branch disabled).

        Expected Behavior:
            When probe_big=False, only path 1 (x1) executes, returning directly after conv1 + padding.
            No shape mismatch occurs because x2 branch (upsampling) is skipped.

        This test validates that the shape mismatch is specific to the probe_big=True case and
        ensures baseline decoder functionality remains intact.
        """
        import torch
        from ptycho_torch.config_params import ModelConfig, DataConfig
        from ptycho_torch.model import Decoder_last

        # --- 1. Configuration with probe_big=False ---
        model_config = ModelConfig(
            mode='Unsupervised',
            probe_big=False,  # Disable x2 branch
            n_filters_scale=2
        )
        data_config = DataConfig(
            N=64,
            grid_size=(1, 1)
        )

        # --- 2. Construct decoder input (same dims as test_probe_big_shape_alignment) ---
        batch_size = 8
        in_channels = 64
        out_channels = 1
        height = 32
        width = 540

        x_input = torch.randn(batch_size, in_channels, height, width)

        # --- 3. Instantiate Decoder_last ---
        decoder = Decoder_last(
            model_config=model_config,
            data_config=data_config,
            in_channels=in_channels,
            out_channels=out_channels,
            activation=torch.sigmoid,
            name='decoder_probe_big_false',
            batch_norm=False
        )
        decoder.eval()

        # --- 4. Execute forward and expect success (no RuntimeError) ---
        output = decoder.forward(x_input)

        # --- 5. Validate output shape (path 1 only) ---
        # Expected: (batch, out_channels, height_upsampled, width_padded)
        # After conv1: preserves spatial dims → (8, 1, 32, 540)
        # After padding (N/4=16 per side): (8, 1, 32, 572)
        # Note: height dimension may be upsampled by conv transpose in earlier encoder layers (32 → 64)
        # For this test, we assert output is 4D and has expected batch/channel counts
        assert output.ndim == 4, f"Expected 4D output, got {output.ndim}D: {output.shape}"
        assert output.shape[0] == batch_size, f"Batch size mismatch: {output.shape[0]} != {batch_size}"
        assert output.shape[1] == out_channels, f"Channel count mismatch: {output.shape[1]} != {out_channels}"

        # Width after padding: original 540 + 2*(N/4) = 540 + 32 = 572
        expected_width = width + 2 * (data_config.N // 4)
        assert output.shape[3] == expected_width, \
            f"Width mismatch: {output.shape[3]} != {expected_width}. " \
            f"Expected: input_width ({width}) + 2*padding ({data_config.N // 4} per side) = {expected_width}"


class TestTrainWithLightningGreen:
    """
    Phase C3 execution config integration tests — validate PyTorchExecutionConfig wiring.

    These tests validate that _train_with_lightning accepts PyTorchExecutionConfig
    and properly threads execution knobs to Lightning Trainer and dataloaders per
    ADR-003 Phase C3 requirements.

    Design requirements (from phase_c3_workflow_integration/plan.md):
    1. _train_with_lightning MUST accept execution_config: PyTorchExecutionConfig parameter
    2. Trainer kwargs (accelerator, deterministic, gradient_clip_val) MUST reflect execution config
    3. Dataloader kwargs (num_workers, pin_memory) MUST respect execution config
    4. Deterministic flag MUST trigger Lightning deterministic mode

    Test strategy:
    - Use monkeypatch to spy on Trainer.__init__ kwargs
    - Supply PyTorchExecutionConfig with non-default values
    - Assert Trainer received correct overrides from execution config
    - Validate dataloader builder honors num_workers/pin_memory
    """

    @pytest.fixture
    def params_cfg_snapshot(self):
        """Snapshot and restore params.cfg state across tests."""
        from ptycho import params
        original = params.cfg.copy()
        yield params.cfg
        params.cfg.clear()
        params.cfg.update(original)

    @pytest.fixture
    def minimal_training_config(self):
        """Create minimal TrainingConfig fixture for execution config tests."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
        )

        training_config = TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9,
            nepochs=2,
        )

        return training_config

    @pytest.fixture
    def dummy_raw_data(self):
        """Create minimal RawData fixture for testing."""
        from ptycho.raw_data import RawData

        nsamples = 10
        N = 64

        dummy_coords = np.linspace(0, 9, nsamples)
        dummy_diff = np.random.rand(nsamples, N, N).astype(np.float32)
        dummy_probe = np.ones((N, N), dtype=np.complex64)
        dummy_scan_index = np.arange(nsamples, dtype=int)

        return RawData(
            xcoords=dummy_coords,
            ycoords=dummy_coords,
            xcoords_start=dummy_coords,
            ycoords_start=dummy_coords,
            diff3d=dummy_diff,
            probeGuess=dummy_probe,
            scan_index=dummy_scan_index,
        )

    def test_execution_config_overrides_trainer(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        RED TEST: _train_with_lightning MUST pass execution config knobs to Trainer.

        Requirement: ADR-003 Phase C3.A3 — thread trainer kwargs from execution config.

        Expected behavior (after wiring):
        - When execution_config supplied, Trainer receives accelerator/deterministic/gradient_clip_val
        - Values override defaults (e.g., accelerator='gpu', deterministic=False, gradient_clip_val=1.0)
        - When execution_config=None, Trainer uses CPU-safe defaults

        Test mechanism:
        - Spy on Trainer.__init__ to capture kwargs
        - Supply non-default PyTorchExecutionConfig (accelerator='gpu', deterministic=False)
        - Assert Trainer received those exact values
        - Expect FAILURE because _train_with_lightning currently ignores execution_config
        """
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import PyTorchExecutionConfig

        # Spy to track Trainer.__init__ kwargs
        trainer_init_kwargs = {"called": False, "kwargs": None}

        class MockTrainer:
            """Stub Trainer that records __init__ kwargs."""
            def __init__(self, **kwargs):
                trainer_init_kwargs["called"] = True
                trainer_init_kwargs["kwargs"] = kwargs

            def fit(self, module, train_dataloaders=None, val_dataloaders=None, **kwargs):
                pass

        # Monkeypatch Lightning Trainer
        monkeypatch.setattr(
            "lightning.pytorch.Trainer",
            MockTrainer
        )

        # Monkeypatch Lightning module to prevent errors
        class StubLightningModule:
            automatic_optimization = True

            def save_hyperparameters(self):
                pass

        monkeypatch.setattr(
            "ptycho_torch.model.PtychoPINN_Lightning",
            lambda *args, **kwargs: StubLightningModule()
        )

        # Create execution config with non-default values
        exec_config = PyTorchExecutionConfig(
            accelerator='gpu',  # Override default 'cpu'
            deterministic=False,  # Override default True
            gradient_clip_val=1.0,  # Override default None
            num_workers=4,  # Override default 0
        )

        # Create minimal containers
        train_container = {"X": np.ones((10, 64, 64))}

        # Call _train_with_lightning with execution_config
        results = torch_components._train_with_lightning(
            train_container=train_container,
            test_container=None,
            config=minimal_training_config,
            execution_config=exec_config  # CRITICAL: new parameter
        )

        # RED PHASE ASSERTION (will fail until Phase C3.A3 implements)
        assert trainer_init_kwargs["called"], "Trainer must be instantiated"

        kwargs = trainer_init_kwargs["kwargs"]

        # Validate accelerator override
        assert kwargs.get('accelerator') == 'gpu', (
            "_train_with_lightning MUST pass execution_config.accelerator to Trainer. "
            f"Expected 'gpu', got {kwargs.get('accelerator')}. "
            "Phase C3.A3 RED: execution config not yet threaded through Trainer kwargs."
        )

        # Validate deterministic override
        assert kwargs.get('deterministic') == False, (
            "_train_with_lightning MUST pass execution_config.deterministic to Trainer. "
            f"Expected False, got {kwargs.get('deterministic')}. "
            "Phase C3.A3 RED: execution config not yet threaded through Trainer kwargs."
        )

        # Validate gradient clipping override
        assert kwargs.get('gradient_clip_val') == 1.0, (
            "_train_with_lightning MUST pass execution_config.gradient_clip_val to Trainer. "
            f"Expected 1.0, got {kwargs.get('gradient_clip_val')}. "
            "Phase C3.A3 RED: execution config not yet threaded through Trainer kwargs."
        )

    def test_execution_config_controls_determinism(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config,
        dummy_raw_data
    ):
        """
        RED TEST: execution_config.deterministic MUST trigger Lightning deterministic mode.

        Requirement: ADR-003 Phase C3.C2 — validate deterministic behaviour.

        Expected behavior:
        - When deterministic=True (default), Trainer receives deterministic=True
        - This triggers torch.use_deterministic_algorithms(True) and seeds
        - When deterministic=False, Trainer allows non-deterministic ops

        Test mechanism:
        - Supply execution_config with deterministic=True
        - Assert Trainer.__init__ received deterministic=True kwarg
        - Expect FAILURE because current stub doesn't wire deterministic flag
        """
        from ptycho_torch.workflows import components as torch_components
        from ptycho.config.config import PyTorchExecutionConfig

        # Spy to track Trainer.__init__ kwargs
        trainer_init_kwargs = {"called": False, "kwargs": None}

        class MockTrainer:
            def __init__(self, **kwargs):
                trainer_init_kwargs["called"] = True
                trainer_init_kwargs["kwargs"] = kwargs

            def fit(self, module, train_dataloaders=None, val_dataloaders=None, **kwargs):
                pass

        monkeypatch.setattr(
            "lightning.pytorch.Trainer",
            MockTrainer
        )

        # Monkeypatch Lightning module
        class StubLightningModule:
            automatic_optimization = True

            def save_hyperparameters(self):
                pass

        monkeypatch.setattr(
            "ptycho_torch.model.PtychoPINN_Lightning",
            lambda *args, **kwargs: StubLightningModule()
        )

        # Create execution config with deterministic=True (default)
        exec_config = PyTorchExecutionConfig(
            deterministic=True,
            accelerator='cpu'
        )

        # Create minimal containers
        train_container = {"X": np.ones((10, 64, 64))}

        # Call _train_with_lightning
        results = torch_components._train_with_lightning(
            train_container=train_container,
            test_container=None,
            config=minimal_training_config,
            execution_config=exec_config
        )

        # RED PHASE ASSERTION
        assert trainer_init_kwargs["called"], "Trainer must be instantiated"

        kwargs = trainer_init_kwargs["kwargs"]

        # Validate deterministic flag is passed
        assert kwargs.get('deterministic') == True, (
            "_train_with_lightning MUST pass execution_config.deterministic=True to Trainer. "
            f"Expected True, got {kwargs.get('deterministic')}. "
            "This flag triggers Lightning's deterministic mode (seeds + deterministic algorithms). "
            "Phase C3.C2 RED: deterministic wiring not yet implemented."
        )


class TestInferenceExecutionConfig:
    """
    Phase C3.B inference execution config tests.

    Validates that inference helpers respect execution config for dataloaders
    and runtime overrides per ADR-003 Phase C3.B requirements.
    """

    @pytest.fixture
    def params_cfg_snapshot(self):
        """Snapshot and restore params.cfg state across tests."""
        from ptycho import params
        original = params.cfg.copy()
        yield params.cfg
        params.cfg.clear()
        params.cfg.update(original)

    @pytest.fixture
    def minimal_training_config(self):
        """Create minimal TrainingConfig fixture."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(N=64, gridsize=2, model_type='pinn')

        return TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            test_data_file=Path("/tmp/dummy_test.npz"),
            n_groups=10,
            batch_size=4,
            neighbor_count=4,
            nphotons=1e9,
        )

    def test_inference_uses_execution_batch_size(
        self,
        monkeypatch,
        params_cfg_snapshot,
        minimal_training_config
    ):
        """
        RED TEST: _build_inference_dataloader MUST respect execution_config.inference_batch_size.

        Requirement: ADR-003 Phase C3.B2 — support inference batch size override.

        Expected behavior:
        - When execution_config.inference_batch_size is set, dataloader uses that batch size
        - When None, dataloader falls back to config.batch_size
        - This enables CPU-safe inference with smaller batches

        Test mechanism:
        - Create mock container
        - Supply execution_config with inference_batch_size=2 (override default 4)
        - Build inference dataloader
        - Assert dataloader.batch_size == 2
        - Expect FAILURE because _build_inference_dataloader currently ignores execution_config
        """
        from ptycho_torch.workflows.components import _build_inference_dataloader
        from ptycho.config.config import PyTorchExecutionConfig

        try:
            import torch
        except ImportError:
            pytest.skip("PyTorch not available")

        # Create mock container
        container = {
            'X': torch.randn(20, 64, 64, dtype=torch.float32),
            'coords_nominal': torch.randn(20, 2, dtype=torch.float32),
        }

        # Create execution config with inference batch size override
        exec_config = PyTorchExecutionConfig(
            inference_batch_size=2,  # Override config.batch_size (4)
            num_workers=0,
            accelerator='cpu'
        )

        # Build inference dataloader (RED phase: execution_config not yet accepted)
        # Need to update signature to accept execution_config parameter
        try:
            infer_loader = _build_inference_dataloader(
                container,
                minimal_training_config,
                execution_config=exec_config  # CRITICAL: new parameter
            )

            # RED PHASE ASSERTION (will fail until C3.B2 implements)
            assert infer_loader.batch_size == 2, (
                "_build_inference_dataloader MUST respect execution_config.inference_batch_size. "
                f"Expected batch_size=2, got {infer_loader.batch_size}. "
                "Phase C3.B2 RED: inference batch size override not yet implemented."
            )

        except TypeError as e:
            # Expected during RED phase: function doesn't accept execution_config yet
            if 'execution_config' in str(e):
                pytest.fail(
                    "_build_inference_dataloader signature must be updated to accept "
                    "execution_config parameter. Phase C3.B1 RED: parameter not yet added."
                )
            else:
                raise


class TestLightningCheckpointCallbacks:
    """
    Phase EB1.E RED Tests: Verify Lightning checkpoint/early-stop callbacks are configured.

    Requirements (ADR-003 Phase EB1.D):
    - _train_with_lightning MUST instantiate ModelCheckpoint callback when enable_checkpointing=True
    - ModelCheckpoint MUST use execution_config.checkpoint_save_top_k, checkpoint_monitor_metric, checkpoint_mode
    - _train_with_lightning MUST instantiate EarlyStopping callback with execution_config.early_stop_patience
    - Callbacks MUST be passed to Lightning Trainer.fit()

    Expected RED Behavior:
    - Tests FAIL because _train_with_lightning does not instantiate ModelCheckpoint/EarlyStopping
    - Tests FAIL because callback configuration does not respect execution_config fields
    - AttributeError if checkpoint_mode field doesn't exist yet

    References:
    - input.md EB1.E (checkpoint controls RED tests)
    - plans/.../phase_e_execution_knobs/plan.md §EB1.D (Lightning callback integration)
    """

    @pytest.fixture
    def minimal_training_config(self):
        """Minimal TrainingConfig for callback tests."""
        from ptycho.config.config import TrainingConfig, ModelConfig

        model_config = ModelConfig(N=64, gridsize=1, model_type='pinn')
        return TrainingConfig(
            model=model_config,
            train_data_file=Path("/tmp/dummy_train.npz"),
            n_groups=10,
            batch_size=4,
            nepochs=2,
            neighbor_count=1,
            nphotons=1e9,
        )

    def test_model_checkpoint_callback_configured(self, minimal_training_config, monkeypatch, tmp_path):
        """
        Test: _train_with_lightning instantiates ModelCheckpoint with execution_config values.
        """
        from ptycho.config.config import PyTorchExecutionConfig
        from unittest.mock import patch, MagicMock

        try:
            from lightning.pytorch.callbacks import ModelCheckpoint
        except ImportError:
            pytest.skip("Lightning not available")

        # Create dummy NPZ files to satisfy path validation
        import numpy as np
        dummy_data = {
            'diffraction': np.random.rand(10, 64, 64).astype(np.float32),
            'xcoords': np.random.rand(10),
            'ycoords': np.random.rand(10),
            'probeGuess': np.ones((64, 64), dtype=np.complex64),
            'objectGuess': np.ones((128, 128), dtype=np.complex64),
        }
        train_file = tmp_path / "train.npz"
        np.savez(str(train_file), **dummy_data)

        # Update config with valid paths
        minimal_training_config.train_data_file = train_file
        minimal_training_config.test_data_file = None  # No test data for this test
        minimal_training_config.output_dir = tmp_path / "outputs"

        # Create execution config with checkpoint overrides
        exec_config = PyTorchExecutionConfig(
            enable_checkpointing=True,
            checkpoint_save_top_k=3,
            checkpoint_monitor_metric='train_loss',
            checkpoint_mode='max',
            accelerator='cpu',
            deterministic=True,
            num_workers=0,
        )

        # Mock ModelCheckpoint to spy on instantiation
        mock_checkpoint_cls = MagicMock(spec=ModelCheckpoint)
        mock_checkpoint_instance = MagicMock()
        mock_checkpoint_cls.return_value = mock_checkpoint_instance

        # Mock Trainer to avoid actual training
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Mock data containers to avoid actual data loading
        mock_train_container = MagicMock()
        mock_test_container = None  # No validation data

        from ptycho_torch.workflows.components import _train_with_lightning

        # Patch callbacks and Trainer
        with patch('lightning.pytorch.callbacks.ModelCheckpoint', mock_checkpoint_cls), \
             patch('lightning.pytorch.Trainer', mock_trainer_cls):
            try:
                _train_with_lightning(
                    train_container=mock_train_container,
                    test_container=mock_test_container,
                    config=minimal_training_config,
                    execution_config=exec_config,
                )
            except Exception as e:
                # May fail during training; we only care about callback setup
                pass

        # GREEN Phase Assertions:
        # 1. ModelCheckpoint was instantiated
        assert mock_checkpoint_cls.called, \
            "ModelCheckpoint not instantiated (_train_with_lightning does not wire checkpoint callback)"

        # 2. ModelCheckpoint was configured with execution_config values
        call_kwargs = mock_checkpoint_cls.call_args.kwargs
        assert call_kwargs.get('save_top_k') == 3, \
            f"Expected save_top_k=3, got {call_kwargs.get('save_top_k')}"
        assert call_kwargs.get('monitor') == 'train_loss', \
            f"Expected monitor='train_loss', got {call_kwargs.get('monitor')}"
        assert call_kwargs.get('mode') == 'max', \
            f"Expected mode='max', got {call_kwargs.get('mode')}"

        # 3. Callback was passed to Trainer via callbacks list
        trainer_call_kwargs = mock_trainer_cls.call_args.kwargs
        callbacks_list = trainer_call_kwargs.get('callbacks', [])
        assert mock_checkpoint_instance in callbacks_list, \
            "ModelCheckpoint instance not found in Trainer callbacks list"

    def test_early_stopping_callback_configured(self, minimal_training_config, monkeypatch, tmp_path):
        """
        Test: _train_with_lightning instantiates EarlyStopping with execution_config values.
        """
        from ptycho.config.config import PyTorchExecutionConfig
        from unittest.mock import patch, MagicMock

        try:
            from lightning.pytorch.callbacks import EarlyStopping
        except ImportError:
            pytest.skip("Lightning not available")

        # Create dummy NPZ files
        import numpy as np
        dummy_data = {
            'diffraction': np.random.rand(10, 64, 64).astype(np.float32),
            'xcoords': np.random.rand(10),
            'ycoords': np.random.rand(10),
            'probeGuess': np.ones((64, 64), dtype=np.complex64),
            'objectGuess': np.ones((128, 128), dtype=np.complex64),
        }
        train_file = tmp_path / "train.npz"
        test_file = tmp_path / "test.npz"
        np.savez(str(train_file), **dummy_data)
        np.savez(str(test_file), **dummy_data)

        # Update config with valid paths
        minimal_training_config.train_data_file = train_file
        minimal_training_config.test_data_file = test_file  # Validation data for early stopping
        minimal_training_config.output_dir = tmp_path / "outputs"

        # Create execution config with early stopping override
        exec_config = PyTorchExecutionConfig(
            early_stop_patience=5,
            checkpoint_monitor_metric='val_loss',
            checkpoint_mode='min',
            accelerator='cpu',
            deterministic=True,
            num_workers=0,
        )

        # Mock EarlyStopping to spy on instantiation
        mock_early_stop_cls = MagicMock(spec=EarlyStopping)
        mock_early_stop_instance = MagicMock()
        mock_early_stop_cls.return_value = mock_early_stop_instance

        # Mock Trainer to avoid actual training
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Mock data containers
        mock_train_container = MagicMock()
        mock_test_container = MagicMock()  # Validation data present

        from ptycho_torch.workflows.components import _train_with_lightning

        # Patch callbacks and Trainer
        with patch('lightning.pytorch.callbacks.EarlyStopping', mock_early_stop_cls), \
             patch('lightning.pytorch.Trainer', mock_trainer_cls):
            try:
                _train_with_lightning(
                    train_container=mock_train_container,
                    test_container=mock_test_container,
                    config=minimal_training_config,
                    execution_config=exec_config,
                )
            except Exception:
                pass  # May fail during training; we only care about callback setup

        # Assertions:
        # 1. EarlyStopping was instantiated
        assert mock_early_stop_cls.called, \
            "EarlyStopping not instantiated (_train_with_lightning does not wire early stopping callback)"

        # 2. EarlyStopping was configured with execution_config patience
        call_kwargs = mock_early_stop_cls.call_args.kwargs
        assert call_kwargs.get('patience') == 5, \
            f"Expected patience=5, got {call_kwargs.get('patience')}"
        # EB2: Monitor now derives from model.val_loss_name (e.g., 'poisson_val_loss')
        # instead of hardcoded 'val_loss'
        monitor_metric = call_kwargs.get('monitor')
        assert monitor_metric is not None, "EarlyStopping monitor not set"
        assert 'val' in monitor_metric, \
            f"Expected monitor to contain 'val', got '{monitor_metric}'"
        assert call_kwargs.get('mode') == 'min', \
            f"Expected mode='min', got {call_kwargs.get('mode')}"

        # 3. Callback was passed to Trainer via callbacks list
        trainer_call_kwargs = mock_trainer_cls.call_args.kwargs
        callbacks_list = trainer_call_kwargs.get('callbacks', [])
        assert mock_early_stop_instance in callbacks_list, \
            "EarlyStopping instance not found in Trainer callbacks list"

    def test_disable_checkpointing_skips_callbacks(self, minimal_training_config, monkeypatch):
        """
        RED Test: When enable_checkpointing=False, ModelCheckpoint/EarlyStopping are NOT instantiated.

        Expected RED Failure:
        - Callbacks instantiated even when checkpointing disabled
        OR
        - Logic to skip callbacks doesn't exist yet
        """
        from ptycho.config.config import PyTorchExecutionConfig
        from unittest.mock import patch, MagicMock

        try:
            from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        except ImportError:
            pytest.skip("Lightning not available")

        # Create execution config with checkpointing disabled
        exec_config = PyTorchExecutionConfig(
            enable_checkpointing=False,  # DISABLE checkpointing
            accelerator='cpu',
            deterministic=True,
            num_workers=0,
        )

        # Mock callbacks to spy on instantiation
        mock_checkpoint_cls = MagicMock(spec=ModelCheckpoint)
        mock_early_stop_cls = MagicMock(spec=EarlyStopping)

        # Mock Trainer
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Mock data containers
        mock_train_container = MagicMock()
        mock_test_container = MagicMock()

        from ptycho_torch.workflows.components import _train_with_lightning

        # Patch at import site inside the function
        with patch('lightning.pytorch.callbacks.ModelCheckpoint', mock_checkpoint_cls), \
             patch('lightning.pytorch.callbacks.EarlyStopping', mock_early_stop_cls), \
             patch('lightning.pytorch.Trainer', mock_trainer_cls):
            try:
                _train_with_lightning(
                    train_container=mock_train_container,
                    test_container=mock_test_container,
                    config=minimal_training_config,
                    execution_config=exec_config,
                )
            except Exception:
                pass

        # GREEN Phase Assertions:
        # When checkpointing disabled, callbacks should NOT be instantiated
        assert not mock_checkpoint_cls.called, \
            "ModelCheckpoint instantiated despite enable_checkpointing=False"
        assert not mock_early_stop_cls.called, \
            "EarlyStopping instantiated despite enable_checkpointing=False"


class TestLightningExecutionConfig:
    """
    Test Lightning Trainer execution knobs: scheduler, gradient accumulation, dynamic monitor.

    Phase EB2: Validates that Lightning callbacks/trainer respect execution config overrides
    and derive monitor metric names from model.val_loss_name (not hardcoded 'val_loss').
    """

    @pytest.fixture
    def minimal_training_config_with_val(self, tmp_path):
        """
        Minimal TrainingConfig with validation data for dynamic monitor testing.
        """
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Create dummy NPZ data
        dummy_data = {
            'xcoords': np.linspace(-5, 5, 64).astype(np.float64),
            'ycoords': np.linspace(-5, 5, 64).astype(np.float64),
            'diffraction': np.ones((64, 64, 64), dtype=np.float32),
            'probeGuess': np.ones((64, 64), dtype=np.complex64),
            'objectGuess': np.ones((128, 128), dtype=np.complex64),
        }
        train_file = tmp_path / "train.npz"
        test_file = tmp_path / "test.npz"
        np.savez(str(train_file), **dummy_data)
        np.savez(str(test_file), **dummy_data)

        model_config = ModelConfig(
            N=64,
            gridsize=2,
            model_type='pinn',
            amp_activation='silu',
        )

        config = TrainingConfig(
            model=model_config,
            train_data_file=train_file,
            test_data_file=test_file,  # Validation data present
            n_groups=64,
            batch_size=4,
            nepochs=2,
            output_dir=tmp_path / "outputs",
        )
        return config

    def test_trainer_receives_accumulation(self, minimal_training_config_with_val, monkeypatch):
        """
        RED Test: Verify Lightning Trainer receives accumulate_grad_batches from execution config.

        Expected RED Failure:
        - Trainer not receiving accum_steps from execution config
        OR
        - accumulate_grad_batches not passed to Trainer kwargs

        Resolution (GREEN):
        - _train_with_lightning should pass execution_config.accum_steps to Trainer(accumulate_grad_batches=...)
        """
        from ptycho.config.config import PyTorchExecutionConfig
        from unittest.mock import patch, MagicMock

        try:
            import lightning.pytorch as L
        except ImportError:
            pytest.skip("Lightning not available")

        # Create execution config with custom accumulation
        exec_config = PyTorchExecutionConfig(
            accum_steps=4,  # Override default (1)
            accelerator='cpu',
            deterministic=True,
            num_workers=0,
            enable_checkpointing=False,  # Disable callbacks for simpler mocking
        )

        # Mock Trainer to spy on kwargs
        mock_trainer_cls = MagicMock(spec=L.Trainer)
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Mock data containers
        mock_train_container = MagicMock()
        mock_test_container = MagicMock()

        from ptycho_torch.workflows.components import _train_with_lightning

        # Patch Trainer at import site and assert EXEC-ACCUM-001 guard fires before Trainer init
        with patch('lightning.pytorch.Trainer', mock_trainer_cls):
            with pytest.raises(RuntimeError, match="accumulate_grad_batches"):
                _train_with_lightning(
                    train_container=mock_train_container,
                    test_container=mock_test_container,
                    config=minimal_training_config_with_val,
                    execution_config=exec_config,
                )

        assert not mock_trainer_cls.called, "Trainer should not instantiate when accum_steps > 1"

    def test_monitor_uses_val_loss_name(self, minimal_training_config_with_val, monkeypatch):
        """
        RED Test: Verify ModelCheckpoint and EarlyStopping derive monitor from model.val_loss_name.

        Expected RED Failure:
        - Callbacks use hardcoded 'val_loss' instead of model.val_loss_name
        OR
        - Checkpoint filename uses hardcoded 'val_loss' string

        Resolution (GREEN):
        - _train_with_lightning should read model.val_loss_name after instantiation
        - Pass dynamic monitor string to ModelCheckpoint(monitor=...) and EarlyStopping(monitor=...)
        - Use dynamic metric name in checkpoint filename template

        Spec Reference:
        - ptycho_torch/model.py:1048-1086 — val_loss_name derivation logic
        - input.md EB2.B3 guidance — "trainer should watch model.val_loss_name"
        """
        from ptycho.config.config import PyTorchExecutionConfig
        from unittest.mock import patch, MagicMock

        try:
            from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
        except ImportError:
            pytest.skip("Lightning not available")

        # Create execution config with checkpointing enabled
        exec_config = PyTorchExecutionConfig(
            enable_checkpointing=True,
            checkpoint_monitor_metric='val_loss',  # User provides generic name
            checkpoint_mode='min',
            early_stop_patience=10,
            accelerator='cpu',
            deterministic=True,
            num_workers=0,
        )

        # Mock callbacks to spy on instantiation args
        mock_checkpoint_cls = MagicMock(spec=ModelCheckpoint)
        mock_checkpoint_instance = MagicMock()
        mock_checkpoint_cls.return_value = mock_checkpoint_instance

        mock_early_stop_cls = MagicMock(spec=EarlyStopping)
        mock_early_stop_instance = MagicMock()
        mock_early_stop_cls.return_value = mock_early_stop_instance

        # Mock Trainer
        mock_trainer_cls = MagicMock()
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Mock data containers
        mock_train_container = MagicMock()
        mock_test_container = MagicMock()  # Validation data present

        from ptycho_torch.workflows.components import _train_with_lightning

        # Patch at import sites
        with patch('lightning.pytorch.callbacks.ModelCheckpoint', mock_checkpoint_cls), \
             patch('lightning.pytorch.callbacks.EarlyStopping', mock_early_stop_cls), \
             patch('lightning.pytorch.Trainer', mock_trainer_cls):
            try:
                _train_with_lightning(
                    train_container=mock_train_container,
                    test_container=mock_test_container,
                    config=minimal_training_config_with_val,
                    execution_config=exec_config,
                )
            except Exception:
                pass  # May fail during training; we only care about callback setup

        # GREEN Phase Assertions:
        # 1. ModelCheckpoint should use model.val_loss_name for monitor
        assert mock_checkpoint_cls.called, "ModelCheckpoint not instantiated"
        checkpoint_kwargs = mock_checkpoint_cls.call_args.kwargs
        monitor_metric = checkpoint_kwargs.get('monitor')

        # The monitor should match the model's val_loss_name, not the hardcoded 'val_loss'
        # For PINN model_type with validation, val_loss_name = 'poisson_val_Amp_loss' or similar
        # We expect it to NOT be the raw 'val_loss' string
        assert monitor_metric is not None, "ModelCheckpoint monitor not set"
        assert 'poisson_val' in monitor_metric or 'mae_val' in monitor_metric, \
            f"Expected monitor to contain model-specific val_loss_name, got '{monitor_metric}'"

        # 2. EarlyStopping should use same dynamic monitor
        assert mock_early_stop_cls.called, "EarlyStopping not instantiated"
        early_stop_kwargs = mock_early_stop_cls.call_args.kwargs
        early_monitor = early_stop_kwargs.get('monitor')
        assert early_monitor == monitor_metric, \
            f"EarlyStopping monitor '{early_monitor}' doesn't match ModelCheckpoint monitor '{monitor_metric}'"

        # 3. Checkpoint filename template should use dynamic metric name
        filename_template = checkpoint_kwargs.get('filename', '')
        # Expected pattern: 'epoch={epoch:02d}-<val_loss_name>={<val_loss_name>:.4f}'
        # Should NOT contain hardcoded 'val_loss='
        assert monitor_metric.replace('_loss', '') in filename_template, \
            f"Checkpoint filename '{filename_template}' should reference dynamic metric '{monitor_metric}'"

    def test_trainer_receives_logger(self, minimal_training_config_with_val, monkeypatch):
        """
        RED Test: Verify Lightning Trainer receives logger instance from execution config.

        Expected RED Failure:
        - Trainer receives logger=False (hardcoded) instead of configured logger
        OR
        - Logger not instantiated based on execution_config.logger_backend

        Resolution (GREEN):
        - _train_with_lightning should instantiate logger based on execution_config.logger_backend
        - Pass logger instance to Trainer(logger=...)
        - Support 'csv', 'tensorboard', 'mlflow', and None backends

        References:
        - input.md EB3.B1 (workflow logger tests)
        - plans/.../phase_e_execution_knobs/2025-10-23T110500Z/decision/approved.md
        """
        from ptycho.config.config import PyTorchExecutionConfig
        from unittest.mock import patch, MagicMock, call

        try:
            import lightning.pytorch as L
            from lightning.pytorch.loggers import CSVLogger
        except ImportError:
            pytest.skip("Lightning not available")

        # Test Case 1: CSV Logger
        exec_config_csv = PyTorchExecutionConfig(
            logger_backend='csv',
            accelerator='cpu',
            deterministic=True,
            num_workers=0,
            enable_checkpointing=False,
        )

        # Mock CSVLogger
        mock_csv_logger_cls = MagicMock(spec=CSVLogger)
        mock_csv_logger_instance = MagicMock()
        mock_csv_logger_cls.return_value = mock_csv_logger_instance

        # Mock Trainer
        mock_trainer_cls = MagicMock(spec=L.Trainer)
        mock_trainer_instance = MagicMock()
        mock_trainer_cls.return_value = mock_trainer_instance

        # Mock data containers
        mock_train_container = MagicMock()
        mock_test_container = MagicMock()

        from ptycho_torch.workflows.components import _train_with_lightning

        # Patch at import sites
        with patch('lightning.pytorch.loggers.CSVLogger', mock_csv_logger_cls), \
             patch('lightning.pytorch.Trainer', mock_trainer_cls):
            try:
                _train_with_lightning(
                    train_container=mock_train_container,
                    test_container=mock_test_container,
                    config=minimal_training_config_with_val,
                    execution_config=exec_config_csv,
                )
            except Exception:
                pass  # May fail during training; we only care about logger setup

        # GREEN Phase Assertions for CSV logger:
        assert mock_csv_logger_cls.called, "CSVLogger not instantiated for logger_backend='csv'"

        # Trainer should receive the logger instance
        assert mock_trainer_cls.called, "Trainer not instantiated"
        trainer_kwargs = mock_trainer_cls.call_args.kwargs
        assert 'logger' in trainer_kwargs, "logger not passed to Trainer"
        # Verify logger is the CSVLogger instance, not False
        assert trainer_kwargs['logger'] is mock_csv_logger_instance, \
            f"Expected Trainer to receive CSVLogger instance, got {trainer_kwargs['logger']}"

        # Test Case 2: logger_backend=None should pass logger=False
        exec_config_none = PyTorchExecutionConfig(
            logger_backend=None,
            accelerator='cpu',
            deterministic=True,
            num_workers=0,
            enable_checkpointing=False,
        )

        mock_trainer_cls_2 = MagicMock(spec=L.Trainer)
        mock_trainer_instance_2 = MagicMock()
        mock_trainer_cls_2.return_value = mock_trainer_instance_2

        with patch('lightning.pytorch.Trainer', mock_trainer_cls_2):
            try:
                _train_with_lightning(
                    train_container=mock_train_container,
                    test_container=mock_test_container,
                    config=minimal_training_config_with_val,
                    execution_config=exec_config_none,
                )
            except Exception:
                pass

        # GREEN Phase Assertion for None backend:
        assert mock_trainer_cls_2.called, "Trainer not instantiated for logger_backend=None"
        trainer_kwargs_none = mock_trainer_cls_2.call_args.kwargs
        assert 'logger' in trainer_kwargs_none, "logger not passed to Trainer"
        # Verify logger=False when backend is None
        assert trainer_kwargs_none['logger'] is False, \
            f"Expected Trainer to receive logger=False for logger_backend=None, got {trainer_kwargs_none['logger']}"
