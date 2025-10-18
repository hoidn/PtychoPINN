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
        def mock_train_cdi_model_torch(train_data, test_data, config):
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

        def mock_lightning_orchestrator(train_container, test_container, config):
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

        def mock_train_cdi_model_torch(train_data, test_data, config):
            """Spy that records train_cdi_model_torch invocation."""
            train_cdi_model_torch_called["called"] = True
            train_cdi_model_torch_called["args"] = (train_data, test_data, config)
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
        train_data_arg, test_data_arg, config_arg = train_cdi_model_torch_called["args"]
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
        def mock_train_cdi_model_torch(train_data, test_data, config):
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
            lightning_init_called["called"] = True
            lightning_init_called["args"] = (model_config, data_config, training_config, inference_config)

            # Return minimal stub module with required Lightning API
            class StubLightningModule:
                def save_hyperparameters(self):
                    pass

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
