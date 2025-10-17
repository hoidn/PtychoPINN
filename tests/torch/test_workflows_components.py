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
        # Phase D2.A: expect NotImplementedError (scaffold only)
        # Phase D2.B/C: expect full execution after training/inference impl
        with pytest.raises(NotImplementedError, match="PyTorch training path not yet implemented"):
            torch_components.run_cdi_example_torch(
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
