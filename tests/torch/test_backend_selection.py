"""
Phase E1.B Backend Selection Tests (INTEGRATE-PYTORCH-001)

This module documents the expected backend selection mechanism for PyTorch vs TensorFlow workflows.
These tests define the desired behavior before implementation (TDD red phase).

Test Coverage:
1. Default backend behavior (backward compatibility)
2. PyTorch backend selection via config flag
3. CONFIG-001 compliance (update_legacy_dict before workflow dispatch)
4. Torch unavailability handling
5. API parity between backends

Implementation Status: These are FAILING tests (Phase E1.B red phase). They document the expected
behavior when a backend selection mechanism is added to the configuration system. The tests will
pass once Phase E1.C implementation completes.

References:
- Phase E plan: plans/active/INTEGRATE-PYTORCH-001/phase_e_integration.md
- Callchain analysis: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T173826Z/phase_e_callchain/
- Spec: specs/ptychodus_api_spec.md §4.1-4.6
"""

import sys
from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, call

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


@pytest.fixture
def params_cfg_snapshot():
    """Fixture to save and restore params.cfg state."""
    import ptycho.params as params
    snapshot = dict(params.cfg)
    yield
    params.cfg.clear()
    params.cfg.update(snapshot)


class TestBackendSelection:
    """
    Test suite for backend selection mechanism (TensorFlow vs PyTorch).

    Phase: E1.B (Evidence-only / Red Tests)
    Status: EXPECTED TO FAIL until Phase E1.C implementation

    These tests define the expected behavior when a 'backend' configuration field
    is added to enable Ptychodus to select between TensorFlow and PyTorch workflows.
    """

    # ============================================================================
    # Test 1: Default Backend Behavior (Backward Compatibility)
    # ============================================================================

    def test_defaults_to_tensorflow_backend(self, params_cfg_snapshot):
        """
        Test that system defaults to TensorFlow when backend parameter unspecified.

        Requirement: Backward compatibility - existing code should continue using TensorFlow
        without any config changes.

        Expected behavior:
        - TrainingConfig() without 'backend' parameter → defaults to 'tensorflow'
        - Imports ptycho.workflows.components (not ptycho_torch.workflows.components)
        - Calls TensorFlow training paths

        Phase: E1.B baseline test
        Reference: phase_e_callchain/summary.md §Default behavior
        """
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Create config without backend parameter (should default to 'tensorflow')
        model_config = ModelConfig(N=64, gridsize=1)
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            batch_size=16,
            nepochs=1
        )

        # Assert default backend is 'tensorflow'
        assert hasattr(config, 'backend'), \
            "TrainingConfig should have 'backend' field"
        assert config.backend == 'tensorflow', \
            "Default backend should be 'tensorflow' for backward compatibility"

    # ============================================================================
    # Test 2: PyTorch Backend Selection
    # ============================================================================

    def test_selects_pytorch_backend(self, params_cfg_snapshot):
        """
        Test that backend='pytorch' routes to PyTorch workflow module.

        Requirement: Explicit PyTorch backend selection via config flag.

        Expected behavior:
        - TrainingConfig(backend='pytorch') → backend field set correctly
        - Workflow orchestrator imports ptycho_torch.workflows.components
        - CONFIG-001 gate: update_legacy_dict() called before PyTorch workflow dispatch

        Phase: E1.B PyTorch selection test
        Reference: phase_e_callchain/summary.md §PyTorch selection
        """
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Create config with explicit PyTorch backend
        model_config = ModelConfig(N=64, gridsize=1)
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            batch_size=16,
            nepochs=1,
            backend='pytorch'  # Explicit backend selection
        )

        # Assert backend is set to 'pytorch'
        assert hasattr(config, 'backend'), \
            "TrainingConfig should have 'backend' field"
        assert config.backend == 'pytorch', \
            "Backend should be 'pytorch' when explicitly specified"

    def test_pytorch_backend_calls_update_legacy_dict(self, params_cfg_snapshot):
        """
        Test that PyTorch backend triggers CONFIG-001 gate before workflow dispatch.

        Requirement: CONFIG-001 compliance - update_legacy_dict must be called before
        any PyTorch workflow functions to synchronize params.cfg.

        Expected behavior:
        - When backend='pytorch', dispatcher calls update_legacy_dict(params.cfg, config)
        - Call happens BEFORE importing/invoking ptycho_torch.workflows.components
        - Prevents shape mismatch errors and CONFIG-001 violations

        Phase: E1.C CONFIG-001 compliance test
        Reference: docs/findings.md ID CONFIG-001
        """
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho.workflows.backend_selector import run_cdi_example_with_backend
        from ptycho.raw_data import RawData
        import ptycho.params as params
        import numpy as np

        # Create PyTorch backend config
        model_config = ModelConfig(N=128, gridsize=2)
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            batch_size=16,
            nepochs=1,
            backend='pytorch'
        )

        # Create minimal dummy data
        dummy_coords = np.array([0.0, 1.0, 2.0])
        dummy_diff = np.random.rand(3, 128, 128).astype(np.float32)
        dummy_probe = np.ones((128, 128), dtype=np.complex64)
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

        # Mock PyTorch components to avoid full workflow execution
        with patch('ptycho_torch.workflows.components.run_cdi_example_torch') as mock_torch_run:
            mock_torch_run.return_value = (None, None, {'history': {}})

            # Mock update_legacy_dict to spy on it (patch in backend_selector module)
            with patch('ptycho.workflows.backend_selector.update_legacy_dict') as mock_update:
                # Call dispatcher with PyTorch backend
                run_cdi_example_with_backend(train_data, None, config, do_stitching=False)

                # Assert update_legacy_dict was called with correct arguments
                mock_update.assert_called_once()
                call_args = mock_update.call_args
                # Check positional args: (params.cfg, config)
                assert call_args[0][0] is params.cfg, "First arg should be params.cfg"
                assert call_args[0][1] is config, "Second arg should be config"

    # ============================================================================
    # Test 3: Torch Unavailability Handling
    # ============================================================================

    def test_pytorch_unavailable_raises_error(self, params_cfg_snapshot):
        """
        Test that actionable error raised when PyTorch selected but unavailable.

        Requirement: Fail-fast with clear guidance when torch runtime unavailable.

        Expected behavior:
        - backend='pytorch' + torch unavailable → RuntimeError
        - Error message contains:
          * "PyTorch backend selected"
          * "ptycho_torch unavailable"
          * Installation guidance (e.g., "pip install torch")

        Phase: E1.C error handling test
        Reference: phase_e_callchain/summary.md §Fallback behavior
        """
        from ptycho.config.config import TrainingConfig, ModelConfig
        from ptycho.workflows.backend_selector import run_cdi_example_with_backend
        from ptycho.raw_data import RawData
        import numpy as np

        model_config = ModelConfig(N=64, gridsize=1)
        config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            batch_size=16,
            nepochs=1,
            backend='pytorch'
        )

        # Create minimal dummy data
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

        # Mock torch unavailability by making the import fail in backend_selector
        def mock_import_failure(name, *args, **kwargs):
            if 'ptycho_torch.workflows' in name:
                raise ImportError(f"No module named '{name}'")
            # Use the real import for everything else
            return __import__(name, *args, **kwargs)

        with patch('builtins.__import__', side_effect=mock_import_failure):
            try:
                # Expected to raise RuntimeError with actionable error message
                run_cdi_example_with_backend(train_data, None, config, do_stitching=False)
                pytest.fail("Should raise RuntimeError when ptycho_torch unavailable")
            except RuntimeError as exc:
                # Assert error message contains actionable guidance
                error_msg = str(exc).lower()
                assert 'pytorch backend selected' in error_msg or 'pytorch' in error_msg, \
                    f"Error should mention PyTorch backend selection, got: {exc}"
                assert 'unavailable' in error_msg or 'not installed' in error_msg, \
                    f"Error should mention unavailability, got: {exc}"
                assert 'pip install' in error_msg, \
                    f"Error should include installation guidance, got: {exc}"

    # ============================================================================
    # Test 4: InferenceConfig Backend Selection
    # ============================================================================

    def test_inference_config_supports_backend_selection(self, params_cfg_snapshot):
        """
        Test that InferenceConfig also supports backend parameter.

        Requirement: Both training and inference workflows should support backend selection.

        Expected behavior:
        - InferenceConfig(backend='pytorch') → backend field set correctly
        - Inference workflows route to ptycho_torch.workflows.components.load_inference_bundle_torch()
        - CONFIG-001 gate: params.cfg restored from archive + update_legacy_dict called

        Phase: E1.B inference backend test
        Reference: specs/ptychodus_api_spec.md §4.6 (inference contract)
        """
        from ptycho.config.config import InferenceConfig, ModelConfig

        model_config = ModelConfig(N=64, gridsize=1)
        config = InferenceConfig(
            model=model_config,
            model_path=Path('model_dir'),
            test_data_file=Path('test.npz'),
            backend='pytorch'  # Explicit backend for inference
        )

        # Assert backend is set to 'pytorch'
        assert hasattr(config, 'backend'), \
            "InferenceConfig should have 'backend' field"
        assert config.backend == 'pytorch', \
            "InferenceConfig backend should be 'pytorch' when specified"

    # ============================================================================
    # Test 5: API Parity Between Backends
    # ============================================================================

    def test_backend_selection_preserves_api_parity(self, params_cfg_snapshot):
        """
        Test that both backends accept identical config signatures.

        Requirement: Switching backends should only require changing 'backend' field,
        not any other parameters.

        Expected behavior:
        - Same TrainingConfig works for backend='tensorflow' and backend='pytorch'
        - Both backends accept identical function signatures
        - Return values have same structure (tuple of amp, phase, results_dict)

        Phase: E1.B API parity test
        Reference: pytorch_workflow_comparison.md §Summary Table
        """
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Create identical config for both backends
        model_config = ModelConfig(N=64, gridsize=1)

        # TensorFlow config
        tf_config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            batch_size=16,
            nepochs=1,
            backend='tensorflow'
        )

        # PyTorch config (identical except backend field)
        pt_config = TrainingConfig(
            model=model_config,
            train_data_file=Path('train.npz'),
            batch_size=16,
            nepochs=1,
            backend='pytorch'
        )

        # Assert both configs are valid and only differ in backend field
        assert tf_config.model.N == pt_config.model.N
        assert tf_config.train_data_file == pt_config.train_data_file
        assert tf_config.batch_size == pt_config.batch_size
        assert tf_config.nepochs == pt_config.nepochs
        assert tf_config.backend != pt_config.backend
        assert tf_config.backend == 'tensorflow'
        assert pt_config.backend == 'pytorch'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
