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
        from ptycho.config.config import DataConfig
        config = TrainingConfig(
            model=model_config,
            data=DataConfig(train_data_file=Path('train.npz')),
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
        from ptycho.config.config import DataConfig
        config = TrainingConfig(
            model=model_config,
            data=DataConfig(train_data_file=Path('train.npz')),
            batch_size=16,
            nepochs=1,
            backend='pytorch'  # Explicit backend selection
        )

        # Assert backend is set to 'pytorch'
        assert hasattr(config, 'backend'), \
            "TrainingConfig should have 'backend' field"
        assert config.backend == 'pytorch', \
            "Backend should be 'pytorch' when explicitly specified"

    def test_pytorch_backend_calls_update_legacy_dict(
        self,
        tmp_path,
        params_cfg_snapshot,
    ):
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
        train_path = tmp_path / "train.npz"
        train_path.touch()
        from ptycho.config.config import DataConfig
        config = TrainingConfig(
            model=model_config,
            data=DataConfig(train_data_file=train_path),
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

    def test_pytorch_unavailable_raises_error(
        self,
        tmp_path,
        params_cfg_snapshot,
    ):
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
        train_path = tmp_path / "train.npz"
        train_path.touch()
        from ptycho.config.config import DataConfig
        config = TrainingConfig(
            model=model_config,
            data=DataConfig(train_data_file=train_path),
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
        real_import = __import__

        def mock_import_failure(name, *args, **kwargs):
            if 'ptycho_torch.workflows' in name:
                raise ImportError(f"No module named '{name}'")
            # Use the real import for everything else
            return real_import(name, *args, **kwargs)

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

    def test_runnable_failure_precedes_legacy_bridge(
        self,
        tmp_path,
        monkeypatch,
        params_cfg_snapshot,
    ):
        from ptycho.config import ModelConfig, TrainingConfig
        from ptycho.workflows import backend_selector

        events = []
        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            lambda *_args: events.append("bridge"),
        )

        from ptycho.config.config import DataConfig
        config = TrainingConfig(
            model=ModelConfig(),
            data=DataConfig(train_data_file=tmp_path / "missing.npz"),
            backend="tensorflow",
        )

        with pytest.raises(ValueError, match="train_data_file.*exist"):
            backend_selector.run_cdi_example_with_backend(
                MagicMock(),
                None,
                config,
            )

        assert events == []

    def test_valid_training_bridge_order_precedes_backend_delegation(
        self,
        tmp_path,
        monkeypatch,
        params_cfg_snapshot,
    ):
        from ptycho.config import ModelConfig, TrainingConfig
        from ptycho.workflows import backend_selector
        from ptycho.workflows import components as tf_components

        train_path = tmp_path / "train.npz"
        train_path.touch()
        from ptycho.config.config import DataConfig
        config = TrainingConfig(
            model=ModelConfig(),
            data=DataConfig(train_data_file=train_path),
            backend="tensorflow",
        )
        events = []
        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            lambda *_args: events.append("bridge"),
        )
        monkeypatch.setattr(
            tf_components,
            "run_cdi_example",
            lambda *_args, **_kwargs: (
                events.append("delegate") or (None, None, {})
            ),
        )

        backend_selector.run_cdi_example_with_backend(
            MagicMock(),
            None,
            config,
        )

        assert events == ["bridge", "delegate"]

    def test_inference_resource_mismatch_fails_before_bridge(
        self,
        tmp_path,
        monkeypatch,
        params_cfg_snapshot,
    ):
        from ptycho.config import InferenceConfig, ModelConfig
        from ptycho.workflows import backend_selector

        model_path = tmp_path / "model.zip"
        model_path.touch()
        test_path = tmp_path / "test.npz"
        test_path.touch()
        config = InferenceConfig(
            model=ModelConfig(),
            model_path=model_path,
            test_data_file=test_path,
        )
        events = []
        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            lambda *_args: events.append("bridge"),
        )

        with pytest.raises(ValueError, match="bundle_dir.*model_path"):
            backend_selector.load_inference_bundle_with_backend(
                tmp_path / "other.zip",
                config,
            )

        assert events == []

    @pytest.mark.parametrize("backend", ["tensorflow", "pytorch"])
    def test_inference_unsupported_model_layout_fails_before_bridge(
        self,
        backend,
        tmp_path,
        monkeypatch,
        params_cfg_snapshot,
    ):
        from ptycho.config import InferenceConfig, ModelConfig
        from ptycho.workflows import backend_selector

        model_path = tmp_path / "empty-model-directory"
        model_path.mkdir()
        test_path = tmp_path / "test.npz"
        test_path.touch()
        config = InferenceConfig(
            model=ModelConfig(),
            model_path=model_path,
            test_data_file=test_path,
            backend=backend,
        )
        events = []
        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            lambda *_args: events.append("bridge"),
        )

        with pytest.raises(ValueError, match="wts\\.h5\\.zip"):
            backend_selector.load_inference_bundle_with_backend(
                model_path,
                config,
            )

        assert events == []

    def test_inference_bridge_order_precedes_loader_and_archive_can_restore(
        self,
        tmp_path,
        monkeypatch,
        params_cfg_snapshot,
    ):
        import ptycho.params as params
        from ptycho.config import InferenceConfig, ModelConfig
        from ptycho.workflows import backend_selector
        from ptycho.workflows import components as tf_components

        model_path = tmp_path / "model"
        model_path.mkdir()
        (model_path / "wts.h5.zip").touch()
        test_path = tmp_path / "test.npz"
        test_path.touch()
        config = InferenceConfig(
            model=ModelConfig(N=128),
            model_path=model_path,
            test_data_file=test_path,
        )
        events = []

        def bridge(cfg, request):
            events.append("bridge")
            cfg["N"] = request.model.N

        def loader(path):
            assert path == model_path
            assert params.cfg["N"] == 128
            events.append("loader")
            params.cfg["N"] = 64
            return object(), {"N": 64}

        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            bridge,
        )
        monkeypatch.setattr(tf_components, "load_inference_bundle", loader)

        backend_selector.load_inference_bundle_with_backend(
            model_path,
            config,
        )

        assert events == ["bridge", "loader"]
        assert params.cfg["N"] == 64

    def test_inference_canonical_path_is_shared_by_bridge_and_loader(
        self,
        tmp_path,
        monkeypatch,
        params_cfg_snapshot,
    ):
        from ptycho.config import InferenceConfig, ModelConfig
        from ptycho.workflows import backend_selector
        from ptycho.workflows import components as tf_components

        first_target = tmp_path / "first-model"
        first_target.mkdir()
        (first_target / "wts.h5.zip").touch()
        second_target = tmp_path / "second-model"
        second_target.mkdir()
        (second_target / "wts.h5.zip").touch()
        alias = tmp_path / "model-alias"
        alias.symlink_to(first_target, target_is_directory=True)
        test_path = tmp_path / "test.npz"
        test_path.touch()
        config = InferenceConfig(
            model=ModelConfig(),
            model_path=alias,
            test_data_file=test_path,
        )
        canonical_path = first_target.resolve()
        bridged_records = []
        delegated_paths = []

        def bridge(_cfg, request):
            bridged_records.append(request)
            alias.unlink()
            alias.symlink_to(second_target, target_is_directory=True)

        def loader(path):
            delegated_paths.append(path)
            return object(), {}

        monkeypatch.setattr(
            backend_selector,
            "update_legacy_dict",
            bridge,
        )
        monkeypatch.setattr(tf_components, "load_inference_bundle", loader)

        backend_selector.load_inference_bundle_with_backend(alias, config)

        assert bridged_records[0] is not config
        assert bridged_records[0].model_path == canonical_path
        assert delegated_paths == [canonical_path]

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
        from ptycho.config.config import TrainingConfig, ModelConfig, DataConfig

        # Create identical config for both backends
        model_config = ModelConfig(N=64, gridsize=1)

        # TensorFlow config
        tf_config = TrainingConfig(
            model=model_config,
            data=DataConfig(train_data_file=Path('train.npz')),
            batch_size=16,
            nepochs=1,
            backend='tensorflow'
        )

        # PyTorch config (identical except backend field)
        pt_config = TrainingConfig(
            model=model_config,
            data=DataConfig(train_data_file=Path('train.npz')),
            batch_size=16,
            nepochs=1,
            backend='pytorch'
        )

        # Assert both configs are valid and only differ in backend field
        assert tf_config.model.N == pt_config.model.N
        assert tf_config.data.train_data_file == pt_config.data.train_data_file
        assert tf_config.batch_size == pt_config.batch_size
        assert tf_config.nepochs == pt_config.nepochs
        assert tf_config.backend != pt_config.backend
        assert tf_config.backend == 'tensorflow'
        assert pt_config.backend == 'pytorch'


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
