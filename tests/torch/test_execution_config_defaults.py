"""
Test PyTorchExecutionConfig accelerator auto-resolution and GPU-first defaults.

This test module validates the auto→cuda resolution logic introduced to make
PyTorchExecutionConfig GPU-first by default per POLICY-001 compliance.

Coverage:
    1. Auto-resolution prefers CUDA when available
    2. Auto-resolution falls back to CPU with POLICY-001 warning when unavailable
    3. Backend selector inherits GPU-first behavior when execution_config=None
    4. Explicit accelerator values (cpu, cuda) bypass auto-resolution

Related:
    - POLICY-001: PyTorch backend defaults to GPU execution
    - ptycho/config/config.py:PyTorchExecutionConfig.__post_init__
    - ptycho_torch/workflows/components.py (execution_config=None call sites)
"""

import pytest
import warnings
from unittest.mock import MagicMock, patch


class TestPyTorchExecutionConfigDefaults:
    """Test suite for PyTorchExecutionConfig auto-resolution behavior."""

    def test_auto_prefers_cuda(self, monkeypatch):
        """
        Verify 'auto' accelerator resolves to 'cuda' when torch.cuda.is_available() == True.

        POLICY-001: PyTorch backend must default to GPU execution when hardware available.
        """
        # Mock torch.cuda.is_available to return True
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(__import__('sys').modules, 'torch', mock_torch)

        # Import after monkeypatch to get mocked torch
        from ptycho.config.config import PyTorchExecutionConfig

        # Create config with default accelerator='auto'
        config = PyTorchExecutionConfig()

        # Assert resolution to cuda
        assert config.accelerator == 'cuda', (
            "Expected accelerator='auto' to resolve to 'cuda' when CUDA available"
        )

    def test_auto_warns_and_falls_back_to_cpu(self, monkeypatch):
        """
        Verify 'auto' accelerator falls back to 'cpu' with POLICY-001 warning when no CUDA.

        POLICY-001: CPU fallback must emit actionable warning about GPU-first policy.
        """
        # Mock torch.cuda.is_available to return False
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(__import__('sys').modules, 'torch', mock_torch)

        # Import after monkeypatch
        from ptycho.config.config import PyTorchExecutionConfig

        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = PyTorchExecutionConfig()

            # Assert CPU fallback
            assert config.accelerator == 'cpu', (
                "Expected accelerator='auto' to fall back to 'cpu' when CUDA unavailable"
            )

            # Assert POLICY-001 warning was emitted
            assert len(w) == 1, f"Expected 1 warning, got {len(w)}"
            assert issubclass(w[0].category, UserWarning), (
                f"Expected UserWarning, got {w[0].category}"
            )
            assert "POLICY-001" in str(w[0].message), (
                f"Expected POLICY-001 in warning message, got: {w[0].message}"
            )
            assert "No CUDA device detected" in str(w[0].message), (
                f"Expected 'No CUDA device detected' in warning, got: {w[0].message}"
            )

    def test_explicit_cpu_bypasses_auto_resolution(self):
        """
        Verify explicit accelerator='cpu' is respected without auto-resolution.

        Users must be able to force CPU execution without triggering auto-detection.
        """
        from ptycho.config.config import PyTorchExecutionConfig

        # Create config with explicit cpu
        config = PyTorchExecutionConfig(accelerator='cpu')

        # Assert CPU is preserved (no auto-resolution)
        assert config.accelerator == 'cpu', (
            "Expected accelerator='cpu' to be preserved without auto-resolution"
        )

    def test_explicit_cuda_bypasses_auto_resolution(self):
        """
        Verify explicit accelerator='cuda' is respected without auto-resolution.

        Users forcing CUDA should not trigger auto-detection logic.
        """
        from ptycho.config.config import PyTorchExecutionConfig

        # Create config with explicit cuda
        config = PyTorchExecutionConfig(accelerator='cuda')

        # Assert CUDA is preserved
        assert config.accelerator == 'cuda', (
            "Expected accelerator='cuda' to be preserved without auto-resolution"
        )

    def test_workflow_auto_instantiates_with_hardware_detection(self):
        """
        Verify PyTorchExecutionConfig default instantiation matches hardware availability.

        Integration test verifying that when external callers (e.g., Ptychodus) omit execution_config,
        the default PyTorchExecutionConfig() instantiation automatically detects hardware and sets
        accelerator='cuda' on CUDA hosts or accelerator='cpu' on CPU-only hosts.

        POLICY-001: External integrations must benefit from GPU-first defaults without explicit config.

        Note: This test uses real torch.cuda.is_available() to verify hardware-aware defaults.
        """
        import torch
        from ptycho.config.config import PyTorchExecutionConfig

        # Test default instantiation (simulating what workflow does when execution_config=None)
        config = PyTorchExecutionConfig()  # Triggers auto-resolution in __post_init__

        # Verify accelerator matches hardware availability
        expected_accelerator = 'cuda' if torch.cuda.is_available() else 'cpu'
        assert config.accelerator == expected_accelerator, (
            f"Expected default PyTorchExecutionConfig() to resolve to '{expected_accelerator}' "
            f"(torch.cuda.is_available()={torch.cuda.is_available()}), "
            f"got '{config.accelerator}'"
        )

        # On CUDA hosts, verify we got GPU-first behavior
        if torch.cuda.is_available():
            assert config.accelerator == 'cuda', (
                "POLICY-001 violation: On CUDA-capable host, default config should use 'cuda'"
            )

    def test_backend_selector_warns_on_cpu_only_hosts(self):
        """
        Verify PyTorch warns about POLICY-001 when auto-resolving to CPU on hosts without CUDA.

        This test verifies warning behavior on CPU-only hosts. On CUDA-capable hosts,
        this test will be skipped since no CPU fallback occurs.

        POLICY-001: CPU fallback must emit actionable warning about GPU-first policy.
        """
        import torch

        # Skip test if CUDA is available (no fallback to test)
        if torch.cuda.is_available():
            pytest.skip("Test requires CPU-only host to verify fallback warning")

        from ptycho.config.config import PyTorchExecutionConfig

        # Capture warnings during execution_config instantiation
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = PyTorchExecutionConfig()  # Should trigger auto-resolution to CPU

            # Verify CPU fallback occurred
            assert config.accelerator == 'cpu', (
                f"Expected accelerator='cpu' on CPU-only host, got '{config.accelerator}'"
            )

            # Verify POLICY-001 warning was emitted
            policy_warnings = [warning for warning in w if "POLICY-001" in str(warning.message)]
            assert len(policy_warnings) >= 1, (
                f"Expected POLICY-001 warning on CPU-only host, got {len(policy_warnings)} warnings"
            )
            assert "No CUDA device detected" in str(policy_warnings[0].message), (
                f"Expected 'No CUDA device detected' in warning, got: {policy_warnings[0].message}"
            )

    def test_backend_selector_inherits_gpu_first_defaults(self, monkeypatch):
        """
        Verify backend_selector inherits GPU-first defaults when torch_execution_config=None.

        This test captures the dispatcher-level behavior where train_cdi_model_torch is
        called without explicit execution config (torch_execution_config=None), ensuring
        the auto-instantiated config resolves to 'cuda' on CUDA-capable hosts.

        POLICY-001: Backend selectors must inherit GPU-first defaults from PyTorchExecutionConfig.

        Implementation notes:
            - Monkeypatches torch.cuda.is_available() to return True
            - Patches train_cdi_model_torch to capture its execution_config argument
            - Verifies auto-instantiated config has accelerator='cuda'
        """
        from unittest.mock import patch

        # Mock torch.cuda.is_available to return True
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        monkeypatch.setitem(__import__('sys').modules, 'torch', mock_torch)

        # Import after monkeypatch
        from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
        from ptycho.workflows.backend_selector import run_cdi_example_with_backend

        # Create minimal training config with pytorch backend
        config = TrainingConfig(model=ModelConfig(N=64), backend='pytorch')

        # Patch run_cdi_example_torch to capture execution_config argument
        with patch('ptycho_torch.workflows.components.run_cdi_example_torch') as mock_run:
            mock_run.return_value = (None, None, {'history': {}, 'train_container': None})

            # Call dispatcher with torch_execution_config=None (typical Ptychodus usage)
            try:
                run_cdi_example_with_backend(
                    train_data=MagicMock(),
                    test_data=None,
                    config=config,
                    torch_execution_config=None  # Force default instantiation
                )
            except Exception:
                # Ignore downstream errors; we only care about execution_config capture
                pass

            # Verify run_cdi_example_torch was called
            assert mock_run.called, (
                "Expected run_cdi_example_torch to be called for backend='pytorch'"
            )

            # Extract execution_config argument from the call
            call_kwargs = mock_run.call_args.kwargs if mock_run.call_args else {}
            execution_config = call_kwargs.get('execution_config')

            # Verify execution_config was auto-instantiated (not None)
            assert execution_config is not None, (
                "Expected backend_selector to auto-instantiate PyTorchExecutionConfig when "
                "torch_execution_config=None, got None"
            )

            # Verify accelerator resolved to 'cuda' (GPU-first policy)
            assert hasattr(execution_config, 'accelerator'), (
                f"Expected execution_config to have 'accelerator' attribute, got: {type(execution_config)}"
            )
            assert execution_config.accelerator == 'cuda', (
                f"Expected auto-instantiated execution_config to resolve to 'cuda' when CUDA available, "
                f"got '{execution_config.accelerator}'"
            )

    def test_backend_selector_cpu_fallback_with_warning(self, monkeypatch):
        """
        Verify backend_selector falls back to CPU with POLICY-001 warning when CUDA unavailable.

        Companion to test_backend_selector_inherits_gpu_first_defaults, verifying that
        when torch_execution_config=None and torch.cuda.is_available() returns False,
        the auto-instantiated config falls back to 'cpu' and emits a POLICY-001 warning.

        POLICY-001: CPU fallback must emit actionable warning about GPU-first policy.

        Implementation notes:
            - Monkeypatches torch.cuda.is_available() to return False
            - Captures warnings during backend_selector execution
            - Verifies execution_config.accelerator='cpu'
            - Verifies POLICY-001 warning text is present
        """
        from unittest.mock import patch

        # Mock torch.cuda.is_available to return False
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        monkeypatch.setitem(__import__('sys').modules, 'torch', mock_torch)

        # Import after monkeypatch
        from ptycho.config.config import TrainingConfig, ModelConfig, PyTorchExecutionConfig
        from ptycho.workflows.backend_selector import run_cdi_example_with_backend

        # Create minimal training config with pytorch backend
        config = TrainingConfig(model=ModelConfig(N=64), backend='pytorch')

        # Capture warnings and patch run_cdi_example_torch
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            with patch('ptycho_torch.workflows.components.run_cdi_example_torch') as mock_run:
                mock_run.return_value = (None, None, {'history': {}, 'train_container': None})

                # Call dispatcher with torch_execution_config=None
                try:
                    run_cdi_example_with_backend(
                        train_data=MagicMock(),
                        test_data=None,
                        config=config,
                        torch_execution_config=None  # Force default instantiation
                    )
                except Exception:
                    # Ignore downstream errors; we only care about warning capture
                    pass

                # Extract execution_config argument
                call_kwargs = mock_run.call_args.kwargs if mock_run.call_args else {}
                execution_config = call_kwargs.get('execution_config')

                # Verify execution_config was auto-instantiated
                assert execution_config is not None, (
                    "Expected backend_selector to auto-instantiate PyTorchExecutionConfig when "
                    "torch_execution_config=None, got None"
                )

                # Verify CPU fallback
                assert execution_config.accelerator == 'cpu', (
                    f"Expected auto-instantiated execution_config to fall back to 'cpu' when CUDA unavailable, "
                    f"got '{execution_config.accelerator}'"
                )

            # Verify POLICY-001 warning was emitted
            policy_warnings = [warning for warning in w if "POLICY-001" in str(warning.message)]
            assert len(policy_warnings) >= 1, (
                f"Expected POLICY-001 warning when falling back to CPU, got {len(policy_warnings)} warnings"
            )
            assert "No CUDA device detected" in str(policy_warnings[0].message), (
                f"Expected 'No CUDA device detected' in warning, got: {policy_warnings[0].message}"
            )


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
