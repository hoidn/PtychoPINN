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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
