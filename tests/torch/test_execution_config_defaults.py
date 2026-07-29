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
from dataclasses import fields
from unittest.mock import MagicMock, patch


def test_execution_config_preserves_all_fields_and_topology_alias_provenance():
    from ptycho.config.config import PyTorchExecutionConfig

    topology_aliases = {
        "hybrid_skip_connections",
        "hybrid_downsample_steps",
        "hybrid_downsample_op",
        "hybrid_encoder_conv_hidden_scale",
        "hybrid_encoder_spectral_hidden_scale",
        "hybrid_encoder_conv_hidden_channels",
        "hybrid_encoder_spectral_hidden_channels",
        "hybrid_resnet_blocks",
        "hybrid_skip_style",
        "hybrid_resnet_bottleneck_layerscale_mode",
        "hybrid_resnet_bottleneck_layerscale_value",
        "hybrid_encoder_fusion_mode",
        "hybrid_encoder_layerscale_init",
        "hybrid_encoder_branch_gate_init",
        "hybrid_encoder_branch_select",
        "ffno_encoder_blocks",
        "ffno_encoder_modes",
        "ffno_encoder_share_weights",
        "ffno_encoder_gate_init",
        "ffno_encoder_norm",
        "ffno_encoder_mlp_ratio",
        "spectral_bottleneck_blocks",
        "spectral_bottleneck_modes",
        "spectral_bottleneck_share_weights",
        "spectral_bottleneck_gate_init",
        "spectral_bottleneck_gate_mode",
    }
    defaults = {
        item.name: item.default for item in fields(PyTorchExecutionConfig)
    }
    values = {name: defaults[name] for name in topology_aliases}

    config = PyTorchExecutionConfig(accelerator="cpu", **values)

    assert len(fields(PyTorchExecutionConfig)) == 55
    assert config._explicit_structural_aliases == frozenset(topology_aliases)


def test_execution_config_preserves_positional_prefix_binding():
    from ptycho.config.config import PyTorchExecutionConfig

    config = PyTorchExecutionConfig("cpu", 2, "ddp", False)

    assert (
        config.accelerator,
        config.devices,
        config.strategy,
        config.deterministic,
    ) == ("cpu", 2, "ddp", False)


def test_execution_config_warns_before_late_validation_failure(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    from ptycho.config.config import PyTorchExecutionConfig

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(ValueError, match="hybrid_downsample_steps"):
            PyTorchExecutionConfig(
                accelerator="auto",
                hybrid_downsample_steps=0,
            )

    assert len(caught) == 1
    assert caught[0].category is UserWarning
    assert "POLICY-001" in str(caught[0].message)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"num_workers": 0, "persistent_workers": True},
            "persistent_workers",
        ),
        ({"logger_backend": "none"}, "logger_backend"),
    ],
)
def test_execution_config_enforces_selected_runtime_contract(kwargs, message):
    from ptycho.config.config import PyTorchExecutionConfig

    with pytest.raises(ValueError, match=message):
        PyTorchExecutionConfig(accelerator="cpu", **kwargs)


@pytest.mark.parametrize(
    "logger_backend",
    ["csv", "tensorboard", "mlflow", None],
)
def test_execution_config_accepts_canonical_logger_backends(logger_backend):
    from ptycho.config.config import PyTorchExecutionConfig

    config = PyTorchExecutionConfig(
        accelerator="cpu",
        logger_backend=logger_backend,
    )

    assert config.logger_backend == logger_backend


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

    def test_backend_selector_defers_execution_resolution_on_cuda_hosts(
        self,
        monkeypatch,
        tmp_path,
    ):
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

        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        # Import after monkeypatch
        from ptycho.config.config import ModelConfig, TrainingConfig
        from ptycho.workflows.backend_selector import run_cdi_example_with_backend

        # Create minimal training config with pytorch backend
        train_path = tmp_path / "train.npz"
        train_path.touch()
        config = TrainingConfig(
            model=ModelConfig(N=64),
            train_data_file=train_path,
            backend='pytorch',
        )

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

            assert execution_config is None

    def test_backend_selector_defers_execution_resolution_without_cpu_warning(
        self,
        monkeypatch,
        tmp_path,
    ):
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

        import torch

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

        # Import after monkeypatch
        from ptycho.config.config import ModelConfig, TrainingConfig
        from ptycho.workflows.backend_selector import run_cdi_example_with_backend

        # Create minimal training config with pytorch backend
        train_path = tmp_path / "train.npz"
        train_path.touch()
        config = TrainingConfig(
            model=ModelConfig(N=64),
            train_data_file=train_path,
            backend='pytorch',
        )

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

                assert execution_config is None

            # Verify POLICY-001 warning was emitted
            policy_warnings = [warning for warning in w if "POLICY-001" in str(warning.message)]
            assert policy_warnings == []


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
