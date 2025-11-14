"""
Minimal pytest for patch stats CLI flag plumbing (FIX-PYTORCH-FORWARD-PARITY-001 Phase A).

Tests that --log-patch-stats and --patch-stats-limit are accepted by the training CLI
and forwarded through the configuration stack.

Usage:
    pytest tests/torch/test_patch_stats_cli.py::TestPatchStatsCLI::test_patch_stats_flags_accepted -v

References:
    - input.md (2025-11-16): Brief for Phase A instrumentation
    - Phase A plan: plans/active/FIX-PYTORCH-FORWARD-PARITY-001/implementation.md §A2
"""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestPatchStatsCLI:
    """
    Minimal test proving --log-patch-stats and --patch-stats-limit CLI flags work.

    Strategy:
    - Mock the factory to avoid full training execution
    - Invoke CLI with patch stats flags
    - Assert flags were accepted (no argparse errors) and passed to overrides
    """

    @pytest.fixture
    def minimal_train_args(self, tmp_path):
        """Minimal required training CLI arguments for testing."""
        train_file = tmp_path / "train.npz"
        train_file.touch()  # Create dummy file so validation passes
        return [
            '--train_data_file', str(train_file),
            '--output_dir', str(tmp_path / 'outputs'),
            '--n_images', '64',
            '--max_epochs', '2',
            '--gridsize', '2',
            '--batch_size', '4',
        ]

    def test_patch_stats_flags_accepted(self, minimal_train_args, monkeypatch, tmp_path):
        """
        Test that --log-patch-stats and --patch-stats-limit are accepted and forwarded.

        Expected GREEN behavior (Phase A):
        - CLI accepts both flags without argparse errors
        - Overrides dict contains log_patch_stats=True and patch_stats_limit=2
        - Factory receives these values and workflow can access them
        """
        # Mock the factory to intercept arguments
        mock_factory = MagicMock()

        # Create a minimal mock payload to avoid downstream errors
        mock_payload = MagicMock()
        mock_payload.pt_data_config = MagicMock(N=64, grid_size=(2,2))
        mock_payload.pt_training_config = MagicMock(epochs=2)
        mock_payload.pt_model_config = MagicMock()
        mock_payload.tf_training_config = MagicMock()
        mock_factory.return_value = mock_payload

        # Mock main() to avoid full training execution
        mock_main = MagicMock()

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory), \
             patch('ptycho_torch.train.main', mock_main):

            # CLI invocation with patch stats flags
            test_args = minimal_train_args + [
                '--log-patch-stats',
                '--patch-stats-limit', '2',
            ]

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit as e:
                # CLI may exit with 0 or non-zero; we only care that flags were parsed
                pass

        # Assert factory was called
        assert mock_factory.called, "Factory was not called (CLI may have failed before factory invocation)"

        # Assert overrides dict contains patch stats flags
        call_kwargs = mock_factory.call_args.kwargs
        assert 'overrides' in call_kwargs, "Overrides dict not passed to factory"

        overrides = call_kwargs['overrides']
        assert 'log_patch_stats' in overrides, "log_patch_stats missing from overrides"
        assert overrides['log_patch_stats'] is True, \
            f"Expected log_patch_stats=True, got {overrides['log_patch_stats']}"

        assert 'patch_stats_limit' in overrides, "patch_stats_limit missing from overrides"
        assert overrides['patch_stats_limit'] == 2, \
            f"Expected patch_stats_limit=2, got {overrides['patch_stats_limit']}"

    def test_patch_stats_default_disabled(self, minimal_train_args, monkeypatch, tmp_path):
        """
        Test that patch stats logging is disabled by default.

        Expected behavior:
        - Without --log-patch-stats, log_patch_stats should be False
        - patch_stats_limit should be None when not specified
        """
        mock_factory = MagicMock()
        mock_payload = MagicMock()
        mock_payload.pt_data_config = MagicMock(N=64, grid_size=(2,2))
        mock_payload.pt_training_config = MagicMock(epochs=2)
        mock_payload.pt_model_config = MagicMock()
        mock_payload.tf_training_config = MagicMock()
        mock_factory.return_value = mock_payload
        mock_main = MagicMock()

        with patch('ptycho_torch.config_factory.create_training_payload', mock_factory), \
             patch('ptycho_torch.train.main', mock_main):

            # No patch stats flags
            test_args = minimal_train_args

            from ptycho_torch.train import cli_main
            monkeypatch.setattr('sys.argv', ['train.py'] + test_args)

            try:
                cli_main()
            except SystemExit:
                pass

        assert mock_factory.called
        call_kwargs = mock_factory.call_args.kwargs
        overrides = call_kwargs.get('overrides', {})

        # Defaults should be False and None
        assert overrides.get('log_patch_stats') is False, \
            "Expected log_patch_stats=False by default"
        assert overrides.get('patch_stats_limit') is None, \
            "Expected patch_stats_limit=None by default"
