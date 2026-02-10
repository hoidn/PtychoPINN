#!/usr/bin/env python3
"""
Tests for ptychi_reconstruct_tike.py CLI interface.
"""

import pytest
from unittest.mock import patch, MagicMock, call
from pathlib import Path
import sys


def test_main_uses_cli_arguments():
    """
    Test that main() honors CLI arguments for input/output paths and parameters.

    This test stubs pty-chi modules and verifies that CLI overrides propagate
    to load_and_convert_tike_data and save_results helpers.
    """
    # Import the script module
    # We need to mock ptychi before importing the script
    mock_ptychi_api = MagicMock()
    mock_ptychi_utils = MagicMock()

    # Mock the ptychi modules
    with patch.dict('sys.modules', {
        'ptychi': MagicMock(),
        'ptychi.api': mock_ptychi_api,
        'ptychi.api.task': MagicMock(),
        'ptychi.utils': mock_ptychi_utils,
    }):
        # Mock the necessary ptychi API components
        mock_ptychi_api.DMOptions = MagicMock
        mock_ptychi_api.LSQMLOptions = MagicMock
        mock_ptychi_api.PIEOptions = MagicMock
        mock_ptychi_api.NoiseModels = MagicMock()
        mock_ptychi_api.Devices = MagicMock()
        mock_ptychi_api.task.PtychographyTask = MagicMock

        mock_ptychi_utils.get_suggested_object_size = MagicMock(return_value=(256, 256))
        mock_ptychi_utils.get_default_complex_dtype = MagicMock()

        # Now import the script using repo-relative path
        import importlib.util
        script_path = Path(__file__).resolve().parents[2] / "scripts" / "reconstruction" / "ptychi_reconstruct_tike.py"
        spec = importlib.util.spec_from_file_location(
            "ptychi_reconstruct_tike",
            str(script_path)
        )
        script_module = importlib.util.module_from_spec(spec)

        # Execute the script's module code to populate functions
        spec.loader.exec_module(script_module)

        # Now mock the helper functions we want to track
        with patch.object(script_module, 'load_and_convert_tike_data') as mock_load, \
             patch.object(script_module, 'configure_reconstruction') as mock_configure, \
             patch.object(script_module, 'run_reconstruction') as mock_run, \
             patch.object(script_module, 'save_results') as mock_save:

            # Setup return values
            mock_load.return_value = {'n_images': 100, 'intensity': MagicMock()}
            mock_configure.return_value = MagicMock()
            mock_run.return_value = MagicMock()
            mock_save.return_value = Path("/fake/output.npz")

            # Now call main with CLI arguments
            test_args = [
                "--input-npz", "test_input.npz",
                "--output-dir", "test_output",
                "--algorithm", "LSQML",
                "--num-epochs", "50",
                "--n-images", "256"
            ]

            result = script_module.main(test_args)

            # Assertions
            assert result == 0, "main() should return 0 on success"

            # Verify load_and_convert_tike_data was called with CLI input
            mock_load.assert_called_once()
            call_args = mock_load.call_args
            assert str(call_args[0][0]) == "test_input.npz", \
                "load_and_convert_tike_data should receive CLI --input-npz path"
            assert call_args[1]['n_images'] == 256, \
                "load_and_convert_tike_data should receive CLI --n-images"

            # Verify configure_reconstruction was called with CLI parameters
            mock_configure.assert_called_once()
            config_args = mock_configure.call_args
            assert config_args[1]['algorithm'] == 'LSQML', \
                "configure_reconstruction should receive CLI --algorithm"
            assert config_args[1]['num_epochs'] == 50, \
                "configure_reconstruction should receive CLI --num-epochs"

            # Verify save_results was called with CLI output directory
            mock_save.assert_called_once()
            save_args = mock_save.call_args
            assert str(save_args[1]['output_dir']) == "test_output", \
                "save_results should receive CLI --output-dir path"


if __name__ == "__main__":
    pytest.main([__file__, "-vv"])
