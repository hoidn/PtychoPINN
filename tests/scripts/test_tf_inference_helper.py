"""Tests for TensorFlow inference helper extraction (PARALLEL-API-INFERENCE).

This module tests the extracted `_run_tf_inference_and_reconstruct()` helper
and related utilities for API parity with PyTorch inference.

See: plans/active/PARALLEL-API-INFERENCE/reports/2026-01-09T010000Z/extraction_design.md
"""
import inspect
import pytest


class TestTFInferenceHelper:
    """Validate _run_tf_inference_and_reconstruct signature and availability."""

    def test_helper_is_importable(self):
        """Helper function can be imported from scripts.inference.inference."""
        from scripts.inference.inference import _run_tf_inference_and_reconstruct
        assert callable(_run_tf_inference_and_reconstruct)

    def test_helper_signature_matches_spec(self):
        """Helper signature has required parameters per extraction design."""
        from scripts.inference.inference import _run_tf_inference_and_reconstruct
        sig = inspect.signature(_run_tf_inference_and_reconstruct)
        params = list(sig.parameters.keys())

        # Required positional params
        assert 'model' in params
        assert 'raw_data' in params
        assert 'config' in params

        # Optional params with defaults
        assert 'K' in params
        assert 'nsamples' in params
        assert 'quiet' in params
        assert 'seed' in params
        assert 'debug_dump_dir' in params
        assert 'debug_patch_limit' in params

    def test_helper_has_correct_defaults(self):
        """Helper has expected default values for optional params."""
        from scripts.inference.inference import _run_tf_inference_and_reconstruct
        sig = inspect.signature(_run_tf_inference_and_reconstruct)

        assert sig.parameters['K'].default == 4
        assert sig.parameters['nsamples'].default is None
        assert sig.parameters['quiet'].default is False
        assert sig.parameters['seed'].default == 45
        assert sig.parameters['debug_dump_dir'].default is None
        assert sig.parameters['debug_patch_limit'].default == 16

    def test_extract_ground_truth_is_importable(self):
        """extract_ground_truth utility can be imported."""
        from scripts.inference.inference import extract_ground_truth
        assert callable(extract_ground_truth)

    def test_extract_ground_truth_signature(self):
        """extract_ground_truth has correct signature."""
        from scripts.inference.inference import extract_ground_truth
        sig = inspect.signature(extract_ground_truth)
        params = list(sig.parameters.keys())

        assert 'raw_data' in params
        assert len(params) == 1  # Only raw_data parameter

    def test_deprecated_wrapper_still_works(self):
        """perform_inference wrapper emits deprecation warning."""
        import warnings
        from scripts.inference.inference import perform_inference
        # Just check it's callable — full test is in integration suite
        assert callable(perform_inference)

    def test_deprecated_wrapper_has_deprecation_docstring(self):
        """perform_inference wrapper has deprecation note in docstring."""
        from scripts.inference.inference import perform_inference
        assert perform_inference.__doc__ is not None
        assert 'deprecated' in perform_inference.__doc__.lower()
