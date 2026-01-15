"""Smoke tests for scripts/pytorch_api_demo.py (PARALLEL-API-INFERENCE)."""
from __future__ import annotations

import inspect
from pathlib import Path

import pytest


class TestPyTorchApiDemo:
    """Smoke tests for the unified API demo script."""

    @pytest.fixture
    def work_dir(self, tmp_path: Path) -> Path:
        """Create temporary work directory."""
        return tmp_path / "api_demo_test"

    def test_demo_module_importable(self):
        """Demo script can be imported without side effects."""
        from scripts import pytorch_api_demo

        assert hasattr(pytorch_api_demo, "run_backend")
        assert hasattr(pytorch_api_demo, "main")

    def test_run_backend_function_signature(self):
        """run_backend function has expected signature."""
        from scripts.pytorch_api_demo import run_backend

        sig = inspect.signature(run_backend)
        params = list(sig.parameters.keys())
        assert "backend" in params
        assert "out_dir" in params

    @pytest.mark.slow
    def test_tensorflow_backend_runs(self, work_dir: Path):
        """TensorFlow backend executes without error (slow)."""
        from scripts.pytorch_api_demo import run_backend

        out_dir = work_dir / "tf"
        run_backend("tensorflow", out_dir)

        # Verify outputs exist
        assert (out_dir / "train_outputs").exists()
        assert (out_dir / "inference_outputs").exists()
        assert (out_dir / "inference_outputs" / "reconstruction.npz").exists()

    @pytest.mark.slow
    def test_pytorch_backend_runs(self, work_dir: Path):
        """PyTorch backend executes without error (slow)."""
        pytest.importorskip("torch", reason="PyTorch not available")
        from scripts.pytorch_api_demo import run_backend

        out_dir = work_dir / "pytorch"
        run_backend("pytorch", out_dir)

        # Verify outputs exist
        assert (out_dir / "train_outputs").exists()
        assert (out_dir / "inference_outputs").exists()
