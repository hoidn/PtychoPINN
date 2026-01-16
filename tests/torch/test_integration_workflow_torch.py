"""
Phase C PyTorch Integration Workflow Tests (TEST-PYTORCH-001)

This module provides end-to-end integration tests for the PyTorch backend,
mirroring the TensorFlow integration test structure in tests/test_integration_workflow.py.

Test Coverage:
1. PyTorch train → save → load → infer workflow using subprocess calls
2. Artifact persistence (Lightning checkpoint, model bundle)
3. Reconstruction output validation
4. CONFIG-001 compliance (params.cfg synchronization)

Implementation Status: Phase C2 (GREEN) — Pytest modernization complete.
Helper function `_run_pytorch_workflow` now executes train/infer subprocesses.

References:
- Phase C plan: plans/active/TEST-PYTORCH-001/reports/2025-10-19T120415Z/phase_c_modernization/plan.md
- Implementation plan: plans/active/TEST-PYTORCH-001/implementation.md
- TensorFlow baseline: tests/test_integration_workflow.py
- Fixture inventory: plans/active/TEST-PYTORCH-001/reports/2025-10-19T115303Z/baseline/inventory.md
"""

import sys
import subprocess
from pathlib import Path
import os
import pytest

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))


# ============================================================================
# Pytest Fixtures
# ============================================================================

@pytest.fixture
def cuda_gpu_env(monkeypatch):
    """
    Force CUDA execution by pinning CUDA_VISIBLE_DEVICES to a specific GPU.

    Skips the test suite if a CUDA-capable device is unavailable to keep the
    GPU-first regression contract documented in docs/workflows/pytorch.md §§11–12.
    """
    try:
        import torch
    except ImportError:  # pragma: no cover - enforced by POLICY-001
        pytest.skip("PyTorch is required for TEST-PYTORCH-001")

    if not torch.cuda.is_available():
        pytest.skip("CUDA GPU required for TEST-PYTORCH-001 baseline (set CUDA_VISIBLE_DEVICES)")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    return env


@pytest.fixture
def data_file():
    """
    Return path to the minimal fixture for PyTorch integration testing.

    Dataset: minimal_dataset_v1.npz (64 scan positions, deterministic subset)
    Per TEST-PYTORCH-001 Phase B3 plan at reports/2025-10-19T214052Z/phase_b_fixture/plan.md
    Previous: Run1084_recon3_postPC_shrunk_3.npz (35 MB, 1087 scan positions) - Phase B1 baseline
    """
    return project_root / "tests" / "fixtures" / "pytorch_integration" / "minimal_dataset_v1.npz"


# ============================================================================
# Helper Functions
# ============================================================================

def _run_pytorch_workflow(tmp_path, data_file, cuda_gpu_env):
    """
    Execute PyTorch train→save→load→infer workflow via subprocess calls.

    Parameters:
        tmp_path: pytest tmp_path fixture for output directories
        data_file: Path to NPZ dataset
        cuda_gpu_env: Environment dict with CUDA_VISIBLE_DEVICES="0"

    Returns:
        SimpleNamespace with:
            - training_output_dir: Path to training outputs
            - inference_output_dir: Path to inference outputs
            - checkpoint_path: Path to Lightning checkpoint
            - recon_amp_path: Path to amplitude reconstruction PNG
            - recon_phase_path: Path to phase reconstruction PNG

    Raises:
        RuntimeError: If training or inference subprocess fails

    Phase: C2 (GREEN)
    Implementation: Ported from legacy unittest harness with subprocess commands
    """
    from types import SimpleNamespace

    # Define output paths
    training_output_dir = tmp_path / "training_outputs"
    inference_output_dir = tmp_path / "pytorch_output"

    # --- 1. Training Step (PyTorch) ---
    # CLI parameters aligned with Phase B1 scope (fixture n_subset=64, deterministic config)
    # Preserves CONFIG-001 ordering per docs/workflows/pytorch.md §12
    train_command = [
        sys.executable, "-m", "ptycho_torch.train",
        "--train_data_file", str(data_file),
        "--test_data_file", str(data_file),
        "--output_dir", str(training_output_dir),
        "--max_epochs", "2",  # Aligned with Phase B1 runtime budget (<45s)
        "--n_images", "64",   # Matches fixture subset size
        "--gridsize", "1",
        "--batch_size", "4",
        "--accelerator", "cuda",  # Deterministic single-GPU execution per cuda_gpu_env fixture
        "--disable_mlflow",
    ]

    train_result = subprocess.run(
        train_command,
        capture_output=True,
        text=True,
        env=cuda_gpu_env,
        check=False
    )

    if train_result.returncode != 0:
        raise RuntimeError(
            f"PyTorch training failed with return code {train_result.returncode}\n"
            f"STDOUT:\n{train_result.stdout}\n"
            f"STDERR:\n{train_result.stderr}"
        )

    # Find checkpoint path (Lightning default location)
    checkpoint_path = training_output_dir / "checkpoints" / "last.ckpt"

    # --- 2. Inference Step (PyTorch) ---
    # Inference parameters aligned with Phase B1 scope (subset inference on minimal fixture)
    inference_command = [
        sys.executable, "-m", "ptycho_torch.inference",
        "--model_path", str(training_output_dir),
        "--test_data", str(data_file),
        "--output_dir", str(inference_output_dir),
        "--n_images", "32",  # Half of fixture size for faster inference validation
        "--accelerator", "cuda",
    ]

    infer_result = subprocess.run(
        inference_command,
        capture_output=True,
        text=True,
        env=cuda_gpu_env,
        check=False
    )

    if infer_result.returncode != 0:
        raise RuntimeError(
            f"PyTorch inference failed with return code {infer_result.returncode}\n"
            f"STDOUT:\n{infer_result.stdout}\n"
            f"STDERR:\n{infer_result.stderr}"
        )

    # Define expected output paths
    recon_amp_path = inference_output_dir / "reconstructed_amplitude.png"
    recon_phase_path = inference_output_dir / "reconstructed_phase.png"

    return SimpleNamespace(
        training_output_dir=training_output_dir,
        inference_output_dir=inference_output_dir,
        checkpoint_path=checkpoint_path,
        recon_amp_path=recon_amp_path,
        recon_phase_path=recon_phase_path
    )


# ============================================================================
# Legacy unittest harness (SKIP during pytest migration)
# ============================================================================

# ============================================================================
# Pytest-Native Integration Tests
# ============================================================================

def test_bundle_loader_returns_modules(tmp_path, data_file, cuda_gpu_env):
    """
    Regression test: bundle loader must return Lightning modules, not dicts.

    Phase C4.D.B4: This test validates that load_inference_bundle_torch returns
    nn.Module instances that support .eval(), not sentinel dicts. The bug was
    discovered during integration testing when inference.py raised AttributeError
    on models_dict['diffraction_to_obj'].eval().

    Test Strategy (TDD RED phase):
    1. Train a minimal model and save bundle
    2. Load bundle via load_inference_bundle_torch
    3. Assert both models in dict are torch.nn.Module instances
    4. Assert models support .eval() without AttributeError

    Expected Failure: Currently returns sentinel dict for autoencoder.
    """
    import torch.nn as nn
    from ptycho_torch.workflows.components import load_inference_bundle_torch

    # Setup: Train and save a bundle (reuse training command from integration test)
    training_output_dir = tmp_path / "training_outputs"
    train_command = [
        sys.executable, "-m", "ptycho_torch.train",
        "--train_data_file", str(data_file),
        "--test_data_file", str(data_file),
        "--output_dir", str(training_output_dir),
        "--max_epochs", "1",  # Minimal training for faster test
        "--n_images", "32",
        "--gridsize", "1",
        "--batch_size", "4",
        "--accelerator", "cuda",
        "--disable_mlflow",
    ]

    train_result = subprocess.run(
        train_command, capture_output=True, text=True, env=cuda_gpu_env, check=False
    )

    if train_result.returncode != 0:
        pytest.fail(f"Training failed:\n{train_result.stderr}")

    # Test: Load bundle and verify module types
    models_dict, params_dict = load_inference_bundle_torch(
        bundle_dir=training_output_dir,
        model_name='diffraction_to_obj'
    )

    # Assertions: Both models MUST be nn.Module instances
    assert 'diffraction_to_obj' in models_dict, "Bundle missing diffraction_to_obj model"
    assert 'autoencoder' in models_dict, "Bundle missing autoencoder model"

    diffraction_model = models_dict['diffraction_to_obj']
    autoencoder_model = models_dict['autoencoder']

    # CRITICAL: Models must be nn.Module, not dicts
    assert isinstance(diffraction_model, nn.Module), \
        f"diffraction_to_obj is {type(diffraction_model)}, expected nn.Module"
    assert isinstance(autoencoder_model, nn.Module), \
        f"autoencoder is {type(autoencoder_model)}, expected nn.Module"

    # CRITICAL: Models must support .eval() (AttributeError was the bug symptom)
    try:
        diffraction_model.eval()
        autoencoder_model.eval()
    except AttributeError as e:
        pytest.fail(f"Model does not support .eval(): {e}")


def test_run_pytorch_train_save_load_infer(tmp_path, data_file, cuda_gpu_env):
    """
    Tests the complete PyTorch train → save → load → infer workflow.

    This validates the PyTorch model persistence layer by simulating a real
    user workflow across separate processes, mirroring the TensorFlow integration test.

    Phase: C2 (GREEN)
    Behavior:
    1. Training subprocess creates Lightning checkpoint at checkpoints/last.ckpt
    2. Inference subprocess loads checkpoint and generates reconstructions
    3. Output images created in inference output directory
    4. Assertions verify artifact existence and non-empty file sizes

    Implementation: _run_pytorch_workflow executes train/infer via subprocess
    """
    # Execute complete workflow via subprocess helper (Phase C2 implementation)
    result = _run_pytorch_workflow(tmp_path, data_file, cuda_gpu_env)

    # Assertions
    assert result.training_output_dir.exists(), "Training output directory not created"

    # Verify wts.h5.zip bundle exists (used by inference, not Lightning checkpoint)
    bundle_path = result.training_output_dir / "wts.h5.zip"
    assert bundle_path.exists(), f"Model bundle not found at {bundle_path}"
    assert bundle_path.stat().st_size > 100000, "Bundle too small or corrupted"

    # Verify inference outputs
    assert result.recon_amp_path.exists(), "Amplitude reconstruction image not created"
    assert result.recon_phase_path.exists(), "Phase reconstruction image not created"

    # Verify non-empty outputs
    assert result.recon_amp_path.stat().st_size > 1000, "Amplitude image too small or empty"
    assert result.recon_phase_path.stat().st_size > 1000, "Phase image too small or empty"


# ============================================================================
# Legacy unittest harness (SKIP during pytest migration)
# ============================================================================

@pytest.mark.skip(reason="Migrated to pytest-native test_run_pytorch_train_save_load_infer")
class TestPyTorchIntegrationWorkflow:
    """
    Legacy unittest harness for PyTorch integration workflow.

    Status: SKIPPED — replaced by pytest-native test_run_pytorch_train_save_load_infer
    Phase: Migrated during TEST-PYTORCH-001 Phase C1

    This class is retained temporarily for reference during pytest migration.
    Will be removed after Phase C3 validation confirms pytest version is stable.
    """

    def test_pytorch_train_save_load_infer_cycle_legacy(self):
        """Legacy test retained for reference; replaced by pytest-native version."""
        pass

    def test_pytorch_tf_output_parity(self):
        """Parity comparison test deferred to Phase E2.D."""
        pass
