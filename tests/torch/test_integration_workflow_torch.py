"""
Phase C PyTorch Integration Workflow Tests (TEST-PYTORCH-001)

This module provides end-to-end integration tests for the PyTorch backend,
mirroring the TensorFlow integration test structure in tests/test_integration_workflow.py.

Test Coverage:
1. PyTorch train → save → load → infer workflow using subprocess calls
2. Artifact persistence (Lightning checkpoint, model bundle)
3. Reconstruction output validation
4. CONFIG-001 compliance (params.cfg synchronization)

Implementation Status: Phase C1 (RED) — Pytest modernization with helper stub.
Tests will fail with NotImplementedError until Phase C2 implementation.

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
def cuda_cpu_env(monkeypatch):
    """
    Force CPU-only execution by setting CUDA_VISIBLE_DEVICES="".

    This ensures deterministic, reproducible behavior regardless of GPU availability.
    Per docs/workflows/pytorch.md §§6,8 and TEST-PYTORCH-001 Phase A prerequisites.
    """
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    return os.environ.copy()


@pytest.fixture
def data_file():
    """
    Return path to the canonical dataset used for PyTorch integration testing.

    Dataset: Run1084_recon3_postPC_shrunk_3.npz (35 MB, 1087 scan positions)
    Per TEST-PYTORCH-001 baseline inventory at reports/2025-10-19T115303Z/baseline/inventory.md
    """
    return project_root / "datasets" / "Run1084_recon3_postPC_shrunk_3.npz"


# ============================================================================
# Helper Functions
# ============================================================================

def _run_pytorch_workflow(tmp_path, data_file, cuda_cpu_env):
    """
    Execute PyTorch train→save→load→infer workflow via subprocess calls.

    Parameters:
        tmp_path: pytest tmp_path fixture for output directories
        data_file: Path to NPZ dataset
        cuda_cpu_env: Environment dict with CUDA_VISIBLE_DEVICES=""

    Returns:
        SimpleNamespace with:
            - training_output_dir: Path to training outputs
            - inference_output_dir: Path to inference outputs
            - checkpoint_path: Path to Lightning checkpoint
            - recon_amp_path: Path to amplitude reconstruction PNG
            - recon_phase_path: Path to phase reconstruction PNG

    Raises:
        NotImplementedError: Stub implementation for Phase C1 RED test

    Phase: C1 (RED)
    TODO: Implement in Phase C2 (GREEN) by porting subprocess logic from legacy unittest
    """
    raise NotImplementedError("PyTorch pytest harness not implemented (Phase C1 stub)")


# ============================================================================
# Legacy unittest harness (SKIP during pytest migration)
# ============================================================================

# ============================================================================
# Pytest-Native Integration Tests
# ============================================================================

def test_run_pytorch_train_save_load_infer(tmp_path, data_file, cuda_cpu_env):
    """
    Tests the complete PyTorch train → save → load → infer workflow.

    This validates the PyTorch model persistence layer by simulating a real
    user workflow across separate processes, mirroring the TensorFlow integration test.

    Phase: C1 (RED)
    Expected Behavior (Phase C2 implementation target):
    1. Training subprocess creates Lightning checkpoint
    2. Checkpoint artifact saved to <output_dir>/checkpoints/last.ckpt or wts.h5.zip
    3. Inference subprocess loads checkpoint and generates reconstructions
    4. Output images created in inference output directory

    Current Status: FAILING — _run_pytorch_workflow stub raises NotImplementedError
    """
    # Call helper function which currently raises NotImplementedError
    result = _run_pytorch_workflow(tmp_path, data_file, cuda_cpu_env)

    # Assertions (will execute once helper is implemented in Phase C2)
    assert result.training_output_dir.exists(), "Training output directory not created"
    assert result.checkpoint_path.exists(), f"Checkpoint not found at {result.checkpoint_path}"
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
