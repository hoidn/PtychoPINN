"""
Torch-optional parity tests for RawDataTorch and PtychoDataContainerTorch adapters.

This module is whitelisted in conftest.py to run without PyTorch installed,
validating that adapters correctly delegate to TensorFlow RawData when torch unavailable.

Test source citations (from test_blueprint.md and data_contract.md):
- RawData API: ptycho/raw_data.py:365-486
- PtychoDataContainer API: ptycho/loader.py:93-138
- Data contract: specs/data_contracts.md:7-70

Phase C.B2/C.B3 Goals (TDD Red Phase):
- Document expected RawDataTorch wrapper behavior
- Document expected PtychoDataContainerTorch API surface
- Capture baseline failure logs for implementation guidance
"""

import pytest
import numpy as np
from pathlib import Path

# Torch-optional import guard (per test_blueprint.md §1.C)
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


@pytest.fixture
def params_cfg_snapshot():
    """
    Save/restore params.cfg state (CRITICAL per CONFIG-001 finding).

    Source: tests/torch/test_config_bridge.py:151-160
    Why Critical: RawData.generate_grouped_data() reads params.cfg['gridsize'].
    Failure to restore causes test pollution.
    """
    import ptycho.params as params
    snapshot = dict(params.cfg)
    yield
    params.cfg.clear()
    params.cfg.update(snapshot)


@pytest.fixture
def minimal_raw_data(params_cfg_snapshot):
    """
    Create synthetic RawData for testing (deterministic, no I/O).

    Source pattern: tests/test_coordinate_grouping.py
    ROI parameters from data_contract.md §7:
    - n_points=100 (minimal for K-NN with K=4)
    - N=64 (grid size)
    - gridsize=2 (standard 2x2 patches)
    """
    from ptycho.config.config import TrainingConfig, ModelConfig, update_legacy_dict
    from ptycho import params as p
    from ptycho.raw_data import RawData

    # 1. Initialize params.cfg (MANDATORY per CLAUDE.md:76-93)
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=2),
        n_groups=64,
        neighbor_count=4,
        nphotons=1e9  # Required per Phase B config bridge validation
    )
    update_legacy_dict(p.cfg, config)

    # 2. Create deterministic synthetic data
    n_points = 100
    x = np.linspace(0, 10, int(np.sqrt(n_points)))
    y = np.linspace(0, 10, int(np.sqrt(n_points)))
    xv, yv = np.meshgrid(x, y)
    xcoords = xv.flatten()[:n_points].astype(np.float64)
    ycoords = yv.flatten()[:n_points].astype(np.float64)

    # Random diffraction patterns (normalized amplitude per data_contracts.md:23-70)
    np.random.seed(42)
    diff3d = np.random.rand(n_points, 64, 64).astype(np.float32) * 0.5  # max < 1.0

    # Simple probe and object (complex64 per data contract)
    probe = np.ones((64, 64), dtype=np.complex64)
    obj = np.ones((128, 128), dtype=np.complex64)
    scan_index = np.arange(n_points, dtype=np.int32)

    # Use factory method to create RawData without separate start coordinates
    return RawData.from_coords_without_pc(
        xcoords, ycoords, diff3d, probe, scan_index, objectGuess=obj
    )


class TestRawDataTorchAdapter:
    """
    Test RawDataTorch wrapper delegates correctly to TensorFlow RawData.

    Expected adapter location (TBD): ptycho_torch/raw_data_bridge.py
    Expected API: RawDataTorch.generate_grouped_data() returns same dict as RawData

    Source contract: data_contract.md §2 (RawData.generate_grouped_data())
    """

    def test_raw_data_torch_matches_tensorflow(self, params_cfg_snapshot, minimal_raw_data):
        """
        RawDataTorch should produce identical grouped data outputs to RawData.

        Expected behavior (from data_contract.md §2):
        - Wrapper delegates to ptycho.raw_data.RawData
        - Returns dict with keys: diffraction, X_full, coords_offsets, coords_relative, nn_indices
        - Shapes: diffraction (nsamples, N, N, gridsize²), coords_offsets (nsamples, 1, 2, 1)
        - Dtypes: diffraction float32, nn_indices int32

        Test source: data_contract.md:110-176
        ROI: N=64, gridsize=2, nsamples=10, K=4
        """
        # Create TensorFlow baseline reference
        tf_grouped = minimal_raw_data.generate_grouped_data(
            N=64, K=4, nsamples=10, gridsize=2
        )

        # Validate baseline conforms to contract (data_contract.md §2)
        assert tf_grouped['diffraction'].shape == (10, 64, 64, 4), \
            "TensorFlow baseline shape mismatch"
        assert tf_grouped['diffraction'].dtype == np.float32, \
            "TensorFlow baseline dtype mismatch"
        assert tf_grouped['nn_indices'].shape == (10, 4), \
            "TensorFlow nn_indices shape mismatch"

        # TODO (Phase C.C1): Implement RawDataTorch adapter
        # Expected implementation:
        #   from ptycho_torch.raw_data_bridge import RawDataTorch
        #   pt_raw = RawDataTorch(xcoords, ycoords, diff3d, probe, obj)
        #   pt_grouped = pt_raw.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=2)
        #
        # Expected assertions (parity):
        #   np.testing.assert_array_equal(tf_grouped['nn_indices'], pt_grouped['nn_indices'])
        #   np.testing.assert_allclose(tf_grouped['diffraction'], pt_grouped['diffraction'])
        #   assert tf_grouped['X_full'].shape == pt_grouped['X_full'].shape

        pytest.fail(
            "RawDataTorch adapter not yet implemented (Phase C.C1). "
            "Expected module: ptycho_torch/raw_data_bridge.py. "
            "Expected delegation: wrapper calls ptycho.raw_data.RawData.generate_grouped_data(). "
            "See plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md §2 "
            "for required output dict structure."
        )


class TestDataContainerParity:
    """
    Test PtychoDataContainerTorch matches TensorFlow PtychoDataContainer API.

    Expected adapter location (TBD): ptycho_torch/data_container.py or loader_bridge.py
    Expected API: Container with attributes X, Y, coords_nominal, probe, etc.

    Source contract: data_contract.md §3 (PtychoDataContainer attributes table)
    """

    def test_data_container_shapes_and_dtypes(self, params_cfg_snapshot, minimal_raw_data):
        """
        PtychoDataContainerTorch must expose same attributes as TensorFlow container.

        Expected attributes (from data_contract.md §3):
        - X: (n_images, N, N, gridsize²) float32 — diffraction patterns
        - Y: (n_images, N, N, gridsize²) complex64 — combined ground truth
        - Y_I: (n_images, N, N, gridsize²) float32 — amplitude patches
        - Y_phi: (n_images, N, N, gridsize²) float32 — phase patches
        - coords_nominal: (n_images, 2) float32 — scan coordinates
        - probe: (N, N) complex64 — probe function
        - nn_indices: (n_images, gridsize²) int32 — nearest neighbor indices
        - global_offsets: (n_images, 1, 2, 1) float32 — coordinate offsets

        Test source: data_contract.md:179-200
        ROI: N=64, gridsize=2, nsamples=10
        """
        # Create TensorFlow baseline container
        from ptycho import loader

        def grouped_data_callback():
            """Callback for loader.load() (per data_contract.md §4)."""
            return minimal_raw_data.generate_grouped_data(
                N=64, K=4, nsamples=10, gridsize=2
            )

        # Probe as TensorFlow tensor (required by loader)
        import tensorflow as tf
        probe_tf = tf.convert_to_tensor(minimal_raw_data.probeGuess, dtype=tf.complex64)

        # Create TensorFlow container baseline
        tf_container = loader.load(
            cb=grouped_data_callback,
            probeGuess=probe_tf,
            which='train',
            create_split=False
        )

        # Validate baseline attributes
        assert hasattr(tf_container, 'X'), "TensorFlow container missing X"
        assert hasattr(tf_container, 'Y'), "TensorFlow container missing Y"
        assert hasattr(tf_container, 'coords_nominal'), "TensorFlow container missing coords_nominal"
        assert tf_container.X.shape == (10, 64, 64, 4), "TensorFlow X shape mismatch"
        assert tf_container.Y.dtype == tf.complex64, "TensorFlow Y dtype mismatch"

        # TODO (Phase C.C2): Implement PtychoDataContainerTorch
        # Expected implementation:
        #   from ptycho_torch.data_container import PtychoDataContainerTorch
        #   pt_container = PtychoDataContainerTorch.from_raw_data(...)
        #
        # Expected assertions (API parity):
        #   assert hasattr(pt_container, 'X')
        #   assert hasattr(pt_container, 'Y')
        #   assert hasattr(pt_container, 'coords_nominal')
        #   assert pt_container.X.shape == tf_container.X.shape
        #   if TORCH_AVAILABLE:
        #       assert isinstance(pt_container.X, torch.Tensor)
        #   else:
        #       assert isinstance(pt_container.X, np.ndarray)

        pytest.fail(
            "PtychoDataContainerTorch not yet implemented (Phase C.C2). "
            "Expected module: ptycho_torch/data_container.py or loader_bridge.py. "
            "Required attributes per data_contract.md §3: "
            "X, Y, Y_I, Y_phi, coords_nominal, probe, nn_indices, global_offsets. "
            "See plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T070200Z/data_contract.md §3 "
            "for complete attribute table with shapes and dtypes."
        )


class TestGroundTruthLoading:
    """
    Test Y patch loading and dtype validation.

    Critical requirement (DATA-001 finding): Y patches MUST be complex64, not float64.
    Historical bug: silent dtype conversion caused major training failure.

    Source: docs/findings.md:DATA-001, specs/data_contracts.md:19
    """

    def test_y_patches_are_complex64(self, params_cfg_snapshot, minimal_raw_data):
        """
        Y patches MUST be complex64 per DATA-001 finding.

        Historical bug: A silent float64 conversion was the source of a major bug.
        Critical validation: Ensure no dtype downcasting or upcasting occurs.

        Test source: specs/data_contracts.md:19, docs/findings.md:DATA-001
        """
        # Baseline: TensorFlow container preserves complex64
        from ptycho import loader
        import tensorflow as tf

        def grouped_data_callback():
            return minimal_raw_data.generate_grouped_data(
                N=64, K=4, nsamples=10, gridsize=2
            )

        probe_tf = tf.convert_to_tensor(minimal_raw_data.probeGuess, dtype=tf.complex64)
        tf_container = loader.load(
            cb=grouped_data_callback,
            probeGuess=probe_tf,
            which='train',
            create_split=False
        )

        # Validate TensorFlow baseline
        assert tf_container.Y.dtype == tf.complex64, \
            "TensorFlow Y dtype violated data contract (CRITICAL)"

        # TODO (Phase C.C2/C.C3): Validate PyTorch container Y dtype
        # Expected assertion:
        #   if TORCH_AVAILABLE:
        #       assert pt_container.Y.dtype == torch.complex64
        #   else:
        #       assert pt_container.Y.dtype == np.complex64

        pytest.fail(
            "PyTorch container Y dtype validation not yet implemented (Phase C.C2/C.C3). "
            "CRITICAL requirement per DATA-001 finding: Y MUST be complex64. "
            "Historical silent float64 conversion caused major training failure. "
            "See docs/findings.md:DATA-001 and specs/data_contracts.md:19."
        )
