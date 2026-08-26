"""
Parity tests for RawData and PtychoDataContainerTorch.

Test source citations (from test_blueprint.md and data_contract.md):
- RawData API: ptycho/raw_data.py:365-486
- PtychoDataContainer API: ptycho/loader.py:93-138
- Data contract: specs/data_contracts.md:7-70

The maintained path groups with RawData and converts that grouped dictionary
directly into PtychoDataContainerTorch.
"""

import pytest
import numpy as np

import torch


@pytest.fixture
def params_cfg_snapshot():
    """
    Save/restore params.cfg state (CRITICAL per CONFIG-001 finding).

    Source: tests/torch/test_config_bridge.py:151-160
    Why Critical: the legacy ground-truth patch fallback still reads translation
    policy from params.cfg. Failure to restore causes test pollution.
    """
    import ptycho.params as params
    snapshot = dict(params.cfg)
    sealed = params._sealed
    yield
    params.cfg.clear()
    params.cfg.update(snapshot)
    if sealed:
        params.seal()
    else:
        params.unseal()


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
    from ptycho.config.config import (
        TrainingConfig, ModelConfig, update_legacy_dict,
        SamplingConfig, DataConfig,
    )
    from ptycho import params as p
    from ptycho.raw_data import RawData

    # 1. Initialize params.cfg (MANDATORY per CLAUDE.md:76-93)
    config = TrainingConfig(
        model=ModelConfig(N=64, gridsize=2),
        sampling=SamplingConfig(n_groups=64, neighbor_count=4),
        data=DataConfig(nphotons=1e9),  # Required per Phase B config bridge validation
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

        # Phase C.C2: Implement PtychoDataContainerTorch
        from ptycho_torch.data_container_bridge import PtychoDataContainerTorch

        pt_grouped = minimal_raw_data.generate_grouped_data(
            N=64, K=4, nsamples=10, gridsize=2
        )

        # Create PyTorch container
        pt_container = PtychoDataContainerTorch(pt_grouped, minimal_raw_data.probeGuess)

        # API parity assertions: PyTorch container must expose same attributes
        assert hasattr(pt_container, 'X'), "PyTorch container missing X"
        assert hasattr(pt_container, 'Y'), "PyTorch container missing Y"
        assert hasattr(pt_container, 'Y_I'), "PyTorch container missing Y_I"
        assert hasattr(pt_container, 'Y_phi'), "PyTorch container missing Y_phi"
        assert hasattr(pt_container, 'coords_nominal'), "PyTorch container missing coords_nominal"
        assert hasattr(pt_container, 'coords_true'), "PyTorch container missing coords_true"
        assert hasattr(pt_container, 'coords'), "PyTorch container missing coords (alias)"
        assert hasattr(pt_container, 'probe'), "PyTorch container missing probe"
        assert hasattr(pt_container, 'nn_indices'), "PyTorch container missing nn_indices"
        assert hasattr(pt_container, 'global_offsets'), "PyTorch container missing global_offsets"
        assert hasattr(pt_container, 'local_offsets'), "PyTorch container missing local_offsets"

        # Shape parity
        assert pt_container.X.shape == (10, 64, 64, 4), \
            f"PyTorch X shape mismatch: {pt_container.X.shape}"
        assert pt_container.Y.shape == (10, 64, 64, 4), \
            f"PyTorch Y shape mismatch: {pt_container.Y.shape}"
        assert pt_container.Y_I.shape == (10, 64, 64, 4), \
            f"PyTorch Y_I shape mismatch: {pt_container.Y_I.shape}"
        assert pt_container.Y_phi.shape == (10, 64, 64, 4), \
            f"PyTorch Y_phi shape mismatch: {pt_container.Y_phi.shape}"
        assert pt_container.coords_nominal.shape == (10, 1, 2, 4), \
            f"PyTorch coords_nominal shape mismatch: {pt_container.coords_nominal.shape}"
        assert pt_container.probe.shape == (64, 64), \
            f"PyTorch probe shape mismatch: {pt_container.probe.shape}"
        assert pt_container.nn_indices.shape == (10, 4), \
            f"PyTorch nn_indices shape mismatch: {pt_container.nn_indices.shape}"
        assert pt_container.global_offsets.shape == (10, 1, 2, 1), \
            f"PyTorch global_offsets shape mismatch: {pt_container.global_offsets.shape}"

        # Dtype parity (torch-required as of Phase F3)
        assert isinstance(pt_container.X, (torch.Tensor, np.ndarray)), \
            f"PyTorch X type mismatch: {type(pt_container.X)}"
        assert isinstance(pt_container.Y, (torch.Tensor, np.ndarray)), \
            f"PyTorch Y type mismatch: {type(pt_container.Y)}"

        # Critical DATA-001 validation: Y must be complex64
        if isinstance(pt_container.Y, torch.Tensor):
            assert pt_container.Y.dtype == torch.complex64, \
                f"DATA-001 violation: PyTorch Y dtype must be torch.complex64, got {pt_container.Y.dtype}"
        else:
            assert pt_container.Y.dtype == np.complex64, \
                f"DATA-001 violation: NumPy Y dtype must be complex64, got {pt_container.Y.dtype}"


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

        # Phase C.C2/C.C3: Validate PyTorch container Y dtype
        from ptycho_torch.data_container_bridge import PtychoDataContainerTorch

        pt_grouped = minimal_raw_data.generate_grouped_data(
            N=64, K=4, nsamples=10, gridsize=2
        )
        pt_container = PtychoDataContainerTorch(pt_grouped, minimal_raw_data.probeGuess)

        # CRITICAL DATA-001 validation (torch-required as of Phase F3)
        if isinstance(pt_container.Y, torch.Tensor):
            assert pt_container.Y.dtype == torch.complex64, \
                f"DATA-001 violation: PyTorch Y must be torch.complex64, got {pt_container.Y.dtype}. " \
                f"Historical silent float64 conversion caused major training failure. " \
                f"See docs/findings.md:DATA-001 and specs/data_contracts.md:19."
        else:
            assert pt_container.Y.dtype == np.complex64, \
                f"DATA-001 violation: NumPy Y must be complex64, got {pt_container.Y.dtype}. " \
                f"Historical silent float64 conversion caused major training failure. " \
                f"See docs/findings.md:DATA-001 and specs/data_contracts.md:19."
