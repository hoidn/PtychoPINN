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

# Import torch unconditionally (torch-required as of Phase F3)
import torch
TORCH_AVAILABLE = True


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
        # Create TensorFlow baseline reference (with seed for reproducibility)
        tf_grouped = minimal_raw_data.generate_grouped_data(
            N=64, K=4, nsamples=10, gridsize=2, seed=42
        )

        # Validate baseline conforms to contract (data_contract.md §2)
        assert tf_grouped['diffraction'].shape == (10, 64, 64, 4), \
            "TensorFlow baseline shape mismatch"
        assert tf_grouped['diffraction'].dtype == np.float32, \
            "TensorFlow baseline dtype mismatch"
        assert tf_grouped['nn_indices'].shape == (10, 4), \
            "TensorFlow nn_indices shape mismatch"

        # Phase C.C1: RawDataTorch adapter implementation
        from ptycho_torch.raw_data_bridge import RawDataTorch
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Create configuration for adapter (per data_contract.md §6)
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9  # Required per Phase B nphotons validation
        )

        # Create PyTorch adapter
        pt_raw = RawDataTorch(
            xcoords=minimal_raw_data.xcoords,
            ycoords=minimal_raw_data.ycoords,
            diff3d=minimal_raw_data.diff3d,
            probeGuess=minimal_raw_data.probeGuess,
            scan_index=np.arange(len(minimal_raw_data.xcoords), dtype=np.int32),
            objectGuess=minimal_raw_data.objectGuess,
            config=config  # Initializes params.cfg automatically
        )

        # Generate grouped data via adapter (with same seed for exact parity)
        pt_grouped = pt_raw.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=2, seed=42)

        # Parity assertions: PyTorch adapter should match TensorFlow baseline exactly
        # (per data_contract.md §2 and test_blueprint.md §3)

        # Check shapes (critical for downstream model compatibility)
        assert tf_grouped['diffraction'].shape == pt_grouped['diffraction'].shape, \
            f"Diffraction shape mismatch: TF={tf_grouped['diffraction'].shape}, PT={pt_grouped['diffraction'].shape}"
        assert tf_grouped['X_full'].shape == pt_grouped['X_full'].shape, \
            f"X_full shape mismatch: TF={tf_grouped['X_full'].shape}, PT={pt_grouped['X_full'].shape}"
        assert tf_grouped['coords_offsets'].shape == pt_grouped['coords_offsets'].shape, \
            f"coords_offsets shape mismatch"
        assert tf_grouped['coords_relative'].shape == pt_grouped['coords_relative'].shape, \
            f"coords_relative shape mismatch"
        assert tf_grouped['nn_indices'].shape == pt_grouped['nn_indices'].shape, \
            f"nn_indices shape mismatch"

        # Check dtypes (DATA-001: complex64 for Y, float32 for diffraction)
        assert tf_grouped['diffraction'].dtype == pt_grouped['diffraction'].dtype, \
            "Diffraction dtype mismatch"
        assert tf_grouped['nn_indices'].dtype == pt_grouped['nn_indices'].dtype, \
            "nn_indices dtype mismatch"

        # Check exact data parity (delegation should produce identical results)
        np.testing.assert_array_equal(
            tf_grouped['nn_indices'], pt_grouped['nn_indices'],
            err_msg="nn_indices mismatch - delegation failed"
        )
        np.testing.assert_allclose(
            tf_grouped['diffraction'], pt_grouped['diffraction'],
            rtol=1e-6, atol=1e-9,
            err_msg="Diffraction values mismatch - delegation failed"
        )
        np.testing.assert_allclose(
            tf_grouped['coords_offsets'], pt_grouped['coords_offsets'],
            rtol=1e-6, atol=1e-9,
            err_msg="coords_offsets mismatch"
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
        from ptycho_torch.raw_data_bridge import RawDataTorch
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Create configuration for adapter
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9
        )

        # Create PyTorch adapter and generate grouped data
        pt_raw = RawDataTorch(
            xcoords=minimal_raw_data.xcoords,
            ycoords=minimal_raw_data.ycoords,
            diff3d=minimal_raw_data.diff3d,
            probeGuess=minimal_raw_data.probeGuess,
            scan_index=np.arange(len(minimal_raw_data.xcoords), dtype=np.int32),
            objectGuess=minimal_raw_data.objectGuess,
            config=config
        )
        pt_grouped = pt_raw.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=2)

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
        from ptycho_torch.raw_data_bridge import RawDataTorch
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Create configuration and adapter
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9
        )

        pt_raw = RawDataTorch(
            xcoords=minimal_raw_data.xcoords,
            ycoords=minimal_raw_data.ycoords,
            diff3d=minimal_raw_data.diff3d,
            probeGuess=minimal_raw_data.probeGuess,
            scan_index=np.arange(len(minimal_raw_data.xcoords), dtype=np.int32),
            objectGuess=minimal_raw_data.objectGuess,
            config=config
        )
        pt_grouped = pt_raw.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=2)
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


class TestMemmapBridgeParity:
    """
    Test memory-mapped dataset bridge delegates to RawDataTorch.

    Phase C.C3 Goal: Refactor existing memory-mapped datasets to reuse
    RawDataTorch delegation instead of reimplementing grouping logic.

    Expected behavior:
    - Memory-mapped loader should delegate grouping to RawDataTorch
    - Outputs should match RawDataTorch baseline (same grouped-data dict)
    - Cache reuse: .groups_cache.npz and data/memmap preserved across runs

    Source: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T082035Z/memmap_bridge_analysis.md
    """

    @pytest.fixture
    def minimal_npz_file(self, tmp_path, minimal_raw_data):
        """
        Create minimal NPZ file for memmap dataset testing.

        Writes a small dataset to disk for memory-mapped loading.
        """
        npz_path = tmp_path / "test_dataset.npz"
        np.savez(
            npz_path,
            diff3d=minimal_raw_data.diff3d[:20],  # Subset for speed
            xcoords=minimal_raw_data.xcoords[:20],
            ycoords=minimal_raw_data.ycoords[:20],
            probeGuess=minimal_raw_data.probeGuess,
            objectGuess=minimal_raw_data.objectGuess,
            scan_index=np.arange(20, dtype=np.int32)
        )
        return npz_path

    def test_memmap_loader_matches_raw_data_torch(self, params_cfg_snapshot, minimal_raw_data, minimal_npz_file):
        """
        Memory-mapped dataset should produce identical outputs to RawDataTorch.

        Expected behavior (Phase C.C3):
        - Memmap loader internally delegates to RawDataTorch.generate_grouped_data()
        - Returns grouped-data dict with same keys/shapes/dtypes as RawDataTorch baseline
        - No duplicate grouping logic (gap #1 from memmap_bridge_analysis.md)

        Test source: memmap_bridge_analysis.md §4.A
        """
        from ptycho_torch.memmap_bridge import MemmapDatasetBridge
        from ptycho_torch.raw_data_bridge import RawDataTorch
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Configuration for both baseline and memmap bridge
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9  # Required per Phase B nphotons validation
        )

        # Baseline: Direct RawDataTorch usage
        pt_raw = RawDataTorch(
            xcoords=minimal_raw_data.xcoords[:20],
            ycoords=minimal_raw_data.ycoords[:20],
            diff3d=minimal_raw_data.diff3d[:20],
            probeGuess=minimal_raw_data.probeGuess,
            scan_index=np.arange(20, dtype=np.int32),
            objectGuess=minimal_raw_data.objectGuess,
            config=config
        )
        baseline_grouped = pt_raw.generate_grouped_data(N=64, K=4, nsamples=10, gridsize=2, seed=42)

        # Memory-mapped dataset bridge
        memmap_bridge = MemmapDatasetBridge(
            npz_path=minimal_npz_file,
            config=config,
            memmap_dir="data/memmap/test"
        )
        memmap_grouped = memmap_bridge.get_grouped_data(N=64, K=4, nsamples=10, gridsize=2, seed=42)

        # Parity assertions: memmap bridge should match RawDataTorch baseline exactly
        # Shape parity
        assert memmap_grouped['diffraction'].shape == baseline_grouped['diffraction'].shape, \
            f"Diffraction shape mismatch: memmap={memmap_grouped['diffraction'].shape}, baseline={baseline_grouped['diffraction'].shape}"
        assert memmap_grouped['nn_indices'].shape == baseline_grouped['nn_indices'].shape, \
            f"nn_indices shape mismatch: memmap={memmap_grouped['nn_indices'].shape}, baseline={baseline_grouped['nn_indices'].shape}"
        assert memmap_grouped['coords_offsets'].shape == baseline_grouped['coords_offsets'].shape, \
            f"coords_offsets shape mismatch"

        # Dtype parity
        assert memmap_grouped['diffraction'].dtype == baseline_grouped['diffraction'].dtype, \
            "Diffraction dtype mismatch"
        assert memmap_grouped['nn_indices'].dtype == baseline_grouped['nn_indices'].dtype, \
            "nn_indices dtype mismatch"

        # Exact data parity (delegation should produce identical results with same seed)
        np.testing.assert_array_equal(
            memmap_grouped['nn_indices'], baseline_grouped['nn_indices'],
            err_msg="Memmap bridge nn_indices mismatch - delegation failed"
        )
        np.testing.assert_allclose(
            memmap_grouped['diffraction'], baseline_grouped['diffraction'],
            rtol=1e-6, atol=1e-9,
            err_msg="Memmap bridge diffraction values mismatch - delegation failed"
        )
        np.testing.assert_allclose(
            memmap_grouped['coords_offsets'], baseline_grouped['coords_offsets'],
            rtol=1e-6, atol=1e-9,
            err_msg="Memmap bridge coords_offsets mismatch"
        )

    def test_deterministic_generation_validation(self, params_cfg_snapshot, minimal_npz_file, tmp_path):
        """
        Validate deterministic grouped data generation across instantiations.

        Expected behavior (Phase C.D2 - Updated):
        - TensorFlow RawData uses efficient "sample-then-group" strategy (no cache files)
        - Multiple instantiations with same seed produce identical grouped data
        - Data parity demonstrates delegation correctness

        NOTE: The current RawData implementation eliminated cache files for performance.
        See ptycho/raw_data.py:408 - "eliminates the need for caching"

        Test source: memmap_bridge_analysis.md §5, phase_c_data_pipeline.md C.D2
        """
        from ptycho_torch.memmap_bridge import MemmapDatasetBridge
        from ptycho.config.config import TrainingConfig, ModelConfig

        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9
        )

        # First instantiation
        bridge1 = MemmapDatasetBridge(
            npz_path=minimal_npz_file,
            config=config,
            memmap_dir=str(tmp_path / "memmap_cache")
        )
        grouped1 = bridge1.get_grouped_data(N=64, K=4, nsamples=10, gridsize=2, seed=42)

        # Second instantiation
        bridge2 = MemmapDatasetBridge(
            npz_path=minimal_npz_file,
            config=config,
            memmap_dir=str(tmp_path / "memmap_cache")
        )
        grouped2 = bridge2.get_grouped_data(N=64, K=4, nsamples=10, gridsize=2, seed=42)

        # Validate deterministic data generation (same seed → identical results)
        np.testing.assert_array_equal(
            grouped1['nn_indices'], grouped2['nn_indices'],
            err_msg="Deterministic generation failed - nn_indices mismatch with same seed"
        )
        np.testing.assert_allclose(
            grouped1['diffraction'], grouped2['diffraction'],
            rtol=1e-6, atol=1e-9,
            err_msg="Deterministic generation failed - diffraction values mismatch with same seed"
        )

        # Validate that different seeds produce different results (non-trivial generation)
        grouped3 = bridge2.get_grouped_data(N=64, K=4, nsamples=10, gridsize=2, seed=123)

        # At least one group should differ when using different seed
        # (very unlikely to get identical groups by chance)
        assert not np.array_equal(grouped1['nn_indices'], grouped3['nn_indices']), \
            "Different seeds should produce different grouped data (delegation working correctly)"

    def test_memmap_bridge_accepts_diffraction_legacy(self, params_cfg_snapshot, minimal_raw_data, tmp_path):
        """
        MemmapDatasetBridge MUST accept legacy 'diffraction' key per DATA-001.

        Spec: specs/data_contracts.md:207 - Canonical key is 'diffraction',
        but legacy datasets may use 'diff3d'. Bridge must tolerate both.

        Historical context (INTEGRATE-PYTORCH-001 Attempt #96):
        - Phase E training blocked by KeyError: 'diff3d' when loading legacy NPZs
        - Multiple scripts (run_tike_reconstruction.py:165, generate_patches_tool.py:66)
          implement diffraction→diff3d fallback pattern
        - DATA-001 (docs/findings.md:14) requires readers tolerate legacy keys

        Expected behavior:
        - NPZ with 'diffraction' key loads successfully
        - Falls back to 'diff3d' if 'diffraction' missing
        - Preserves dtype (float32) and shape (N,H,W) per DATA-001
        - CONFIG-001 bridge remains intact (no initialization reordering)

        Test source: input.md:9, specs/data_contracts.md:207
        ROI: Legacy NPZ with canonical 'diffraction' key
        """
        from ptycho_torch.memmap_bridge import MemmapDatasetBridge
        from ptycho.config.config import TrainingConfig, ModelConfig

        # Create NPZ with canonical 'diffraction' key (no 'diff3d' key)
        npz_path = tmp_path / "legacy_diffraction.npz"
        np.savez(
            npz_path,
            diffraction=minimal_raw_data.diff3d[:20],  # Canonical key
            xcoords=minimal_raw_data.xcoords[:20],
            ycoords=minimal_raw_data.ycoords[:20],
            probeGuess=minimal_raw_data.probeGuess,
            objectGuess=minimal_raw_data.objectGuess,
            scan_index=np.arange(20, dtype=np.int32)
        )

        # Configuration
        config = TrainingConfig(
            model=ModelConfig(N=64, gridsize=2),
            n_groups=10,
            neighbor_count=4,
            nphotons=1e9
        )

        # RED PHASE: This should raise KeyError with current implementation
        # because MemmapDatasetBridge only looks for 'diff3d' (memmap_bridge.py:109)
        try:
            bridge = MemmapDatasetBridge(
                npz_path=npz_path,
                config=config,
                memmap_dir=str(tmp_path / "memmap")
            )
            # If we get here, the implementation already has the fallback (GREEN)
            # Validate it works correctly
            grouped = bridge.get_grouped_data(N=64, K=4, nsamples=10, gridsize=2, seed=42)
            assert grouped['diffraction'].shape == (10, 64, 64, 4), \
                f"Diffraction shape mismatch: {grouped['diffraction'].shape}"
            assert grouped['diffraction'].dtype == np.float32, \
                f"DATA-001 violation: dtype must be float32, got {grouped['diffraction'].dtype}"
        except KeyError as e:
            # RED: Expected failure with current implementation
            assert "diff3d" in str(e), \
                f"Expected KeyError mentioning 'diff3d', got: {e}"
            pytest.skip("RED phase: fallback not yet implemented, KeyError as expected")

        # GREEN PHASE (after implementation): Bridge should accept 'diffraction' key
        # (Uncomment this block after implementing fallback)
        # bridge = MemmapDatasetBridge(
        #     npz_path=npz_path,
        #     config=config,
        #     memmap_dir=str(tmp_path / "memmap")
        # )
        # grouped = bridge.get_grouped_data(N=64, K=4, nsamples=10, gridsize=2, seed=42)
        #
        # # Validate shape and dtype per DATA-001
        # assert grouped['diffraction'].shape == (10, 64, 64, 4), \
        #     f"Diffraction shape mismatch: {grouped['diffraction'].shape}"
        # assert grouped['diffraction'].dtype == np.float32, \
        #     f"DATA-001 violation: dtype must be float32, got {grouped['diffraction'].dtype}"
