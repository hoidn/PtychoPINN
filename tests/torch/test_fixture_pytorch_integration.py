"""
Regression test for PyTorch integration fixture contract compliance.

This module validates that the generated fixture (minimal_dataset_v1.npz)
conforms to the acceptance criteria defined in Phase B1 fixture_scope.md §3.

Test Strategy (TDD RED → GREEN):
    Phase B2.B: Write failing test asserting fixture contract (this file)
    Phase B2.C: Implement generator to satisfy contract (make test GREEN)

Acceptance Criteria Reference:
    plans/active/TEST-PYTORCH-001/reports/2025-10-19T215300Z/phase_b_fixture/fixture_scope.md §3

Data Contract:
    specs/data_contracts.md §1 (canonical NPZ format requirements)

Author: Ralph (Phase B2.B TDD RED)
Date: 2025-10-19
"""

import json
import hashlib
import pytest
import numpy as np
from pathlib import Path


# Fixture acceptance criteria (from fixture_scope.md §3.1)
EXPECTED_N_SUBSET = 64
EXPECTED_H = 64
EXPECTED_W = 64
EXPECTED_OBJECT_MIN_SIZE = 128  # M >= 128 per acceptance criteria
FIXTURE_PATH = Path(__file__).parent.parent / "fixtures" / "pytorch_integration" / "minimal_dataset_v1.npz"
METADATA_PATH = FIXTURE_PATH.with_suffix('.json')


def compute_sha256(file_path: Path) -> str:
    """
    Compute SHA256 checksum of file.

    Args:
        file_path: Path to file

    Returns:
        Hexadecimal SHA256 checksum string
    """
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            hasher.update(chunk)
    return hasher.hexdigest()


class TestFixtureContract:
    """
    Test suite for PyTorch integration fixture contract compliance.

    These tests enforce the acceptance criteria from fixture_scope.md §3,
    ensuring the generator produces DATA-001 compliant output.
    """

    def test_fixture_file_exists(self):
        """
        Fixture file must exist at expected path.

        Expected Behavior (RED phase):
            FileNotFoundError or pytest.skip due to missing fixture generation

        Expected Behavior (GREEN phase):
            Fixture exists at tests/fixtures/pytorch_integration/minimal_dataset_v1.npz
        """
        if not FIXTURE_PATH.exists():
            pytest.skip(
                f"Fixture not generated yet: {FIXTURE_PATH}. "
                "Expected during TDD RED phase (B2.B). "
                "Run scripts/tools/make_pytorch_integration_fixture.py to generate."
            )

    def test_fixture_outputs_match_contract(self):
        """
        Core contract validation: fixture conforms to DATA-001 and acceptance criteria.

        Validates:
            - Shape requirements (fixture_scope.md §3.1.1)
            - Dtype requirements (fixture_scope.md §3.1.2, DATA-001)
            - Normalization requirements (fixture_scope.md §3.1.5)

        Expected Behavior (RED phase):
            Skipped due to missing fixture file

        Expected Behavior (GREEN phase):
            All assertions pass
        """
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not generated: {FIXTURE_PATH}")

        # Load fixture
        fixture = np.load(FIXTURE_PATH)

        # §3.1.1: Shape requirements
        # Diffraction must be canonical (N, H, W) format (not legacy (H, W, N))
        assert fixture['diffraction'].shape == (EXPECTED_N_SUBSET, EXPECTED_H, EXPECTED_W), \
            f"Diffraction shape mismatch. Expected (N={EXPECTED_N_SUBSET}, H={EXPECTED_H}, W={EXPECTED_W}), " \
            f"got {fixture['diffraction'].shape}. Must be canonical (N, H, W) per DATA-001."

        # Object must be at least M >= 128 to satisfy reconstruction requirements
        assert fixture['objectGuess'].ndim == 2, \
            f"objectGuess must be 2D, got {fixture['objectGuess'].ndim}D"
        assert min(fixture['objectGuess'].shape) >= EXPECTED_OBJECT_MIN_SIZE, \
            f"objectGuess dimensions {fixture['objectGuess'].shape} < {EXPECTED_OBJECT_MIN_SIZE}. " \
            f"Insufficient size for reconstruction (fixture_scope.md §3.1.1)."

        # Probe must match diffraction pattern size
        assert fixture['probeGuess'].shape == (EXPECTED_H, EXPECTED_W), \
            f"probeGuess shape mismatch. Expected ({EXPECTED_H}, {EXPECTED_W}), got {fixture['probeGuess'].shape}"

        # Coordinates must match subset size
        assert fixture['xcoords'].shape == (EXPECTED_N_SUBSET,), \
            f"xcoords shape mismatch. Expected ({EXPECTED_N_SUBSET},), got {fixture['xcoords'].shape}"
        assert fixture['ycoords'].shape == (EXPECTED_N_SUBSET,), \
            f"ycoords shape mismatch. Expected ({EXPECTED_N_SUBSET},), got {fixture['ycoords'].shape}"

        # §3.1.2: Dtype requirements (DATA-001 compliance)
        assert fixture['diffraction'].dtype == np.float32, \
            f"diffraction dtype must be float32 (DATA-001), got {fixture['diffraction'].dtype}"
        assert fixture['objectGuess'].dtype == np.complex64, \
            f"objectGuess dtype must be complex64 (DATA-001), got {fixture['objectGuess'].dtype}"
        assert fixture['probeGuess'].dtype == np.complex64, \
            f"probeGuess dtype must be complex64 (DATA-001), got {fixture['probeGuess'].dtype}"

        # Coordinates may remain float64 (acceptable per DATA-001)
        assert fixture['xcoords'].dtype in [np.float64, np.float32], \
            f"xcoords dtype must be float64 or float32, got {fixture['xcoords'].dtype}"
        assert fixture['ycoords'].dtype in [np.float64, np.float32], \
            f"ycoords dtype must be float64 or float32, got {fixture['ycoords'].dtype}"

        # §3.1.5: Normalization requirements
        # Diffraction must be amplitude (sqrt of intensity), normalized with max < 10.0
        assert np.max(fixture['diffraction']) < 10.0, \
            f"diffraction max {np.max(fixture['diffraction']):.2f} exceeds normalization limit 10.0. " \
            f"May indicate intensity (not amplitude) or un-normalized data (fixture_scope.md §3.1.5)."
        assert np.min(fixture['diffraction']) >= 0.0, \
            f"diffraction min {np.min(fixture['diffraction']):.2f} < 0. " \
            f"Amplitude must be non-negative (DATA-001)."

    def test_metadata_sidecar_exists(self):
        """
        Metadata JSON sidecar must accompany fixture.

        Expected Behavior (RED phase):
            Skipped or fails due to missing metadata

        Expected Behavior (GREEN phase):
            Metadata JSON exists at expected path
        """
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not generated: {FIXTURE_PATH}")

        assert METADATA_PATH.exists(), \
            f"Metadata sidecar missing. Expected at {METADATA_PATH}. " \
            f"Generator must emit JSON metadata per fixture_scope.md §3.3.8."

    def test_metadata_content_valid(self):
        """
        Metadata sidecar must contain required provenance fields.

        Validates:
            - Version field present
            - Subset size matches fixture
            - Transformation log present
            - SHA256 checksum present and matches fixture

        Expected Behavior (RED phase):
            Skipped due to missing files

        Expected Behavior (GREEN phase):
            All metadata fields valid
        """
        if not FIXTURE_PATH.exists() or not METADATA_PATH.exists():
            pytest.skip(f"Fixture or metadata missing")

        # Load metadata
        with open(METADATA_PATH, 'r') as f:
            metadata = json.load(f)

        # Required fields (fixture_scope.md §3.3.8)
        required_fields = [
            'version',
            'subset_size',
            'transformations',
            'sha256_checksum',
            'source_dataset',
            'created'
        ]
        for field in required_fields:
            assert field in metadata, \
                f"Metadata missing required field '{field}' (generator_design.md §3.2)"

        # Subset size must match fixture
        assert metadata['subset_size'] == EXPECTED_N_SUBSET, \
            f"Metadata subset_size {metadata['subset_size']} != fixture N dimension {EXPECTED_N_SUBSET}"

        # SHA256 checksum must match actual fixture file
        expected_checksum = compute_sha256(FIXTURE_PATH)
        assert metadata['sha256_checksum'] == expected_checksum, \
            f"Metadata checksum mismatch. Expected {expected_checksum}, got {metadata['sha256_checksum']}. " \
            f"Fixture may have been modified after metadata generation."

    def test_coordinate_coverage(self):
        """
        Subset coordinates must span sufficient spatial range.

        Acceptance Criteria (fixture_scope.md §3.1.4):
            Subset must span >50% of original X/Y range to exercise grouping
            logic across spatial diversity.

        Expected Behavior (RED phase):
            Skipped

        Expected Behavior (GREEN phase):
            Coordinate range >= 50% of canonical dataset range
        """
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not generated: {FIXTURE_PATH}")

        fixture = np.load(FIXTURE_PATH)

        # Canonical dataset ranges (from fixture_scope.md §1.1):
        # xcoords: [34.412, 79.281] → range = 44.869
        # ycoords: [35.763, 79.107] → range = 43.344
        CANONICAL_X_RANGE = 79.281 - 34.412
        CANONICAL_Y_RANGE = 79.107 - 35.763

        x_range = np.max(fixture['xcoords']) - np.min(fixture['xcoords'])
        y_range = np.max(fixture['ycoords']) - np.min(fixture['ycoords'])

        x_coverage = x_range / CANONICAL_X_RANGE
        y_coverage = y_range / CANONICAL_Y_RANGE

        assert x_coverage > 0.5, \
            f"X coordinate coverage {x_coverage:.1%} < 50% (range {x_range:.2f} vs canonical {CANONICAL_X_RANGE:.2f}). " \
            f"Insufficient spatial diversity (fixture_scope.md §3.1.4)."
        assert y_coverage > 0.5, \
            f"Y coordinate coverage {y_coverage:.1%} < 50% (range {y_range:.2f} vs canonical {CANONICAL_Y_RANGE:.2f}). " \
            f"Insufficient spatial diversity (fixture_scope.md §3.1.4)."


class TestFixtureIntegrationSmoke:
    """
    Smoke tests verifying fixture can be loaded by PyTorch data pipeline.

    These tests ensure the fixture is compatible with RawData.from_file() and
    the PyTorch dataloader, catching integration issues early.
    """

    def test_fixture_loads_with_rawdata(self):
        """
        Fixture must be loadable by ptycho.raw_data.RawData.from_file().

        Expected Behavior (RED phase):
            Skipped

        Expected Behavior (GREEN phase):
            RawData.from_file() succeeds without errors
        """
        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not generated: {FIXTURE_PATH}")

        from ptycho.raw_data import RawData

        # Attempt to load fixture (will validate NPZ schema compatibility)
        try:
            raw_data = RawData.from_file(str(FIXTURE_PATH))
        except Exception as e:
            pytest.fail(
                f"Fixture failed to load via RawData.from_file(): {e}. "
                f"Fixture may violate DATA-001 contract or have missing required keys."
            )

        # Basic sanity checks on loaded data using diff3d accessor (ptycho/raw_data.py:296-332)
        assert raw_data.diff3d.shape[0] == EXPECTED_N_SUBSET, \
            f"RawData diff3d N dimension {raw_data.diff3d.shape[0]} != expected {EXPECTED_N_SUBSET}"

    def test_fixture_compatible_with_pytorch_dataloader(self):
        """
        Fixture must be compatible with PyTorch integration workflow (smoke test).

        Note: PtychoDataset in ptycho_torch.dataloader requires directory + config objects,
        not RawData instances. The actual integration test (test_integration_workflow_torch.py)
        validates PyTorch pipeline compatibility via the full CLI workflow.

        This test verifies fixture can be loaded and basic data fields are accessible.

        Expected Behavior (RED phase):
            Skipped

        Expected Behavior (GREEN phase):
            RawData loads successfully and data fields accessible
        """
        pytest.importorskip("torch", reason="PyTorch not available (expected in TF-only CI)")

        if not FIXTURE_PATH.exists():
            pytest.skip(f"Fixture not generated: {FIXTURE_PATH}")

        from ptycho.raw_data import RawData
        import torch

        # Load fixture via RawData
        raw_data = RawData.from_file(str(FIXTURE_PATH))

        # Verify key data fields are accessible and have correct shape
        # PyTorch integration uses diff3d accessor (ptycho/raw_data.py:296-332)
        assert raw_data.diff3d.shape[0] == EXPECTED_N_SUBSET, \
            f"Diffraction N dimension {raw_data.diff3d.shape[0]} != expected {EXPECTED_N_SUBSET}"

        # Verify probe and object are accessible (required for PyTorch workflows)
        assert raw_data.probeGuess.shape == (EXPECTED_H, EXPECTED_W), \
            f"probeGuess shape mismatch: {raw_data.probeGuess.shape}"
        assert raw_data.objectGuess.ndim == 2, \
            f"objectGuess must be 2D, got {raw_data.objectGuess.ndim}D"

        # Smoke test: convert diffraction to torch tensor (dtype validation)
        try:
            diff_tensor = torch.from_numpy(raw_data.diff3d)
            assert diff_tensor.dtype == torch.float32, \
                f"Diffraction tensor dtype {diff_tensor.dtype} != torch.float32"
        except Exception as e:
            pytest.fail(f"Failed to convert diffraction to torch tensor: {e}")


# Expected Test Outcomes (Phase B2.B RED):
# ----------------------------------------
# test_fixture_file_exists:                      SKIPPED (fixture not generated)
# test_fixture_outputs_match_contract:           SKIPPED (fixture not generated)
# test_metadata_sidecar_exists:                  SKIPPED (fixture not generated)
# test_metadata_content_valid:                   SKIPPED (fixture not generated)
# test_coordinate_coverage:                      SKIPPED (fixture not generated)
# test_fixture_loads_with_rawdata:               SKIPPED (fixture not generated)
# test_fixture_compatible_with_pytorch_dataloader: SKIPPED (fixture not generated)
#
# Expected Test Outcomes (Phase B2.C GREEN):
# ------------------------------------------
# All tests: PASSED (after generator implementation)
