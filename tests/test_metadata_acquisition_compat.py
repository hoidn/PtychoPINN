import numpy as np

from ptycho.metadata import MetadataManager
from ptycho.raw_data import RawData


def test_saved_metadata_is_safe_for_canonical_acquisition_decode(tmp_path):
    path = tmp_path / "metadata.npz"
    metadata = {"physics_parameters": {"nphotons": 123.0}}
    MetadataManager.save_with_metadata(
        str(path),
        {
            "xcoords": np.arange(2, dtype=np.float64),
            "ycoords": np.arange(2, dtype=np.float64),
            "diff3d": np.ones((2, 4, 4), dtype=np.float32),
            "probeGuess": np.ones((4, 4), dtype=np.complex64),
        },
        metadata,
    )

    with np.load(path) as archive:
        assert archive[MetadataManager.METADATA_KEY].dtype.kind in {"U", "S"}

    assert RawData.from_file(path).metadata == metadata


def test_metadata_manager_reads_legacy_object_metadata(tmp_path):
    path = tmp_path / "legacy-metadata.npz"
    np.savez(
        path,
        values=np.arange(2),
        _metadata=np.array('{"source": "legacy"}', dtype=object),
    )

    data, metadata = MetadataManager.load_with_metadata(str(path))

    np.testing.assert_array_equal(data["values"], np.arange(2))
    assert metadata == {"source": "legacy"}


def test_metadata_manager_decodes_utf8_bytes_metadata(tmp_path):
    path = tmp_path / "bytes-metadata.npz"
    np.savez(
        path,
        values=np.arange(2),
        _metadata=np.array(b'{"source": "bytes"}'),
    )

    data, metadata = MetadataManager.load_with_metadata(str(path))

    np.testing.assert_array_equal(data["values"], np.arange(2))
    assert metadata == {"source": "bytes"}


def test_metadata_manager_ignores_malformed_legacy_object_metadata(tmp_path):
    path = tmp_path / "malformed-legacy-metadata.npz"
    np.savez(path, values=np.arange(2), _metadata=np.array({"not": "json"}))

    data, metadata = MetadataManager.load_with_metadata(str(path))

    np.testing.assert_array_equal(data["values"], np.arange(2))
    assert metadata is None
