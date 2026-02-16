from pathlib import Path
import json

import numpy as np

from scripts.studies.prepare_fly001_128_external_split import prepare_dataset


def _write_raw_npz(path: Path) -> None:
    n = 10
    N = 128
    np.savez(
        path,
        diff3d=np.arange(n * N * N, dtype=np.uint16).reshape(n, N, N),
        xcoords=np.linspace(1.0, 2.0, n),
        ycoords=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64),
        xcoords_start=np.linspace(1.0, 2.0, n),
        ycoords_start=np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.float64),
        scan_index=np.zeros(n, dtype=np.int64),
        probeGuess=np.ones((N, N), dtype=np.complex64),
        objectGuess=np.ones((462, 461), dtype=np.complex64),
    )


def test_prepare_dataset_writes_canonical_and_disjoint_splits(tmp_path):
    raw = tmp_path / "fly001_128_train.npz"
    out = tmp_path / "fly001_128"
    _write_raw_npz(raw)

    result = prepare_dataset(raw_npz=raw, output_dir=out)

    canonical = np.load(result["canonical_npz"], allow_pickle=True)
    top = np.load(result["train_npz"], allow_pickle=True)
    bottom = np.load(result["test_npz"], allow_pickle=True)

    assert "diffraction" in canonical.files
    assert canonical["diffraction"].dtype == np.float32
    assert "diff3d" not in canonical.files

    y_top = top["ycoords"]
    y_bottom = bottom["ycoords"]
    assert y_top.min() >= result["split_threshold"]
    assert y_bottom.max() < result["split_threshold"]

    manifest = json.loads(Path(result["manifest_json"]).read_text())
    assert manifest["source_file"] == str(raw)
    assert manifest["n_total"] == 10
    assert manifest["n_train"] + manifest["n_test"] == 10
