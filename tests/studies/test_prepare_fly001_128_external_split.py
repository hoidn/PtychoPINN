from pathlib import Path
import json

import numpy as np

from scripts.studies.prepare_fly001_128_external_split import main, prepare_dataset


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


def test_prepare_dataset_writes_top_half_train_and_full_test(tmp_path):
    raw = tmp_path / "fly001_128_train.npz"
    out = tmp_path / "fly001_128"
    _write_raw_npz(raw)

    result = prepare_dataset(raw_npz=raw, output_dir=out)

    canonical = np.load(result["canonical_npz"], allow_pickle=True)
    top = np.load(result["train_npz"], allow_pickle=True)
    full_test = np.load(result["test_npz"], allow_pickle=True)

    assert "diffraction" in canonical.files
    assert canonical["diffraction"].dtype == np.float32
    assert "diff3d" not in canonical.files

    y_top = top["ycoords"]
    y_test = full_test["ycoords"]
    assert y_top.min() >= result["split_threshold"]
    assert y_top.size < canonical["ycoords"].size
    assert y_test.size == canonical["ycoords"].size
    assert np.array_equal(np.sort(y_test), np.sort(canonical["ycoords"]))

    manifest = json.loads(Path(result["manifest_json"]).read_text())
    assert manifest["source_file"] == str(raw)
    assert manifest["n_total"] == 10
    assert manifest["n_train"] < manifest["n_total"]
    assert manifest["n_test"] == manifest["n_total"]


def test_cli_writes_manifest_with_required_fields(tmp_path, monkeypatch, capsys):
    raw = tmp_path / "raw.npz"
    out = tmp_path / "out"
    _write_raw_npz(raw)

    argv = [
        "prepare_fly001_128_external_split.py",
        "--input-npz",
        str(raw),
        "--output-dir",
        str(out),
    ]
    monkeypatch.setattr("sys.argv", argv)
    main()
    _ = capsys.readouterr()

    manifest = json.loads((out / "manifest.json").read_text())
    for key in [
        "source_sha256",
        "canonical_npz",
        "train_npz",
        "test_npz",
        "n_train",
        "n_test",
    ]:
        assert key in manifest
