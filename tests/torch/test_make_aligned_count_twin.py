"""Tests for scripts/studies/make_aligned_count_twin.py.

CPU-only: builds a small synthetic NPZ fixture in tmp_path so no GPU or
frozen `.artifacts/` data is required. The fixture constructs diffraction
amplitudes from known integer counts divided by a chosen S_true, so the
tool's recovered S can be checked exactly against S_true.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ptycho.config.config import ModelConfig, TrainingConfig
from ptycho.metadata import MetadataManager

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "studies" / "make_aligned_count_twin.py"

SHAPE = (8, 16, 16, 1)
S_TRUE = {"train": 37.5, "test": 52.0}
SEEDS = {"train": 1, "test": 2}


def _make_split_npz(path: Path, seed: int, s_true: float) -> float:
    """Write a synthetic split NPZ; return the true nphotons used."""
    rng = np.random.default_rng(seed)
    n, h, w, c = SHAPE
    yy, xx = np.mgrid[0:h, 0:w]
    base = 50.0 + 40.0 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    base = np.clip(base, 1.0, None)
    counts = rng.poisson(lam=base[None, :, :, None], size=SHAPE).astype(np.float64)

    nphotons_true = float(counts.sum(axis=(1, 2, 3)).mean())
    diffraction = (np.sqrt(counts) / s_true).astype(np.float32)

    Y_I = rng.random((n, h, w, c)).astype(np.float32)
    coords_nominal = rng.random((n, 2)).astype(np.float32)
    probeGuess = (rng.random((h, w)) + 1j * rng.random((h, w))).astype(np.complex64)
    metadata = np.array({"nphotons": nphotons_true}, dtype=object)

    np.savez(
        path,
        diffraction=diffraction,
        Y_I=Y_I,
        coords_nominal=coords_nominal,
        probeGuess=probeGuess,
        _metadata=metadata,
    )
    return nphotons_true


def _run_tool(src_dir: Path, out_root: Path) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--src-dir", str(src_dir), "--out-root", str(out_root)],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    return result


@pytest.fixture
def built_twins(tmp_path):
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_root = tmp_path / "out"
    nphotons_true = {
        split: _make_split_npz(src_dir / f"{split}.npz", SEEDS[split], S_TRUE[split])
        for split in ("train", "test")
    }
    _run_tool(src_dir, out_root)
    return {"src_dir": src_dir, "out_root": out_root, "nphotons_true": nphotons_true}


def test_recovered_s_matches_s_true(built_twins):
    provenance = json.loads((built_twins["out_root"] / "provenance.json").read_text())
    for split in ("train", "test"):
        assert provenance[split]["S"] == pytest.approx(S_TRUE[split], rel=1e-6)


def test_data_amp_mean_total_matches_nphotons(built_twins):
    out_root = built_twins["out_root"]
    for split in ("train", "test"):
        with np.load(out_root / "data_amp" / f"{split}.npz") as npz:
            diffraction = npz["diffraction"]
        total = np.sum(diffraction.astype(np.float64) ** 2, axis=(1, 2, 3)).mean()
        assert total == pytest.approx(built_twins["nphotons_true"][split], rel=1e-6)


def test_data_intensity_equals_data_amp_squared(built_twins):
    out_root = built_twins["out_root"]
    for split in ("train", "test"):
        with np.load(out_root / "data_amp" / f"{split}.npz") as npz:
            amp = npz["diffraction"]
        with np.load(out_root / "data_intensity" / f"{split}.npz") as npz:
            intensity = npz["diffraction"]
        np.testing.assert_allclose(intensity, amp ** 2, rtol=1e-6)


def test_non_diffraction_keys_byte_identical(built_twins):
    src_dir, out_root = built_twins["src_dir"], built_twins["out_root"]
    for split in ("train", "test"):
        with np.load(src_dir / f"{split}.npz", allow_pickle=True) as src:
            src_keys = set(src.files)
            src_data = {k: src[k] for k in src_keys}
        for out_dir in ("data_amp", "data_intensity"):
            with np.load(out_root / out_dir / f"{split}.npz", allow_pickle=True) as out:
                assert set(out.files) == src_keys
                for key in src_keys - {"diffraction"}:
                    if key == "_metadata":
                        assert out[key].item() == src_data[key].item()
                    else:
                        assert np.array_equal(out[key], src_data[key])


def test_recovered_s_matches_s_true_with_canonical_metadata_schema(tmp_path):
    """The real Task-2 pipeline writes `_metadata` via MetadataManager: a JSON
    string with nphotons nested under physics_parameters. Cover that schema
    (rather than the flat/raw-dict shortcut used by `_make_split_npz` above)."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    out_root = tmp_path / "out"
    s_true = 44.0
    seed = 7

    rng = np.random.default_rng(seed)
    n, h, w, c = SHAPE
    yy, xx = np.mgrid[0:h, 0:w]
    base = 50.0 + 40.0 * np.sin(xx / 3.0) * np.cos(yy / 4.0)
    base = np.clip(base, 1.0, None)
    counts = rng.poisson(lam=base[None, :, :, None], size=SHAPE).astype(np.float64)
    nphotons_true = float(counts.sum(axis=(1, 2, 3)).mean())
    diffraction = (np.sqrt(counts) / s_true).astype(np.float32)

    data_dict = {
        "diffraction": diffraction,
        "Y_I": rng.random((n, h, w, c)).astype(np.float32),
        "coords_nominal": rng.random((n, 2)).astype(np.float32),
        "probeGuess": (rng.random((h, w)) + 1j * rng.random((h, w))).astype(np.complex64),
    }
    config = TrainingConfig(model=ModelConfig(), nphotons=nphotons_true)
    metadata = MetadataManager.create_metadata(config, script_name="test_make_aligned_count_twin")
    MetadataManager.save_with_metadata(str(src_dir / "train.npz"), data_dict, metadata)
    _make_split_npz(src_dir / "test.npz", SEEDS["test"], S_TRUE["test"])

    _run_tool(src_dir, out_root)

    provenance = json.loads((out_root / "provenance.json").read_text())
    assert provenance["train"]["S"] == pytest.approx(s_true, rel=1e-6)


def test_output_diffraction_dtype_float32(built_twins):
    out_root = built_twins["out_root"]
    for split in ("train", "test"):
        with np.load(out_root / "data_amp" / f"{split}.npz") as npz:
            assert npz["diffraction"].dtype == np.float32
        with np.load(out_root / "data_intensity" / f"{split}.npz") as npz:
            assert npz["diffraction"].dtype == np.float32
