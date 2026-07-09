import importlib
import inspect
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from ptycho import params


def _make_complex_object(seed: int = 0, size: int = 128) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.meshgrid(
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        np.linspace(-1.0, 1.0, size, dtype=np.float32),
        indexing="ij",
    )
    amp = 0.25 + np.exp(-3.0 * (x * x + y * y)) + 0.05 * rng.standard_normal((size, size))
    amp = np.clip(amp, 1e-4, None).astype(np.float32)
    phase = (0.8 * np.sin(6.0 * x) + 0.6 * np.cos(4.0 * y) + 0.05 * rng.standard_normal((size, size))).astype(
        np.float32
    )
    return (amp * np.exp(1j * phase)).astype(np.complex64)


def _as_hw1(obj: np.ndarray) -> np.ndarray:
    return np.asarray(obj, dtype=np.complex64)[..., None]


def test_evaluation_import_does_not_require_single_image_frc_module():
    repo_root = Path(__file__).resolve().parents[1]
    code = r'''
import builtins
real_import = builtins.__import__
def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "ptycho.single_image_frc":
        raise ModuleNotFoundError(name)
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = guarded_import
import ptycho.evaluation
print("evaluation import ok")
'''
    proc = subprocess.run([sys.executable, "-c", code], cwd=repo_root, text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert "evaluation import ok" in proc.stdout


def test_single_image_frc_removed_from_evaluation_api():
    import ptycho.evaluation as evaluation

    assert not hasattr(evaluation, "single_image_frc_metrics")
    signature = inspect.signature(evaluation.eval_reconstruction)
    assert not any(name.startswith("single_image_frc") for name in signature.parameters)


def test_eval_reconstruction_rejects_single_image_frc_kwarg():
    from ptycho import evaluation

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=11, size=128))
    pred = _as_hw1(_make_complex_object(seed=12, size=128))
    with pytest.raises(TypeError, match="single_image_frc"):
        evaluation.eval_reconstruction(pred, gt, label="pinn", single_image_frc=True)


def test_eval_reconstruction_default_keeps_reference_metrics_only():
    from ptycho.evaluation import eval_reconstruction

    params.set("offset", 4)
    gt = _as_hw1(_make_complex_object(seed=21, size=128))
    pred = _as_hw1(_make_complex_object(seed=22, size=128))
    out = eval_reconstruction(pred, gt, label="pinn")
    expected = {"mae", "mse", "psnr", "ssim", "ms_ssim", "frc50", "frc1over7", "frc"}
    assert expected.issubset(out)
    assert "single_frc50" not in out
    assert "single_frc1over7" not in out


def test_frc_package_no_longer_exports_single_image_frc_helpers():
    import frc

    assert "single_image_frc_metrics" not in frc.__all__
    assert not hasattr(frc, "single_image_frc_metrics")
    assert not hasattr(frc, "single_image_frc_curve")


def test_frc_single_image_frc_module_removed_from_tracked_api():
    sys.modules.pop("frc.single_image_frc", None)
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("frc.single_image_frc")


def test_frc50_uses_subbin_interpolation_for_threshold_crossing(monkeypatch):
    from ptycho import evaluation
    from ptycho.FRC import fourier_ring_corr

    curve = np.asarray([0.95, 0.72, 0.55, 0.45, 0.20], dtype=np.float64)

    def _fake_fsc(_a, _b):
        return curve.copy()

    monkeypatch.setattr(fourier_ring_corr, "FSC", _fake_fsc)
    _, cutoff = evaluation.frc50(np.ones((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32))
    assert np.isclose(float(cutoff), 2.5, atol=1e-9)


def test_frc50_returns_full_length_when_curve_never_crosses_threshold(monkeypatch):
    from ptycho import evaluation
    from ptycho.FRC import fourier_ring_corr

    curve = np.asarray([0.95, 0.72, 0.66, 0.51], dtype=np.float64)

    def _fake_fsc(_a, _b):
        return curve.copy()

    monkeypatch.setattr(fourier_ring_corr, "FSC", _fake_fsc)
    _, cutoff = evaluation.frc50(np.ones((16, 16), dtype=np.float32), np.ones((16, 16), dtype=np.float32))
    assert np.isclose(float(cutoff), float(len(curve)), atol=1e-9)


def test_frc_cutoffs_reports_interpolated_frc1over7(monkeypatch):
    from ptycho import evaluation
    from ptycho.FRC import fourier_ring_corr

    curve = np.asarray([0.80, 0.40, 0.20, 0.10], dtype=np.float64)

    def _fake_fsc(_a, _b):
        return curve.copy()

    monkeypatch.setattr(fourier_ring_corr, "FSC", _fake_fsc)
    _, cutoff_50, cutoff_1o7 = evaluation.frc_cutoffs(
        np.ones((16, 16), dtype=np.float32),
        np.ones((16, 16), dtype=np.float32),
    )
    assert np.isclose(float(cutoff_50), 0.75, atol=1e-9)
    assert np.isclose(float(cutoff_1o7), 2.5714285714285716, atol=1e-9)
