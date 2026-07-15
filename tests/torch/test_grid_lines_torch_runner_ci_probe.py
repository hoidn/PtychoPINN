"""Unit tests for the CI physics-probe selection helper (CI-SCALE-001 E2).

Pure/numpy-only tests for ``_select_ci_probe`` — no Lightning, no GPU, no
training. The helper decides whether the CI (count-intensity) container probe
should be the stored ``probeGuess`` or the ``probe_simulated`` array that the
grid-lines writer (Task E1) stores when it is available.
"""
import numpy as np
import pytest

from scripts.studies.grid_lines_torch_runner import _select_ci_probe


N = 8


def _guess(scale: float = 1.0) -> np.ndarray:
    base = (np.arange(N * N, dtype=np.float32).reshape(N, N) + 1.0)
    return (base * scale).astype(np.complex64)


def _split(*, probe_guess=None, probe_simulated=None, diffraction=True) -> dict:
    """Assemble a dataset split dict with the optional keys populated."""
    split = {}
    if diffraction:
        split["diffraction"] = np.ones((2, N, N, 1), dtype=np.float32)
    if probe_guess is not None:
        split["probeGuess"] = probe_guess
    if probe_simulated is not None:
        split["probe_simulated"] = probe_simulated
    return split


def test_ci_active_both_carry_simulated_selects_simulated():
    guess = _guess()
    sim = _guess(3.168)
    train = _split(probe_guess=guess, probe_simulated=sim)
    test = _split(probe_guess=guess, probe_simulated=sim)

    probe, test_probe, provenance = _select_ci_probe(train, test, ci_dict_active=True)

    assert provenance == "simulated"
    assert probe is sim
    assert test_probe is sim


def test_ci_active_neither_carries_simulated_uses_guess():
    train_guess = _guess()
    test_guess = _guess(2.0)
    train = _split(probe_guess=train_guess)
    test = _split(probe_guess=test_guess)

    probe, test_probe, provenance = _select_ci_probe(train, test, ci_dict_active=True)

    assert provenance == "stored_guess"
    assert probe is train_guess
    assert test_probe is test_guess


def test_ci_active_train_only_simulated_real_test_raises():
    guess = _guess()
    sim = _guess(3.168)
    train = _split(probe_guess=guess, probe_simulated=sim)
    test = _split(probe_guess=guess)  # real test split, no probe_simulated

    with pytest.raises(ValueError, match="probe_simulated"):
        _select_ci_probe(train, test, ci_dict_active=True)


def test_ci_active_test_only_simulated_raises():
    guess = _guess()
    sim = _guess(3.168)
    train = _split(probe_guess=guess)
    test = _split(probe_guess=guess, probe_simulated=sim)

    with pytest.raises(ValueError, match="probe_simulated"):
        _select_ci_probe(train, test, ci_dict_active=True)


def test_ci_inactive_ignores_simulated():
    guess = _guess()
    sim = _guess(3.168)
    train = _split(probe_guess=guess, probe_simulated=sim)
    test = _split(probe_guess=guess, probe_simulated=sim)

    probe, test_probe, provenance = _select_ci_probe(train, test, ci_dict_active=False)

    assert provenance == "stored_guess"
    assert probe is guess
    assert test_probe is guess


def test_no_test_split_does_not_crash():
    guess = _guess()
    sim = _guess(3.168)
    train = _split(probe_guess=guess, probe_simulated=sim)

    for empty in (None, {}):
        probe, test_probe, provenance = _select_ci_probe(
            train, empty, ci_dict_active=True
        )
        assert provenance == "simulated"
        assert probe is sim
        assert test_probe is None


def test_ci_active_test_split_without_any_probe_falls_back_no_raise():
    """A test split carrying diffraction but no probe at all must not raise the
    mismatch error; provenance follows the train split and test_probe is None so
    the runner's ``if test_probe is None: test_probe = probe`` fallback applies."""
    guess = _guess()
    sim = _guess(3.168)
    train = _split(probe_guess=guess, probe_simulated=sim)
    test = _split(diffraction=True)  # no probeGuess, no probe_simulated

    probe, test_probe, provenance = _select_ci_probe(train, test, ci_dict_active=True)

    assert provenance == "simulated"
    assert probe is sim
    assert test_probe is None
