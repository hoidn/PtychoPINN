"""Fast, no-training unit tests for scripts/studies/flux_sweep_eval.py's CLI
parameterization (Task E2b) and its pure metric/persistence helpers (I1, I4).

Covers only argparse plumbing, the scratch-dir derivation, and synthetic-array
unit tests of `measurement_domain_error`/the scale-JSON writer -- no
checkpoint loading, no GPU, no `main()` execution.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts" / "studies"))
import flux_sweep_eval as fse  # noqa: E402


def test_default_out_matches_phase1_checkpoint_root():
    args = fse.build_parser().parse_args([])
    assert args.out == fse.OUT


def test_default_label_is_cnn():
    args = fse.build_parser().parse_args([])
    assert args.label == "cnn"


def test_default_skip_anchor_is_false():
    args = fse.build_parser().parse_args([])
    assert args.skip_anchor is False


def test_parses_out_label_and_skip_anchor_overrides():
    args = fse.build_parser().parse_args(
        ["--out", "/tmp/x", "--label", "cnn", "--skip-anchor"]
    )
    assert args.out == Path("/tmp/x")
    assert args.label == "cnn"
    assert args.skip_anchor is True


def test_scratch_for_derives_label_specific_directory():
    scratch = fse.scratch_for("cnn")
    assert "cnn" in str(scratch)
    assert scratch == fse.SC / "cnn"


# ---------------------------------------------------------------------------
# I1 -- measurement-domain (Fourier) error metric (plan L52/L89). Pure numpy,
# no GPU/checkpoint: identical arrays -> 0; a known uniform scale -> a known
# nonzero relative-L2 value.
# ---------------------------------------------------------------------------

def test_measurement_domain_error_zero_for_identical_arrays():
    rng = np.random.default_rng(0)
    measured = rng.uniform(0, 5, size=(8, 8))

    err = fse.measurement_domain_error(measured, measured)

    assert err == pytest.approx(0.0, abs=1e-12)


def test_measurement_domain_error_known_nonzero_for_scaled_prediction():
    rng = np.random.default_rng(1)
    measured = rng.uniform(0.1, 5, size=(8, 8))
    pred = 1.5 * measured

    err = fse.measurement_domain_error(pred, measured)

    # ||1.5*m - m|| / ||m|| == 0.5 exactly, independent of the array contents.
    assert err == pytest.approx(0.5, rel=1e-9)


def test_measurement_domain_error_double_prediction_gives_relative_error_one():
    rng = np.random.default_rng(2)
    measured = rng.uniform(0.1, 5, size=(4, 4))
    pred = 2.0 * measured

    err = fse.measurement_domain_error(pred, measured)

    assert err == pytest.approx(1.0, rel=1e-9)


# ---------------------------------------------------------------------------
# I4 -- full-precision machine-readable scale-scalar dump. Pure JSON
# round-trip on synthetic values (no checkpoint/model dependency).
# ---------------------------------------------------------------------------

def test_scale_json_round_trips_full_precision_values(tmp_path):
    payload = fse.build_scale_payload(
        "cnn",
        [
            dict(mean=1, s1=0.029403123456789, s2=-0.010982222222222, cA=0.031398765432109,
                 on=0.023456789012345, off=8.368712345678901,
                 meas_err_on=0.041234567890123, meas_err_off=0.198765432109876),
        ],
    )
    out_path = tmp_path / "cnn_scale.json"

    fse.write_scale_json(out_path, payload)
    loaded = json.loads(out_path.read_text())

    assert loaded == payload
    row = loaded["rows"][0]
    assert row["s1"] == pytest.approx(0.029403123456789, abs=0.0, rel=1e-15)
    assert row["mean_ct"] == 1


def test_build_scale_payload_derives_c_phi_deg_from_s1_s2():
    payload = fse.build_scale_payload(
        "cnn_ri",
        [dict(mean=100, s1=1.0, s2=1.0, cA=1.4142135623730951, on=1.0, off=1.0,
              meas_err_on=0.01, meas_err_off=0.5)],
    )

    assert payload["label"] == "cnn_ri"
    row = payload["rows"][0]
    assert row["c_phi_deg"] == pytest.approx(45.0)
    assert row["c_A"] == pytest.approx(1.4142135623730951)
    assert row["O_on"] == pytest.approx(1.0)
    assert row["O_off"] == pytest.approx(1.0)
    assert row["measurement_error_on"] == pytest.approx(0.01)
    assert row["measurement_error_off"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# physics_forward_mode guard (review Fix 2). Pure string-in/string-or-None-out
# helper, no checkpoint/model dependency.
# ---------------------------------------------------------------------------

def test_physics_forward_mode_warning_none_for_amplitude():
    assert fse.physics_forward_mode_warning("amplitude") is None


def test_physics_forward_mode_warning_nonempty_for_rectangular_scaled():
    warning = fse.physics_forward_mode_warning("rectangular_scaled")

    assert warning is not None
    assert "rectangular_scaled" in warning
    assert "predicted_diffraction_amplitude" in warning
