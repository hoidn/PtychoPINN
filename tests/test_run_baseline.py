"""Focused tests for the maintained TensorFlow baseline CLI."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import tensorflow as tf

from ptycho.config.config import ModelConfig, TrainingConfig
from scripts import run_baseline


def _container(*, channels: int, offset_channels: int):
    shape = (2, 8, 8, channels)
    data = SimpleNamespace(
        X=tf.ones(shape, dtype=tf.float32),
        Y_I=tf.ones(shape, dtype=tf.float32),
        Y_phi=tf.zeros(shape, dtype=tf.float32),
        global_offsets=tf.ones(
            (2, 1, 2, offset_channels),
            dtype=tf.float64,
        ),
    )
    return SimpleNamespace(train_data=data, test_data=data)


def test_prepare_baseline_data_flattens_gridsize2_and_replicates_group_offsets():
    dataset = _container(channels=4, offset_channels=1)
    config = SimpleNamespace(model=SimpleNamespace(gridsize=2))

    prepared = run_baseline._prepare_baseline_data_inputs(dataset, config)

    x_train, y_i, y_phi, x_test, offsets = prepared
    assert x_train.shape == (8, 8, 8, 1)
    assert y_i.shape == (8, 8, 8, 1)
    assert y_phi.shape == (8, 8, 8, 1)
    assert x_test.shape == (8, 8, 8, 1)
    assert offsets.shape == (8, 1, 2, 1)


def test_prepare_baseline_data_passes_through_gridsize1():
    dataset = _container(channels=1, offset_channels=1)
    config = SimpleNamespace(model=SimpleNamespace(gridsize=1))

    prepared = run_baseline._prepare_baseline_data_inputs(dataset, config)

    for value in prepared[:4]:
        assert value.shape == (2, 8, 8, 1)
    assert prepared[4].shape == (2, 1, 2, 1)
    np.testing.assert_allclose(prepared[0], dataset.train_data.X)


def _training_config(tmp_path: Path, *, gridsize: int = 2) -> TrainingConfig:
    return TrainingConfig(
        model=ModelConfig(N=64, gridsize=gridsize, model_type="supervised"),
        train_data_file=tmp_path / "train.npz",
        test_data_file=tmp_path / "test.npz",
        output_dir=tmp_path / "outputs",
        nepochs=3,
        batch_size=8,
    )


def test_run_identity_comes_only_from_resolved_config_and_explicit_timestamp(
    tmp_path: Path,
):
    from ptycho import params
    from ptycho.config.legacy_state import legacy_params_scope

    config = _training_config(tmp_path)
    with legacy_params_scope():
        params.cfg.clear()
        params.cfg.update(
            {
                "label": "poison-label",
                "output_prefix": "poison-output",
                "timestamp": "poison-time",
            }
        )

        identity = run_baseline._resolve_run_identity(
            config,
            timestamp="07/28/2026, 12:34:56",
        )

        assert identity.label == "baseline_gs2"
        assert identity.timestamp == "07/28/2026, 12:34:56"
        assert identity.output_prefix == (
            f"{config.output_dir}/07-28-2026-12.34.56_baseline_gs2/"
        )


@pytest.mark.parametrize("initially_sealed", (False, True))
def test_tensorflow_leaf_scope_projects_owned_values_and_restores_success(
    initially_sealed: bool,
    tmp_path: Path,
):
    from ptycho import params
    from ptycho.config.legacy_state import legacy_params_scope

    config = _training_config(tmp_path)
    with legacy_params_scope():
        params.cfg.clear()
        params.cfg.update({"N": -1, "ambient": "keep"})
        if initially_sealed:
            params.seal()
        else:
            params.unseal()

        with run_baseline._baseline_tensorflow_scope(
            config,
            intensity_scale=17.5,
        ):
            assert params.cfg["N"] == 64
            assert params.cfg["gridsize"] == 2
            assert params.cfg["nepochs"] == 3
            assert params.cfg["batch_size"] == 8
            assert params.cfg["intensity_scale"] == 17.5

        assert params.cfg == {"N": -1, "ambient": "keep"}
        assert params._sealed is initially_sealed


def test_tensorflow_leaf_scope_restores_after_failure(tmp_path: Path):
    from ptycho import params
    from ptycho.config.legacy_state import legacy_params_scope

    config = _training_config(tmp_path, gridsize=1)
    with legacy_params_scope():
        params.cfg.clear()
        params.cfg.update({"ambient": "keep"})
        params.seal()

        with pytest.raises(RuntimeError, match="leaf failed"):
            with run_baseline._baseline_tensorflow_scope(
                config,
                intensity_scale=9.0,
            ):
                assert params.cfg["N"] == 64
                raise RuntimeError("leaf failed")

        assert params.cfg == {"ambient": "keep"}
        assert params._sealed is True


def test_run_baseline_uses_owned_runtime_values_and_explicit_persistence(
    monkeypatch,
    tmp_path: Path,
):
    from ptycho import params
    from ptycho.config.legacy_state import legacy_params_scope

    config = _training_config(tmp_path)
    offsets = np.asarray(
        [[[[1.0], [2.0]]], [[[3.0], [4.0]]]],
        dtype=np.float64,
    )
    train_data = SimpleNamespace(norm_Y_I=np.asarray(13.0, dtype=np.float32))
    test_data = SimpleNamespace(global_offsets=offsets)
    dataset = SimpleNamespace(train_data=train_data, test_data=test_data)
    ground_truth = np.ones((1, 12, 12, 1), dtype=np.complex64)
    stitched = np.ones((1, 12, 12, 1), dtype=np.complex64)
    aligned_recon = np.ones((8, 8), dtype=np.complex64)
    aligned_gt = np.ones((8, 8), dtype=np.complex64)
    observed = {}

    monkeypatch.setattr(
        run_baseline,
        "_load_baseline_dataset",
        lambda _config: (dataset, ground_truth),
    )
    monkeypatch.setattr(
        run_baseline,
        "_prepare_baseline_data_inputs",
        lambda _dataset, _config: (
            np.zeros((2, 8, 8, 1), dtype=np.float32),
            np.zeros((2, 8, 8, 1), dtype=np.float32),
            np.zeros((2, 8, 8, 1), dtype=np.float32),
            np.zeros((2, 8, 8, 1), dtype=np.float32),
            offsets,
        ),
    )

    class FakeModel:
        def save(self, path):
            observed["model_path"] = str(path)

    def fake_train(*args, **kwargs):
        observed["train"] = (args, kwargs)
        return FakeModel(), object(), "pred_i", "pred_phi"

    monkeypatch.setattr(run_baseline, "_train_baseline_and_predict", fake_train)
    monkeypatch.setattr(
        run_baseline,
        "_reassemble_predictions",
        lambda *args, **kwargs: stitched,
    )

    def fake_align(**kwargs):
        observed["alignment"] = kwargs
        return aligned_recon, aligned_gt

    monkeypatch.setattr(run_baseline, "align_for_evaluation", fake_align)

    def fake_eval(recon, gt, *, trim_offset, **kwargs):
        observed["eval"] = (recon, gt, trim_offset, kwargs)
        return {"mae": (0.1, 0.2), "psnr": (20.0, 21.0)}

    monkeypatch.setattr(
        run_baseline.evaluation,
        "eval_reconstruction_explicit",
        fake_eval,
    )

    def fake_save_metrics(recon, gt, **kwargs):
        observed["save_metrics"] = (recon, gt, kwargs)
        return {}

    monkeypatch.setattr(
        run_baseline.evaluation,
        "save_metrics_explicit",
        fake_save_metrics,
    )
    monkeypatch.setattr(
        run_baseline,
        "_save_reconstructions_legacy",
        lambda **kwargs: observed.setdefault("save_recons", kwargs),
    )

    with legacy_params_scope():
        params.cfg.clear()
        params.cfg.update(
            {
                "N": -999,
                "gridsize": -999,
                "label": "poison",
                "output_prefix": "poison",
                "intensity_scale": -999.0,
            }
        )
        params.seal()
        monkeypatch.setattr(
            params,
            "get",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("orchestration must not read params.cfg")
            ),
        )

        result = run_baseline.run_baseline(
            config,
            timestamp="07/28/2026, 12:34:56",
        )

        assert params.cfg == {
            "N": -999,
            "gridsize": -999,
            "label": "poison",
            "output_prefix": "poison",
            "intensity_scale": -999.0,
        }
        assert params._sealed is True

    expected_prefix = f"{config.output_dir}/07-28-2026-12.34.56_baseline_gs2/"
    assert observed["model_path"] == f"{expected_prefix}baseline_model.h5"
    assert observed["train"][1]["intensity_scale"] == 13.0
    np.testing.assert_array_equal(
        observed["alignment"]["scan_coords_yx"],
        np.squeeze(offsets)[:, [1, 0]],
    )
    assert observed["alignment"]["stitch_patch_size"] == 20
    assert observed["eval"][2] == 4
    metric_kwargs = observed["save_metrics"][2]
    assert metric_kwargs["label"] == "baseline_gs2"
    assert metric_kwargs["trim_offset"] == 4
    assert metric_kwargs["output_dir"] == expected_prefix
    snapshot = metric_kwargs["config_snapshot"]
    assert snapshot["N"] == 64
    assert snapshot["gridsize"] == 2
    assert snapshot["intensity_scale"] == 13.0
    assert snapshot["label"] == "baseline_gs2"
    assert snapshot["output_prefix"] == expected_prefix
    assert observed["save_recons"]["output_prefix"] == expected_prefix
    assert result["metrics"] == {"mae": (0.1, 0.2), "psnr": (20.0, 21.0)}


def test_train_leaf_projects_training_values_and_restores_params(
    monkeypatch,
    tmp_path: Path,
):
    from ptycho import baselines, params
    from ptycho.config.legacy_state import legacy_params_scope

    config = replace(
        _training_config(tmp_path),
        model=replace(_training_config(tmp_path).model, n_filters_scale=3),
    )
    observed = {}

    class FakeModel:
        def predict(self, value):
            observed["predict"] = value
            return "pred_i", "pred_phi"

    def fake_train(x, y_i, y_phi):
        observed["inputs"] = (x, y_i, y_phi)
        observed["params"] = dict(params.cfg)
        # W3.2: n_filters_scale flows through params.cfg (build_model reads it
        # at call time); the module-global projection protocol is retired.
        return FakeModel(), "history"

    monkeypatch.setattr(baselines, "train", fake_train)

    with legacy_params_scope():
        params.cfg.clear()
        params.cfg.update({"N": -1, "ambient": "keep"})
        params.seal()

        result = run_baseline._train_baseline_and_predict(
            "X",
            "YI",
            "Yphi",
            "Xtest",
            config=config,
            intensity_scale=23.0,
        )

        assert params.cfg == {"N": -1, "ambient": "keep"}
        assert params._sealed is True

    assert result[1:] == ("history", "pred_i", "pred_phi")
    assert observed["inputs"] == ("X", "YI", "Yphi")
    assert observed["predict"] == "Xtest"
    assert observed["params"]["N"] == 64
    assert observed["params"]["gridsize"] == 2
    assert observed["params"]["nepochs"] == 3
    assert observed["params"]["batch_size"] == 8
    assert observed["params"]["intensity_scale"] == 23.0
    assert observed["params"]["n_filters_scale"] == 3


def test_reassembly_leaf_uses_config_geometry_and_restores_ambient(
    monkeypatch,
    tmp_path: Path,
):
    from ptycho import params, tf_helper
    from ptycho.config.legacy_state import legacy_params_scope

    config = _training_config(tmp_path)
    offsets = np.zeros((2, 1, 2, 1), dtype=np.float64)
    observed = {}

    def fake_reassemble(patches, received_offsets, *, M):
        observed["patches"] = patches
        observed["offsets"] = received_offsets
        observed["M"] = M
        observed["params"] = dict(params.cfg)
        return tf.ones((6, 6, 1), dtype=tf.complex64)

    monkeypatch.setattr(tf_helper, "reassemble_position", fake_reassemble)

    with legacy_params_scope():
        params.cfg.clear()
        params.cfg.update({"N": -1, "ambient": "keep"})
        params.unseal()

        stitched = run_baseline._reassemble_predictions(
            tf.ones((2, 8, 8, 1), dtype=tf.float32),
            tf.zeros((2, 8, 8, 1), dtype=tf.float32),
            offsets,
            config=config,
            intensity_scale=5.0,
        )

        assert params.cfg == {"N": -1, "ambient": "keep"}
        assert params._sealed is False

    assert stitched.shape == (1, 6, 6, 1, 1)
    assert observed["M"] == 20
    assert observed["offsets"] is offsets
    assert observed["params"]["N"] == 64
    assert observed["params"]["gridsize"] == 2
    assert observed["params"]["intensity_scale"] == 5.0
