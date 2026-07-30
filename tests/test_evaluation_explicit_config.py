import dill
import numpy as np

from ptycho import evaluation, params


def _metrics():
    return {
        "mae": (0.1, 0.2),
        "mse": (0.01, 0.02),
        "psnr": (30.0, 20.0),
        "ssim": (0.9, 0.8),
        "ms_ssim": (0.91, 0.81),
        "frc50": (12.0, 10.0),
        "frc1over7": (18.0, 16.0),
        "frc": ([1.0, 0.4], [1.0, 0.3]),
    }


def test_save_metrics_explicit_uses_injected_snapshot_and_filenames(
    monkeypatch,
    tmp_path,
):
    poisoned_cfg = {"offset": 98, "poison": "must-remain-untouched"}
    monkeypatch.setattr(params, "cfg", poisoned_cfg)
    calls = []

    def fake_eval(stitched, ground_truth, *, label, trim_offset):
        calls.append(
            {
                "stitched_shape": stitched.shape,
                "ground_truth_shape": ground_truth.shape,
                "label": label,
                "trim_offset": trim_offset,
            }
        )
        return _metrics()

    monkeypatch.setattr(evaluation, "eval_reconstruction_explicit", fake_eval)
    snapshot = {"N": 64, "label": "snapshot-label"}
    output_dir = tmp_path / "metrics"
    stitched = np.ones((1, 8, 8, 1), dtype=np.complex64)
    ground_truth = np.ones((8, 8, 1), dtype=np.complex64)

    result = evaluation.save_metrics_explicit(
        stitched,
        ground_truth,
        label="row-a",
        trim_offset=4,
        output_dir=output_dir,
        config_snapshot=snapshot,
    )

    assert calls == [
        {
            "stitched_shape": (1, 8, 8, 1),
            "ground_truth_shape": (8, 8, 1),
            "label": "row-a",
            "trim_offset": 4,
        }
    ]
    assert (output_dir / "params.dill").is_file()
    assert (output_dir / "metrics.csv").is_file()
    with (output_dir / "params.dill").open("rb") as handle:
        persisted = dill.load(handle)
    assert persisted["N"] == 64
    assert persisted["label"] == "row-a"
    assert persisted["mae"] == (0.1, 0.2)
    assert snapshot == {"N": 64, "label": "snapshot-label"}
    assert params.cfg == poisoned_cfg
    assert result == {
        key: _metrics()[key]
        for key in ("mae", "mse", "psnr", "frc50", "frc1over7", "frc")
    }
