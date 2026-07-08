import h5py
import numpy as np
import torch


def _write_stats_file(path):
    with h5py.File(path, "w") as handle:
        handle.create_dataset(
            "nu",
            data=np.array(
                [
                    [[1, 1], [1, 1]],
                    [[3, 3], [3, 3]],
                    [[100, 100], [100, 100]],
                ],
                dtype=np.float32,
            ),
        )
        handle.create_dataset(
            "tensor",
            data=np.array(
                [
                    [[[10, 10], [10, 10]]],
                    [[[14, 14], [14, 14]]],
                    [[[1000, 1000], [1000, 1000]]],
                ],
                dtype=np.float32,
            ),
        )


def test_static_operator_stats_are_separate_and_train_only(tmp_path):
    from scripts.studies.pdebench_image128.normalization import compute_static_operator_stats

    data_file = tmp_path / "stats.hdf5"
    _write_stats_file(data_file)

    input_stats, target_stats = compute_static_operator_stats(
        data_file=data_file,
        input_dataset="nu",
        target_dataset="tensor",
        train_indices=[0, 1],
    )

    assert input_stats["mean"] == [2.0]
    assert target_stats["mean"] == [12.0]
    assert input_stats["std"] == [1.0]
    assert target_stats["std"] == [2.0]
    assert input_stats["num_samples"] == 2
    assert target_stats["num_samples"] == 2
    assert input_stats["source"] == "train_split_inputs_only"
    assert target_stats["source"] == "train_split_targets_only"


def test_static_operator_metrics_denormalize_predictions_with_target_stats():
    from scripts.studies.pdebench_image128.metrics import static_operator_metric_payload

    target_stats = {"mean": [10.0], "std": [2.0]}
    predictions_normalized = [torch.tensor([[[[1.0, 2.0]]]])]
    targets_normalized = [torch.tensor([[[[0.0, 2.0]]]])]

    payload = static_operator_metric_payload(
        predictions_normalized,
        targets_normalized,
        normalized=True,
        target_stats=target_stats,
    )

    assert payload["metric_units"] == "denormalized_target_units"
    assert payload["err_RMSE"] == 1.4142135381698608
    assert payload["err_nRMSE"] == 0.11624763906002045
    assert payload["horizon"] == "static_operator"


def test_relative_l2_formula_matches_hand_computation():
    from scripts.studies.pdebench_image128.metrics import err_nrmse

    prediction = torch.tensor([[[[2.0, 0.0], [0.0, 0.0]]]])
    target = torch.tensor([[[[1.0, 2.0], [2.0, 1.0]]]])

    observed = err_nrmse(prediction, target)

    assert torch.isclose(observed, torch.tensor(1.0))
