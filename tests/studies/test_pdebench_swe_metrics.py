import json

import torch


def test_err_rmse_and_nrmse_match_hand_calculation():
    from scripts.studies.pdebench_swe.metrics import err_nrmse, err_rmse

    prediction = torch.tensor([[[[2.0, 4.0]]]])
    target = torch.tensor([[[[1.0, 2.0]]]])

    assert torch.isclose(err_rmse(prediction, target), torch.sqrt(torch.tensor(2.5)))
    expected_nrmse = torch.sqrt(torch.tensor(5.0 / 5.0))
    assert torch.isclose(err_nrmse(prediction, target), expected_nrmse)


def test_metric_payload_emits_aggregate_and_per_channel_values():
    from scripts.studies.pdebench_swe.metrics import metric_payload

    prediction = [torch.zeros(2, 2, 2, 2)]
    target = [torch.ones(2, 2, 2, 2)]

    payload = metric_payload(prediction, target, normalized=False, stats=None)

    assert payload["err_RMSE"] == 1.0
    assert payload["err_nRMSE"] == 1.0
    assert payload["per_channel"]["err_RMSE"] == [1.0, 1.0]
    assert payload["per_channel"]["err_nRMSE"] == [1.0, 1.0]
    assert payload["num_eval_batches"] == 1
    assert payload["num_eval_pairs"] == 2
    json.dumps(payload)


def test_normalization_round_trip_and_train_only_stats():
    from scripts.studies.pdebench_swe.metrics import (
        compute_channel_stats,
        denormalize_batch,
        normalize_batch,
    )

    class ToyDataset:
        def __init__(self):
            self.items = [
                {"input": torch.tensor([[[1.0, 3.0]]])},
                {"input": torch.tensor([[[5.0, 7.0]]])},
                {"input": torch.tensor([[[100.0, 100.0]]])},
            ]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            return self.items[index]

    stats = compute_channel_stats(ToyDataset(), max_batches=2)
    assert stats["num_samples"] == 2
    assert stats["mean"] == [4.0]
    assert stats["std"][0] > 0

    batch = torch.tensor([[[[1.0, 3.0]]]])
    restored = denormalize_batch(normalize_batch(batch, stats), stats)
    assert torch.allclose(restored, batch)


def test_metric_payload_denormalizes_when_stats_are_available():
    from scripts.studies.pdebench_swe.metrics import metric_payload

    stats = {"mean": [10.0], "std": [2.0], "num_samples": 1}
    prediction = [torch.zeros(1, 1, 1, 1)]
    target = [torch.ones(1, 1, 1, 1)]

    payload = metric_payload(prediction, target, normalized=True, stats=stats)

    assert payload["metric_units"] == "denormalized_physical_units"
    assert payload["err_RMSE"] == 2.0
