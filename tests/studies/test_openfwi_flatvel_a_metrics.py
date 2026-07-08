import pytest
import torch


def test_mae_rmse_are_hand_computed():
    from scripts.studies.openfwi_flatvel_a.metrics import metric_payload

    pred = torch.tensor([[[[1.0, 3.0]]]])
    target = torch.tensor([[[[2.0, 1.0]]]])
    payload = metric_payload([pred], [target], normalized=False, target_stats=None)

    assert payload["MAE"] == pytest.approx(1.5)
    assert payload["RMSE"] == pytest.approx((2.5) ** 0.5)


def test_identity_ssim_is_one():
    from scripts.studies.openfwi_flatvel_a.metrics import metric_payload

    image = torch.ones(1, 1, 70, 70)
    payload = metric_payload([image], [image], normalized=False, target_stats=None)

    assert payload["SSIM"] == pytest.approx(1.0)


def test_denormalized_metric_path_uses_target_stats():
    from scripts.studies.openfwi_flatvel_a.metrics import metric_payload

    pred = torch.zeros(1, 1, 70, 70)
    target = torch.ones(1, 1, 70, 70)
    stats = {"target": {"mean": 10.0, "std": 2.0}}
    payload = metric_payload([pred], [target], normalized=True, target_stats=stats)

    assert payload["MAE"] == pytest.approx(2.0)
    assert payload["metric_units"] == "denormalized_velocity"


def test_metric_payload_rejects_mismatched_shapes():
    from scripts.studies.openfwi_flatvel_a.metrics import metric_payload

    with pytest.raises(ValueError, match="shape"):
        metric_payload(
            [torch.zeros(1, 1, 70, 70)],
            [torch.zeros(1, 1, 69, 70)],
            normalized=False,
            target_stats=None,
        )
