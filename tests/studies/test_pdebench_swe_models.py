import pytest
import torch


@pytest.mark.parametrize("model_name", ["hybrid_resnet", "unet", "fno"])
def test_build_model_preserves_channel_first_forward_contract(model_name):
    from scripts.studies.pdebench_swe.models import ModelBuildBlocker, build_model

    try:
        model = build_model(
            model_name,
            in_channels=2,
            out_channels=2,
            spatial_shape=(8, 8),
            smoke_config={"hidden_channels": 4, "fno_modes": 4, "fno_blocks": 3},
        )
    except ModelBuildBlocker as exc:
        assert model_name == "fno"
        assert exc.to_payload(run_id="model-test")["reason"] == "model_dependency_unavailable"
        return

    x = torch.randn(2, 2, 8, 8)
    y = model(x)
    assert y.shape == x.shape
    loss = y.square().mean()
    loss.backward()
    assert any(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_model_description_reports_parameter_count():
    from scripts.studies.pdebench_swe.models import build_model, describe_model

    model = build_model(
        "unet",
        in_channels=1,
        out_channels=1,
        spatial_shape=(8, 8),
        smoke_config={"hidden_channels": 4},
    )

    description = describe_model(model, model_name="unet", smoke_config={"hidden_channels": 4})

    assert description["model_name"] == "unet"
    assert description["parameter_count"] > 0
    assert description["smoke_config"]["hidden_channels"] == 4


def test_fno_dependency_failure_is_a_typed_blocker(monkeypatch):
    from scripts.studies.pdebench_swe import models
    from scripts.studies.pdebench_swe.models import ModelBuildBlocker

    def _raise_import_error():
        raise ImportError("no neuralop")

    monkeypatch.setattr(models, "_import_neuralop_fno", _raise_import_error)

    with pytest.raises(ModelBuildBlocker) as exc_info:
        models.build_model(
            "fno",
            in_channels=1,
            out_channels=1,
            spatial_shape=(8, 8),
            smoke_config={},
        )

    payload = exc_info.value.to_payload(run_id="fno-blocked")
    assert payload["model"] == "fno"
    assert payload["run_id"] == "fno-blocked"
    assert payload["reason"] == "model_dependency_unavailable"


def test_unknown_model_name_fails_clearly():
    from scripts.studies.pdebench_swe.models import build_model

    with pytest.raises(ValueError, match="unknown SWE smoke model"):
        build_model(
            "not-a-model",
            in_channels=1,
            out_channels=1,
            spatial_shape=(8, 8),
            smoke_config={},
        )
