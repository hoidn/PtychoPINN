import torch


def test_image128_model_profiles_build_and_record_parameter_counts():
    from scripts.studies.pdebench_image128.models import build_model_from_profile, describe_model
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    for profile_id in ["hybrid_resnet_base", "fno_base", "unet_strong", "unet_tiny_smoke"]:
        profile = get_model_profile(profile_id)
        model = build_model_from_profile(
            profile,
            in_channels=1,
            out_channels=1,
            spatial_shape=(128, 128),
        )
        y = model(torch.zeros(1, 1, 128, 128))
        description = describe_model(model, profile=profile)

        assert tuple(y.shape) == (1, 1, 128, 128)
        assert description["profile_id"] == profile_id
        assert description["parameter_count"] > 0


def test_unet_tiny_smoke_is_not_a_strong_baseline():
    from scripts.studies.pdebench_image128.models import assert_strong_baseline_profile
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    tiny = get_model_profile("unet_tiny_smoke")
    strong = get_model_profile("unet_strong")

    assert_strong_baseline_profile(strong)
    try:
        assert_strong_baseline_profile(tiny)
    except ValueError as exc:
        assert "readiness-only" in str(exc)
    else:
        raise AssertionError("unet_tiny_smoke must not satisfy the strong-baseline gate")


def test_missing_fno_dependency_is_reported_as_blocker(monkeypatch):
    from scripts.studies.pdebench_image128 import models
    from scripts.studies.pdebench_image128.models import ModelBuildBlocker
    from scripts.studies.pdebench_image128.run_config import get_model_profile

    def fail_import():
        raise ImportError("missing neuralop")

    monkeypatch.setattr(models, "_import_neuralop_fno", fail_import)
    profile = get_model_profile("fno_base")

    try:
        models.build_model_from_profile(profile, in_channels=1, out_channels=1, spatial_shape=(128, 128))
    except ModelBuildBlocker as exc:
        payload = exc.to_payload()
        assert payload["model"] == "fno_base"
        assert payload["reason"] == "model_dependency_unavailable"
        assert "neuralop" in payload["message"]
    else:
        raise AssertionError("missing neuralop should block FNO explicitly")
