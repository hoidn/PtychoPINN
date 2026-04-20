import pytest


def test_default_profile_ids_match_plan():
    from scripts.studies.openfwi_flatvel_a.run_config import (
        DEFAULT_RUN_BUDGET,
        OPTIONAL_PROFILE_IDS,
        PRIMARY_PROFILE_IDS,
    )

    assert PRIMARY_PROFILE_IDS == ["hybrid_resnet_smoke", "unet_smoke"]
    assert OPTIONAL_PROFILE_IDS == ["fno_smoke", "official_inversionnet_probe"]
    assert DEFAULT_RUN_BUDGET["split_seed"] == 20260420


def test_parse_profile_ids_accepts_csv_and_lists():
    from scripts.studies.openfwi_flatvel_a.run_config import parse_profile_ids

    assert parse_profile_ids("hybrid_resnet_smoke,unet_smoke") == [
        "hybrid_resnet_smoke",
        "unet_smoke",
    ]
    assert parse_profile_ids(["hybrid_resnet_smoke"]) == ["hybrid_resnet_smoke"]


def test_unknown_profile_is_rejected():
    from scripts.studies.openfwi_flatvel_a.run_config import get_model_profile

    with pytest.raises(ValueError, match="unknown OpenFWI FlatVel-A profile_id"):
        get_model_profile("not-a-profile")


def test_validate_run_budget_coerces_and_checks_positive_values():
    from scripts.studies.openfwi_flatvel_a.run_config import validate_run_budget

    budget = validate_run_budget(
        {
            "epochs": "1",
            "batch_size": "4",
            "learning_rate": "0.001",
            "train_samples": "32",
            "val_samples": "16",
            "test_samples": "16",
            "split_seed": "20260420",
            "device": "cpu",
            "num_workers": "0",
        }
    )

    assert budget["epochs"] == 1
    assert budget["learning_rate"] == pytest.approx(0.001)
    assert budget["profiles"] == ["hybrid_resnet_smoke", "unet_smoke"]
