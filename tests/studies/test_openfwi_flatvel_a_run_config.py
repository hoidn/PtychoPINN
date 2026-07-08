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


def test_default_budget_inherits_grid_lines_hybrid_resnet_training_recipe():
    from scripts.studies.openfwi_flatvel_a.run_config import DEFAULT_RUN_BUDGET

    assert DEFAULT_RUN_BUDGET["epochs"] == 5
    assert DEFAULT_RUN_BUDGET["batch_size"] == 16
    assert DEFAULT_RUN_BUDGET["learning_rate"] == pytest.approx(2e-4)
    assert DEFAULT_RUN_BUDGET["scheduler"] == "ReduceLROnPlateau"
    assert DEFAULT_RUN_BUDGET["plateau_factor"] == pytest.approx(0.5)
    assert DEFAULT_RUN_BUDGET["plateau_patience"] == 2
    assert DEFAULT_RUN_BUDGET["plateau_min_lr"] == pytest.approx(1e-5)
    assert DEFAULT_RUN_BUDGET["plateau_threshold"] == pytest.approx(0.0)
    assert DEFAULT_RUN_BUDGET["optimizer"] == "adam"
    assert DEFAULT_RUN_BUDGET["weight_decay"] == pytest.approx(0.0)
    assert DEFAULT_RUN_BUDGET["beta1"] == pytest.approx(0.9)
    assert DEFAULT_RUN_BUDGET["beta2"] == pytest.approx(0.999)


def test_hybrid_profile_inherits_grid_lines_hybrid_resnet_architecture_recipe():
    from scripts.studies.openfwi_flatvel_a.run_config import get_model_profile

    hybrid = get_model_profile("hybrid_resnet_smoke")

    assert hybrid.hidden_channels == 32
    assert hybrid.fno_modes == 12
    assert hybrid.fno_blocks == 4
    assert hybrid.hybrid_downsample_steps == 2
    assert hybrid.hybrid_resnet_blocks == 6


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
            "scheduler": "ReduceLROnPlateau",
            "plateau_factor": "0.5",
            "plateau_patience": "2",
            "plateau_min_lr": "0.00001",
            "plateau_threshold": "0.0",
            "optimizer": "adam",
            "weight_decay": "0.0",
            "beta1": "0.9",
            "beta2": "0.999",
        }
    )

    assert budget["epochs"] == 1
    assert budget["learning_rate"] == pytest.approx(0.001)
    assert budget["scheduler"] == "ReduceLROnPlateau"
    assert budget["plateau_min_lr"] == pytest.approx(0.00001)

    with pytest.raises(ValueError, match="plateau_min_lr"):
        validate_run_budget({**budget, "plateau_min_lr": "0.0001"})
    assert budget["profiles"] == ["hybrid_resnet_smoke", "unet_smoke"]
