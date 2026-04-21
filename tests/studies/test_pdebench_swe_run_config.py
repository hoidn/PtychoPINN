import json

import pytest


def test_builtin_profile_registry_contains_primary_and_ablation_profiles():
    from scripts.studies.pdebench_swe.run_config import (
        BUILTIN_PROFILE_IDS,
        get_model_profile,
    )

    assert BUILTIN_PROFILE_IDS == [
        "hybrid_resnet_base",
        "fno_base",
        "unet_base",
        "hybrid_resnet_spectral_reduced",
        "hybrid_resnet_local_reduced",
    ]
    hybrid = get_model_profile("hybrid_resnet_base")
    spectral = get_model_profile("hybrid_resnet_spectral_reduced")
    local = get_model_profile("hybrid_resnet_local_reduced")

    assert hybrid.profile_id == "hybrid_resnet_base"
    assert hybrid.base_model == "hybrid_resnet"
    assert hybrid.hidden_channels == 32
    assert hybrid.fno_modes == 12
    assert hybrid.fno_blocks == 4
    assert hybrid.hybrid_downsample_steps == 2
    assert hybrid.hybrid_resnet_blocks == 6
    assert spectral.fno_modes == 6
    assert local.hybrid_resnet_blocks == 2


def test_budget_validation_accepts_plan_budget_and_rejects_missing_baseline():
    from scripts.studies.pdebench_swe.run_config import validate_run_budget

    budget = {
        "schema_version": "pdebench_swe_run_budget_v1",
        "budget_id": "target",
        "epochs": 15,
        "batch_size": 16,
        "learning_rate": 2e-4,
        "scheduler": "ReduceLROnPlateau",
        "plateau_factor": 0.5,
        "plateau_patience": 2,
        "plateau_min_lr": 1e-5,
        "plateau_threshold": 0.0,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "beta1": 0.9,
        "beta2": 0.999,
        "max_train_trajectories": 800,
        "max_val_trajectories": 100,
        "max_test_trajectories": 100,
        "max_pairs_per_trajectory": 10,
        "normalization_max_samples": 8000,
        "eval_splits": ["val", "test"],
        "num_workers": 2,
        "device": "cuda",
        "training_seed": 20260420,
        "primary_profiles": ["hybrid_resnet_base", "fno_base", "unet_base"],
        "ablation_profiles": ["hybrid_resnet_spectral_reduced", "hybrid_resnet_local_reduced"],
    }

    normalized = validate_run_budget(budget)
    json.dumps(normalized)
    assert normalized["primary_profiles"] == ["hybrid_resnet_base", "fno_base", "unet_base"]
    assert normalized["training_seed"] == 20260420
    assert normalized["scheduler"] == "ReduceLROnPlateau"
    assert normalized["plateau_min_lr"] == pytest.approx(1e-5)

    budget_without_floor = dict(budget)
    budget_without_floor.pop("plateau_min_lr")
    defaulted = validate_run_budget(budget_without_floor)
    assert defaulted["plateau_min_lr"] == pytest.approx(1e-5)

    budget_too_high_floor = dict(budget)
    budget_too_high_floor["plateau_min_lr"] = 1e-4
    with pytest.raises(ValueError, match="plateau_min_lr"):
        validate_run_budget(budget_too_high_floor)

    budget["primary_profiles"] = ["hybrid_resnet_base", "fno_base"]
    with pytest.raises(ValueError, match="unet_base"):
        validate_run_budget(budget)

    budget["primary_profiles"] = ["hybrid_resnet_base", "fno_base", "unet_base"]
    budget.pop("training_seed")
    with pytest.raises(ValueError, match="training_seed"):
        validate_run_budget(budget)
