import torch


def _metadata() -> dict:
    return {
        "task_id": "2d_cfd_cns",
        "field_order": ["density", "Vx", "Vy", "pressure"],
        "history_len": 2,
        "dx": 0.5,
        "dy": 0.25,
        "dt": 0.1,
        "boundary_condition": "periodic",
        "eta": 0.01,
        "zeta": 0.02,
    }


def _state_stats() -> dict:
    return {
        "mean": [0.0, 0.0, 0.0, 0.0],
        "std": [1.0, 1.0, 1.0, 1.0],
        "field_order": ["density", "Vx", "Vy", "pressure"],
    }


def _config(**overrides):
    from scripts.studies.pdebench_image128.physics_losses import PhysicsRegularizationConfig

    payload = {
        "enabled": True,
        "positivity_weight": 0.0,
        "continuity_weight": 0.0,
        "global_mass_weight": 0.0,
    }
    payload.update(overrides)
    return PhysicsRegularizationConfig(**payload)


def test_periodic_central_difference_of_constant_field_is_zero():
    from scripts.studies.pdebench_image128.physics_losses import periodic_central_difference

    field = torch.full((2, 3, 4), 7.0)

    dx = periodic_central_difference(field, spacing=0.5, dim=-1)
    dy = periodic_central_difference(field, spacing=0.25, dim=-2)

    assert torch.allclose(dx, torch.zeros_like(dx))
    assert torch.allclose(dy, torch.zeros_like(dy))


def test_cns_positivity_loss_is_zero_for_positive_density_and_pressure():
    from scripts.studies.pdebench_image128.physics_losses import build_physics_regularizer

    regularizer = build_physics_regularizer(
        task_id="2d_cfd_cns",
        metadata=_metadata(),
        state_stats=_state_stats(),
        config=_config(positivity_weight=1.0),
    )

    x_norm = torch.zeros((1, 8, 2, 2))
    pred_norm = torch.zeros((1, 4, 2, 2))
    pred_norm[:, 0] = 1.0
    pred_norm[:, 3] = 2.0
    target_norm = pred_norm.clone()

    result = regularizer.compute(x_norm=x_norm, pred_norm=pred_norm, target_norm=target_norm)

    assert torch.isclose(result.total, torch.tensor(0.0))
    assert torch.isclose(result.terms["positivity"], torch.tensor(0.0))


def test_cns_positivity_loss_penalizes_negative_density_and_pressure():
    from scripts.studies.pdebench_image128.physics_losses import build_physics_regularizer

    regularizer = build_physics_regularizer(
        task_id="2d_cfd_cns",
        metadata=_metadata(),
        state_stats=_state_stats(),
        config=_config(positivity_weight=1.0),
    )

    x_norm = torch.zeros((1, 8, 2, 2))
    pred_norm = torch.zeros((1, 4, 2, 2))
    pred_norm[:, 0, 0, 0] = -2.0
    pred_norm[:, 3, 1, 1] = -3.0
    target_norm = torch.zeros((1, 4, 2, 2))

    result = regularizer.compute(x_norm=x_norm, pred_norm=pred_norm, target_norm=target_norm)

    assert result.total > 0
    assert result.terms["positivity"] > 0


def test_cns_continuity_loss_is_zero_for_constant_stationary_state():
    from scripts.studies.pdebench_image128.physics_losses import build_physics_regularizer

    regularizer = build_physics_regularizer(
        task_id="2d_cfd_cns",
        metadata=_metadata(),
        state_stats=_state_stats(),
        config=_config(continuity_weight=1.0),
    )

    previous = torch.zeros((1, 4, 3, 3))
    previous[:, 0] = 2.0
    previous[:, 1] = 0.5
    previous[:, 2] = -0.25
    previous[:, 3] = 1.0

    x_norm = torch.cat([torch.zeros_like(previous), previous], dim=1)
    pred_norm = previous.clone()
    target_norm = previous.clone()

    result = regularizer.compute(x_norm=x_norm, pred_norm=pred_norm, target_norm=target_norm)

    assert torch.allclose(result.total, torch.tensor(0.0), atol=1e-7)
    assert torch.allclose(result.terms["continuity"], torch.tensor(0.0), atol=1e-7)


def test_cns_global_mass_loss_detects_density_drift():
    from scripts.studies.pdebench_image128.physics_losses import build_physics_regularizer

    regularizer = build_physics_regularizer(
        task_id="2d_cfd_cns",
        metadata=_metadata(),
        state_stats=_state_stats(),
        config=_config(global_mass_weight=1.0),
    )

    previous = torch.zeros((1, 4, 2, 2))
    previous[:, 0] = 1.0
    previous[:, 3] = 1.0
    predicted = previous.clone()
    predicted[:, 0] = 2.0
    x_norm = torch.cat([torch.zeros_like(previous), previous], dim=1)
    target_norm = previous.clone()

    result = regularizer.compute(x_norm=x_norm, pred_norm=predicted, target_norm=target_norm)

    assert result.total > 0
    assert result.terms["global_mass"] > 0


def test_build_physics_regularizer_rejects_unsupported_task():
    from scripts.studies.pdebench_image128.physics_losses import build_physics_regularizer

    try:
        build_physics_regularizer(
            task_id="darcy",
            metadata=_metadata(),
            state_stats=_state_stats(),
            config=_config(positivity_weight=1.0),
        )
    except ValueError as exc:
        assert "not supported" in str(exc)
    else:
        raise AssertionError("unsupported task must fail closed")
