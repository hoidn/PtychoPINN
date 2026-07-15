"""Named CI profile + fail-closed contract coherence validation (Conformance D3).

Covers the resolution recorded in
docs/superpowers/plans/2026-07-14-ci-paper-conformance-audit.md (Theme 3):

1. ``resolve_ci_profile`` returns the paper's canonical PtychoPINN-CI bundle
   as a single named preset (finding 3.1) and fail-closes on contract-field
   contradictions instead of silently mixing profiles.
2. ``create_training_payload(profile='ci')`` resolves a coherent CI payload
   end-to-end; ``profile=None`` stays bit-identical to prior behavior.
3. Bare-default construction remains valid legacy behavior (design spec
   2026-07-09 §"Amplitude mode does not activate CI even when absent profile
   fields receive CI defaults") while explicitly half-configured CI intent
   (``rect_s1s2_init='data'`` without the rectangular forward) raises.
4. ``validate_contract_coherence`` is wired into the factory: rectangular +
   count_intensity + mae now fails at payload creation, not at training time.
"""

import numpy as np
import pytest

from ptycho_torch.config_factory import create_training_payload
from ptycho_torch.config_params import (
    DataConfig as PTDataConfig,
    ModelConfig as PTModelConfig,
    TrainingConfig as PTTrainingConfig,
)


CANONICAL_CI_BUNDLE = {
    "scale_contract_version": "ci_intensity_v2",
    "measurement_domain": "count_intensity",
    "physics_forward_mode": "rectangular_scaled",
    "torch_loss_mode": "poisson",
    "loss_function": "Poisson",
    "amplitude_physics_gain": 1.0,
    "rect_s1s2_trainable": True,
    "rect_s1s2_init": "data",
    "cnn_output_mode": "real_imag",
}

CONTRACT_FIELD_CONTRADICTIONS = {
    "scale_contract_version": "legacy_v1",
    "measurement_domain": "normalized_amplitude",
    "physics_forward_mode": "amplitude",
    "torch_loss_mode": "mae",
    "loss_function": "MAE",
}


@pytest.fixture
def tiny_train_npz(tmp_path):
    """Smallest DATA-001-compliant NPZ (N=64 square probe, 16 positions)."""
    N = 64
    n_images = 16
    npz_path = tmp_path / "tiny_train.npz"
    rng = np.random.default_rng(20260714)
    np.savez(
        npz_path,
        diffraction=rng.uniform(0.1, 1.0, size=(n_images, N, N)).astype(np.float32),
        probeGuess=np.ones((N, N), dtype=np.complex64),
        objectGuess=np.ones((2 * N, 2 * N), dtype=np.complex64),
        xcoords=np.linspace(0.0, 8.0, n_images),
        ycoords=np.linspace(0.0, 8.0, n_images),
        scan_index=np.arange(n_images, dtype=np.int32),
    )
    return npz_path


# ---------------------------------------------------------------------------
# resolve_ci_profile: canonical bundle + override precedence
# ---------------------------------------------------------------------------

def test_resolve_ci_profile_returns_exact_canonical_bundle():
    from ptycho_torch.config_factory import resolve_ci_profile

    assert resolve_ci_profile() == CANONICAL_CI_BUNDLE
    assert resolve_ci_profile(None) == CANONICAL_CI_BUNDLE


def test_resolve_ci_profile_passes_through_non_contract_overrides():
    from ptycho_torch.config_factory import resolve_ci_profile

    resolved = resolve_ci_profile(
        {"n_groups": 4, "batch_size": 8, "rect_s1s2_init": "ones"}
    )
    # User wins for non-contract fields (rect_s1s2_init is profile-defaulted
    # but not one of the five fail-closed contract fields).
    assert resolved["n_groups"] == 4
    assert resolved["batch_size"] == 8
    assert resolved["rect_s1s2_init"] == "ones"
    # Contract fields stay canonical.
    for field in CONTRACT_FIELD_CONTRADICTIONS:
        assert resolved[field] == CANONICAL_CI_BUNDLE[field]


def test_resolve_ci_profile_does_not_mutate_caller_overrides():
    from ptycho_torch.config_factory import resolve_ci_profile

    user = {"n_groups": 4}
    resolve_ci_profile(user)
    assert user == {"n_groups": 4}


@pytest.mark.parametrize(
    "field,bad_value", sorted(CONTRACT_FIELD_CONTRADICTIONS.items())
)
def test_resolve_ci_profile_rejects_contradicting_contract_override(field, bad_value):
    from ptycho_torch.config_factory import resolve_ci_profile

    with pytest.raises(ValueError) as excinfo:
        resolve_ci_profile({field: bad_value})
    message = str(excinfo.value)
    # Fail-closed: the error names both the required and the passed value.
    assert field in message
    assert repr(CANONICAL_CI_BUNDLE[field]) in message
    assert repr(bad_value) in message


def test_resolve_ci_profile_accepts_matching_contract_override():
    from ptycho_torch.config_factory import resolve_ci_profile

    resolved = resolve_ci_profile({"torch_loss_mode": "poisson"})
    assert resolved == CANONICAL_CI_BUNDLE


# ---------------------------------------------------------------------------
# create_training_payload(profile='ci')
# ---------------------------------------------------------------------------

def test_create_training_payload_ci_profile_resolves_coherent_payload(
    tiny_train_npz, tmp_path
):
    payload = create_training_payload(
        train_data_file=tiny_train_npz,
        output_dir=tmp_path / "out",
        overrides={"n_groups": 4, "gridsize": 1, "batch_size": 4},
        profile="ci",
    )
    assert payload.pt_data_config.scale_contract_version == "ci_intensity_v2"
    assert payload.pt_data_config.measurement_domain == "count_intensity"
    assert payload.pt_model_config.physics_forward_mode == "rectangular_scaled"
    assert payload.pt_model_config.rect_s1s2_trainable is True
    assert payload.pt_model_config.rect_s1s2_init == "data"
    assert payload.pt_model_config.cnn_output_mode == "real_imag"
    assert payload.pt_model_config.amplitude_physics_gain == 1.0
    assert payload.pt_model_config.loss_function == "Poisson"
    assert payload.pt_training_config.torch_loss_mode == "poisson"
    # The resolved payload must itself satisfy the coherence validator.
    from ptycho_torch.scaling_contract import validate_contract_coherence

    validate_contract_coherence(
        payload.pt_data_config,
        payload.pt_model_config,
        payload.pt_training_config,
    )


def test_create_training_payload_ci_profile_rejects_contradiction(
    tiny_train_npz, tmp_path
):
    with pytest.raises(ValueError, match="torch_loss_mode"):
        create_training_payload(
            train_data_file=tiny_train_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 4, "torch_loss_mode": "mae"},
            profile="ci",
        )


def test_create_training_payload_rejects_unknown_profile(tiny_train_npz, tmp_path):
    with pytest.raises(ValueError, match="profile"):
        create_training_payload(
            train_data_file=tiny_train_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 4},
            profile="paper",
        )


def test_profile_none_is_bit_identical_to_default(tiny_train_npz, tmp_path):
    overrides = {
        "n_groups": 4,
        "gridsize": 1,
        "batch_size": 8,
        "torch_loss_mode": "mae",
        "architecture": "cnn",
    }
    baseline = create_training_payload(
        train_data_file=tiny_train_npz,
        output_dir=tmp_path / "out",
        overrides=dict(overrides),
    )
    explicit_none = create_training_payload(
        train_data_file=tiny_train_npz,
        output_dir=tmp_path / "out",
        overrides=dict(overrides),
        profile=None,
    )
    assert baseline.pt_data_config == explicit_none.pt_data_config
    assert baseline.pt_model_config == explicit_none.pt_model_config
    assert baseline.pt_training_config == explicit_none.pt_training_config
    assert baseline.tf_training_config == explicit_none.tf_training_config
    assert baseline.overrides_applied == explicit_none.overrides_applied


# ---------------------------------------------------------------------------
# Coherence validation: legacy defaults stay valid; active contradictions fail
# ---------------------------------------------------------------------------

def test_bare_default_legacy_construction_remains_valid(tiny_train_npz, tmp_path):
    """Design-spec invariant: amplitude mode does not activate CI even though
    bare DataConfig defaults are CI-flavored (2026-07-09 design §profiles)."""
    payload = create_training_payload(
        train_data_file=tiny_train_npz,
        output_dir=tmp_path / "out",
        overrides={"n_groups": 4},
    )
    assert payload.pt_model_config.physics_forward_mode == "amplitude"
    assert payload.pt_data_config.scale_contract_version == "ci_intensity_v2"
    assert payload.pt_data_config.measurement_domain == "count_intensity"


def test_half_configured_ci_intent_via_overrides_raises(tiny_train_npz, tmp_path):
    """rect_s1s2_init='data' is a CI-only knob (docs/model_baselines.md); passing
    it without the rectangular forward is half-configured CI, not legacy."""
    with pytest.raises(ValueError, match=r"profile='ci'"):
        create_training_payload(
            train_data_file=tiny_train_npz,
            output_dir=tmp_path / "out",
            overrides={"n_groups": 4, "rect_s1s2_init": "data"},
        )


def test_count_intensity_mae_rectangular_raises_at_factory(
    tiny_train_npz, tmp_path
):
    with pytest.raises(ValueError, match="poisson"):
        create_training_payload(
            train_data_file=tiny_train_npz,
            output_dir=tmp_path / "out",
            overrides={
                "n_groups": 4,
                "physics_forward_mode": "rectangular_scaled",
                "torch_loss_mode": "mae",
            },
        )


def test_validate_contract_coherence_passes_coherent_legacy_and_ci():
    from ptycho_torch.scaling_contract import validate_contract_coherence

    # Coherent legacy: bare defaults (amplitude forward ignores CI fields).
    assert (
        validate_contract_coherence(
            PTDataConfig(), PTModelConfig(), PTTrainingConfig()
        )
        is None
    )
    # Coherent CI: rectangular forward + CI contract + poisson loss.
    assert (
        validate_contract_coherence(
            PTDataConfig(),
            PTModelConfig(physics_forward_mode="rectangular_scaled"),
            PTTrainingConfig(torch_loss_mode="poisson"),
        )
        is None
    )


def test_validate_contract_coherence_rejects_active_ci_with_mae():
    from ptycho_torch.scaling_contract import validate_contract_coherence

    with pytest.raises(ValueError, match="poisson"):
        validate_contract_coherence(
            PTDataConfig(),
            PTModelConfig(physics_forward_mode="rectangular_scaled"),
            PTTrainingConfig(torch_loss_mode="mae"),
        )


# ---------------------------------------------------------------------------
# Native CLI surface: --profile ci
# ---------------------------------------------------------------------------

def test_cli_profile_flag_forwards_to_factory(tmp_path, monkeypatch):
    from unittest.mock import MagicMock, patch

    train_file = tmp_path / "train.npz"
    train_file.touch()
    mock_factory = MagicMock()
    mock_factory.return_value = MagicMock()

    with patch(
        "ptycho_torch.config_factory.create_training_payload", mock_factory
    ):
        from ptycho_torch.train import cli_main

        monkeypatch.setattr(
            "sys.argv",
            [
                "train.py",
                "--train_data_file", str(train_file),
                "--output_dir", str(tmp_path / "outputs"),
                "--n_images", "4",
                "--max_epochs", "1",
                "--profile", "ci",
            ],
        )
        try:
            cli_main()
        except SystemExit:
            pass

    assert mock_factory.called, "Factory was not called"
    assert mock_factory.call_args.kwargs.get("profile") == "ci"


def test_workflow_forwards_torch_overrides_to_lightning(monkeypatch):
    from ptycho.config.config import ModelConfig, TrainingConfig
    from ptycho_torch.workflows import components

    execution = object()
    overrides = {"physics_forward_mode": "rectangular_scaled"}
    captured = {}

    monkeypatch.setattr(components.ptycho_config, "update_legacy_dict", lambda *a: None)

    def fake_train(train_data, test_data, config, execution_config=None, overrides=None):
        captured["execution_config"] = execution_config
        captured["overrides"] = overrides
        return {"models": {}}

    monkeypatch.setattr(components, "train_cdi_model_torch", fake_train)

    components.run_cdi_example_torch(
        object(),
        None,
        TrainingConfig(model=ModelConfig(), output_dir=""),
        execution_config=execution,
        overrides=overrides,
    )

    assert captured["execution_config"] is execution
    assert captured["overrides"] is overrides


def test_cli_profile_reaches_training_execution(tiny_train_npz, tmp_path, monkeypatch):
    from ptycho_torch.train import cli_main
    from ptycho_torch.workflows import components

    captured = {}
    monkeypatch.setattr(
        "ptycho.raw_data.RawData.from_file",
        lambda path: object(),
    )

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return None, None, {"models": {}}

    monkeypatch.setattr(components, "run_cdi_example_torch", fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "train.py",
            "--train_data_file", str(tiny_train_npz),
            "--output_dir", str(tmp_path / "outputs"),
            "--n_images", "4",
            "--max_epochs", "1",
            "--profile", "ci",
        ],
    )

    cli_main()

    forwarded = captured["overrides"]
    assert captured["execution_config"] is not None
    assert forwarded["physics_forward_mode"] == "rectangular_scaled"
    assert forwarded["scale_contract_version"] == "ci_intensity_v2"
    assert forwarded["measurement_domain"] == "count_intensity"
    assert forwarded["torch_loss_mode"] == "poisson"
    assert forwarded["rect_s1s2_init"] == "data"
