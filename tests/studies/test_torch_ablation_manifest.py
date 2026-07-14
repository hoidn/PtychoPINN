from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType

import pytest

from scripts.studies.ablation import manifest
from scripts.studies.ablation.configuration import (
    ConfigResolutionError,
    resolve_torch_configs,
)


CI_COMPATIBILITY_SPEC = Path(
    "scripts/studies/specs/hybrid_resnet_ci_compatibility.toml"
)
EXPERIMENTAL_NPZ_EXAMPLE = Path(
    "scripts/studies/specs/examples/experimental_npz_dataset.toml"
)

GRID_LINES_CONVERGENCE_SPEC = Path(
    "scripts/studies/specs/grid_lines_ci_convergence.toml"
)
TASK30_LEGACY_PHYSICAL_GAIN = 12.452229360013307


def _exact_bridge_evidence_payload(requirement) -> dict[str, object]:
    return {
        "schema_version": "hybrid_resnet_integration_bridge_evidence_v3",
        "contract": requirement.to_mapping(),
        "checkpoint_sha256": "1" * 64,
        "selected_checkpoint": "artifacts/checkpoints/best.ckpt",
        "train_npz_sha256": "5" * 64,
        "test_npz_sha256": "6" * 64,
        "pre_stitch_patch_sha256": "2" * 64,
        "historical_canvas_sha256": "3" * 64,
        "ground_truth_sha256": "7" * 64,
        "generic_canvas_sha256": "3" * 64,
        "historical_mask_sha256": "4" * 64,
        "generic_mask_sha256": "4" * 64,
        "canvases_equivalent": True,
        "masks_equivalent": True,
        "no_resize_asserted": True,
        "gauge_handling": "declared_none",
        "recorded_differences": [],
        "fixture_amp_mae": 0.08,
        "fixture_phase_mae": 0.12,
        "fixture_amp_ssim": 0.88,
        "fixture_phase_ssim": 0.96,
        "architecture": "hybrid_resnet",
        "generator_output_mode": "real_imag",
        "hybrid_encoder_conv_hidden_scale": 2.0,
        "training_patch_weighting": "central_mask",
        "physics_forward_mode": "amplitude",
        "amplitude_physics_gain": 16.0,
        "torch_loss_mode": "mae",
        "seed": 3,
        "epochs": 5,
    }


def test_grid_lines_ci_convergence_expands_exact_checked_six_arm_protocol() -> None:
    checked = manifest.load_manifest(GRID_LINES_CONVERGENCE_SPEC)
    runs = manifest.expand_runs(checked)

    assert checked.schema_version == 1
    assert checked.study_id == "grid-lines-ci-convergence"
    assert checked.seeds == (3,)
    assert checked.diagnostics.milestones == (5, 20, 40, 80)
    assert checked.output_root == (
        ".artifacts/ci_compatibility/task30/corrected_lines_seed3"
    )
    assert len(runs) == 6
    assert {
        (run.dimensions["architecture"], run.dimensions["physics_profile"])
        for run in runs
    } == {
        (architecture, profile)
        for architecture in ("hybrid_resnet", "cnn")
        for profile in ("legacy_mae", "legacy_nll", "ci_nll")
    }
    for run in runs:
        overrides = run.overrides
        assert overrides["data.N"] == 64
        assert overrides["data.C"] == 4
        assert overrides["training.epochs"] == 80
        assert overrides["model.architecture"] == run.dimensions["architecture"]
        assert "dictionary_parity" not in overrides.values()
        profile = run.dimensions["physics_profile"]
        if profile == "ci_nll":
            assert overrides["model.amplitude_physics_gain"] == 1.0
            assert run.dataset_id == "lines_ci_3p5m"
            assert overrides["data.scale_contract_version"] == "ci_intensity_v2"
            assert overrides["data.measurement_domain"] == "count_intensity"
            assert overrides["model.physics_forward_mode"] == "rectangular_scaled"
            assert overrides["model.rect_s1s2_trainable"] is True
            assert overrides["model.loss_function"] == "Poisson"
            assert overrides["training.torch_loss_mode"] == "poisson"
            assert overrides["inference.varpro_scaling"] is True
        else:
            assert overrides["model.amplitude_physics_gain"] == pytest.approx(
                TASK30_LEGACY_PHYSICAL_GAIN, rel=0.0, abs=0.0
            )
            assert run.dataset_id == "lines_legacy_amp"
            assert overrides["data.normalize"] == "Batch"
            assert overrides["data.scale_contract_version"] == "legacy_v1"
            assert overrides["data.measurement_domain"] == "normalized_amplitude"
            assert overrides["model.physics_forward_mode"] == "amplitude"
            assert overrides["model.rect_s1s2_trainable"] is False
            assert overrides["inference.varpro_scaling"] is False
            expected_loss = "MAE" if profile == "legacy_mae" else "Poisson"
            expected_mode = "mae" if profile == "legacy_mae" else "poisson"
            assert overrides["model.loss_function"] == expected_loss
            assert overrides["training.torch_loss_mode"] == expected_mode
        if run.dimensions["architecture"] == "cnn":
            assert overrides["model.decoder_last_amp_channels"] == 4
            assert overrides["model.probe_big"] is True


def test_grid_lines_ci_convergence_pins_exact_lines_dataset_bytes() -> None:
    checked = manifest.load_manifest(GRID_LINES_CONVERGENCE_SPEC)
    datasets = {dataset.id: dataset for dataset in checked.datasets}

    assert set(datasets) == {
        "deadleaves_ci_3p5m",
        "deadleaves_legacy_amp",
        "lines_ci_3p5m",
        "lines_legacy_amp",
    }
    canonical = {
        dataset.id: dataset
        for dataset in manifest.load_manifest(CI_COMPATIBILITY_SPEC).datasets
    }
    for dataset_id in (
        "deadleaves_ci_3p5m",
        "deadleaves_legacy_amp",
        "lines_ci_3p5m",
        "lines_legacy_amp",
    ):
        assert datasets[dataset_id].metadata == canonical[dataset_id].metadata
    ci = datasets["lines_ci_3p5m"]
    legacy = datasets["lines_legacy_amp"]
    assert (ci.metadata["train"], ci.metadata["train_sha256"]) == (
        ".artifacts/ci_compatibility/datasets_v3/lines_ci_3p5m_train.npz",
        "316c4ac841eb45184fd84f5430a08be6f80ef919963d9aeaa2406d5d1a0e239b",
    )
    assert (ci.metadata["test"], ci.metadata["test_sha256"]) == (
        ".artifacts/ci_compatibility/datasets_v3/lines_ci_3p5m_test.npz",
        "b120bd1364053969fdaf8b1dc7d6651a54c097f52116e1704579c81909b6c747",
    )
    assert (legacy.metadata["train"], legacy.metadata["train_sha256"]) == (
        ".artifacts/ci_compatibility/datasets_v3/lines_legacy_amp_train.npz",
        "97e3933abf1ff27e443d1d0541e776ebb5e52c0d6edb2f2e3f2e3a744bdbf38f",
    )
    assert (legacy.metadata["test"], legacy.metadata["test_sha256"]) == (
        ".artifacts/ci_compatibility/datasets_v3/lines_legacy_amp_test.npz",
        "19f9c98e2c40f8136957ba30aca28cf97c05cdacbcb96380aaec7599759ea6cc",
    )
    for dataset in (ci, legacy):
        metadata = dataset.metadata
        assert metadata["provenance"] == (
            ".artifacts/ci_compatibility/datasets_v3/"
            "ci_compatibility_provenance.json"
        )
        assert metadata["provenance_sha256"] == (
            "dd88d0b5892537d79317772124115d10dc34635ea65848826a6f06eaf0711742"
        )
        assert dataset.kind == "synthetic"
        assert dataset.truth == "object_truth"
        assert metadata["coords_convention"] == "xy_pixels"
        assert metadata["format"] == "npz_mmap"
        assert metadata["detector_shape"] == (64, 64)
        assert metadata["grouping_max_C"] == 4
        assert metadata["measurement_key"] == "diff3d"
        assert metadata["probe_key"] == "probeGuess"
        assert metadata["probe_modes"] == 1
        assert metadata["truth_key"] == "objectGuess"
        assert metadata["truth_location"] == "embedded_test"
        assert metadata["x_key"] == "xcoords"
        assert metadata["y_key"] == "ycoords"
        assert metadata["probe"]["mask_policy"] == "model_config"
        assert metadata["probe"]["source"] == (
            "/home/ollie/Documents/PtychoPINN/datasets/fly/fly001.npz"
        )
    assert ci.metadata["probe"]["calibration"] == "count_amplitude"
    assert ci.metadata["probe"]["gauge"] == "physical_count_amplitude"
    assert ci.metadata["probe"]["train_sha256"] == (
        "f26e41a4d6be8ff99cf82906e608b39974f422ced361ad54aacc7e2e318d0e15"
    )
    assert ci.metadata["probe"]["test_sha256"] == (
        "09280f5a9543ba44ec07ad5d6fb181674bde94513fecf89146b6f6ac587e4535"
    )
    assert dict(ci.metadata["dose"]["test"]) == {
        "counts_mean": 863.987602734375,
        "dtype_max": 65535,
        "max_observed_count": 3975,
        "photons_per_image_mean": 3538893.2208,
        "photons_per_image_min": 1758704.0,
        "saturation_fraction": 0.0,
    }
    assert dict(ci.metadata["dose"]["train"]) == {
        "counts_mean": 863.99861640625,
        "dtype_max": 65535,
        "max_observed_count": 4019,
        "photons_per_image_mean": 3538938.3328,
        "photons_per_image_min": 1738342.0,
        "saturation_fraction": 0.0,
    }
    assert legacy.metadata["probe"]["calibration"] == "legacy_normalized"
    assert legacy.metadata["probe"]["gauge"] == "legacy_normalized"
    assert legacy.metadata["probe"]["sha256"] == (
        "fbdf985d265544cd0c20fe75c19a0299e3f13c601bbd599218b2bd3c9ec4dd73"
    )


@pytest.mark.parametrize(
    ("value", "match"),
    [
        ("[]", "must not be empty"),
        ("[0, 5]", "positive integer"),
        ("[5, 5]", "strictly increasing"),
        ("[20, 5]", "strictly increasing"),
        ("[true, 5]", "positive integer"),
        ("[5, 20]", "must not exceed training.epochs"),
    ],
)
def test_grid_lines_ci_convergence_rejects_malformed_milestones(
    value: str, match: str
) -> None:
    text = BASE_MANIFEST.replace(
        "[datasets.synthetic]",
        f"[diagnostics]\nmilestones = {value}\n\n[datasets.synthetic]",
    )

    with pytest.raises(manifest.ManifestError, match=match):
        manifest.loads_manifest(text)


@pytest.fixture(scope="module")
def checked_contract():
    checked = manifest.load_manifest(CI_COMPATIBILITY_SPEC)
    resolved = manifest.resolve_manifest(checked)
    return checked, resolved


def test_checked_manifest_opts_into_strict_explicit_resolution() -> None:
    checked = manifest.load_manifest(CI_COMPATIBILITY_SPEC)
    generic = manifest.loads_manifest(BASE_MANIFEST)

    assert checked.require_all_explicit is True
    assert checked.output_root == ".artifacts/ci_compatibility/full_v3"
    assert generic.require_all_explicit is False


def test_manifest_rejects_non_boolean_strict_explicit_flag(tmp_path: Path) -> None:
    text = BASE_MANIFEST.replace(
        "seeds = [3, 11]",
        'seeds = [3, 11]\nrequire_all_explicit = "yes"',
    )

    with pytest.raises(manifest.ManifestError, match="require_all_explicit"):
        manifest.load_manifest(_write(tmp_path, text))


def test_checked_manifest_resolves_every_arm_strictly_against_sealed_bundle(
    checked_contract,
) -> None:
    checked, resolved = checked_contract

    for arm in resolved.arms:
        configs = resolve_torch_configs(
            dict(arm.overrides),
            require_all_explicit=checked.require_all_explicit,
        )

        assert configs.dataset_id == arm.dataset_id
        assert configs.data_config.N == 64
        assert configs.data_config.grid_size == (2, 2)
        assert configs.data_config.C == 4
        assert configs.data_config.probe_normalize is True
        assert configs.data_config.probe_scale == 4.0
        assert configs.model_config.C_model == 4
        assert configs.model_config.C_forward == 4
        assert configs.model_config.object_big is True
        assert configs.model_config.probe_big is True
        assert configs.model_config.training_patch_weighting == "probe"
        assert configs.model_config.probe_mask is False
        assert configs.training_config.batch_size == 16
        assert configs.training_config.epochs == 80
        assert configs.training_config.optimizer == "adam"
        assert configs.training_config.learning_rate == pytest.approx(2e-4)
        assert configs.training_config.scheduler == "ReduceLROnPlateau"
        assert configs.training_config.plateau_factor == pytest.approx(0.5)
        assert configs.training_config.plateau_patience == 2
        assert configs.training_config.plateau_min_lr == pytest.approx(1e-4)
        assert configs.training_config.gradient_clip_val == pytest.approx(1.0)
        assert configs.training_config.gradient_clip_algorithm == "norm"
        assert configs.training_config.log_grad_norm is True
        assert configs.execution_config.deterministic is True
        assert configs.execution_config.accelerator == "cuda"
        assert configs.execution_config.devices == 1
        assert configs.execution_config.precision == "32-true"
        assert configs.execution_config.enable_checkpointing is True
        assert configs.execution_config.checkpoint_save_top_k == 1
        assert configs.execution_config.checkpoint_monitor_metric == "val_loss"
        assert configs.execution_config.checkpoint_mode == "min"
        assert configs.inference_config.middle_trim == 32
        profile = arm.dimensions["physics_profile"]
        expected_gain = (
            1.0 if profile == "ci_nll" else TASK30_LEGACY_PHYSICAL_GAIN
        )
        assert configs.model_config.amplitude_physics_gain == pytest.approx(
            expected_gain, rel=0.0, abs=0.0
        )
        assert configs.inference_config.patch_weighting == "probe"

        architecture = arm.dimensions["architecture"]
        profile = arm.dimensions["physics_profile"]
        if architecture == "hybrid_resnet":
            assert configs.model_config.generator_output_mode == "real_imag"
        else:
            assert architecture == "cnn"
            assert configs.model_config.cnn_output_mode == "real_imag"
            assert configs.model_config.use_shared_decoder is False

        if profile == "ci_nll":
            assert configs.model_config.amplitude_physics_gain == 1.0
            assert configs.data_config.scale_contract_version == "ci_intensity_v2"
            assert configs.data_config.measurement_domain == "count_intensity"
            assert configs.model_config.physics_forward_mode == "rectangular_scaled"
            assert configs.model_config.rect_s1s2_trainable is True
            assert configs.model_config.loss_function == "Poisson"
            assert configs.training_config.torch_loss_mode == "poisson"
            assert configs.training_config.nll is True
            assert configs.inference_config.varpro_scaling is True
        else:
            assert configs.model_config.amplitude_physics_gain == pytest.approx(
                TASK30_LEGACY_PHYSICAL_GAIN, rel=0.0, abs=0.0
            )
            assert configs.data_config.scale_contract_version == "legacy_v1"
            assert configs.data_config.measurement_domain == "normalized_amplitude"
            assert configs.data_config.normalize == "Batch"
            assert configs.data_config.data_scaling == "Parseval"
            assert configs.model_config.physics_forward_mode == "amplitude"
            assert configs.model_config.rect_s1s2_trainable is False
            assert configs.inference_config.varpro_scaling is False
            expected_mode = "poisson" if profile == "legacy_nll" else "mae"
            expected_loss = "Poisson" if profile == "legacy_nll" else "MAE"
            assert configs.model_config.loss_function == expected_loss
            assert configs.training_config.torch_loss_mode == expected_mode
            assert configs.training_config.nll is (profile == "legacy_nll")


@pytest.mark.parametrize(
    ("architecture", "profile", "missing_path", "expected_error_path"),
    [
        ("hybrid_resnet", "ci_nll", "training.scheduler", "training.scheduler"),
        (
            "hybrid_resnet",
            "ci_nll",
            "training.plateau_factor",
            "training.plateau_factor",
        ),
        (
            "hybrid_resnet",
            "ci_nll",
            "data.probe_normalize",
            "data.probe_normalize",
        ),
        ("hybrid_resnet", "ci_nll", "data.probe_scale", "data.probe_scale"),
        (
            "hybrid_resnet",
            "ci_nll",
            "execution.deterministic",
            "execution.deterministic",
        ),
        (
            "hybrid_resnet",
            "ci_nll",
            "execution.precision",
            "execution.precision",
        ),
        (
            "hybrid_resnet",
            "ci_nll",
            "execution.checkpoint_monitor_metric",
            "execution.checkpoint_monitor_metric",
        ),
        (
            "hybrid_resnet",
            "ci_nll",
            "model.generator_output_mode",
            "model.generator_output_mode",
        ),
        ("cnn", "ci_nll", "model.cnn_output_mode", "model.cnn_output_mode"),
    ],
)
def test_checked_manifest_mutations_cannot_fall_back_to_defaults(
    architecture: str,
    profile: str,
    missing_path: str,
    expected_error_path: str,
) -> None:
    checked = manifest.load_manifest(CI_COMPATIBILITY_SPEC)
    resolved = manifest.resolve_manifest(checked)
    arm = next(
        arm
        for arm in resolved.arms
        if arm.dimensions["architecture"] == architecture
        and arm.dimensions["physics_profile"] == profile
    )
    overrides = dict(arm.overrides)
    del overrides[missing_path]

    with pytest.raises(ConfigResolutionError) as error:
        resolve_torch_configs(overrides, require_all_explicit=True)

    assert "sealed validated dataset" not in str(error.value)
    assert expected_error_path in str(error.value)


def test_checked_ci_compatibility_manifest_expands_two_families_and_six_valid_arms() -> (
    None
):
    """The claim-grade study expands the six controls for each object family."""
    from scripts.studies.ablation.configuration import resolve_torch_configs

    checked = manifest.load_manifest(CI_COMPATIBILITY_SPEC)
    resolved = manifest.resolve_manifest(checked)

    assert checked.seeds == (3, 17, 29)
    assert len(resolved.arms) == 12
    assert len(resolved.runs) == 36
    assert len({run.id for run in resolved.runs}) == 36
    valid_arms = {
        ("hybrid_resnet", "ci_nll"),
        ("cnn", "ci_nll"),
        ("hybrid_resnet", "legacy_nll"),
        ("hybrid_resnet", "legacy_mae"),
        ("cnn", "legacy_nll"),
        ("cnn", "legacy_mae"),
    }
    assert {
        (
            arm.dimensions["object_family"],
            arm.dimensions["architecture"],
            arm.dimensions["physics_profile"],
        )
        for arm in resolved.arms
    } == {
        (family, architecture, profile)
        for family in ("deadleaves", "lines")
        for architecture, profile in valid_arms
    }
    assert {run.seed for run in resolved.runs} == {3, 17, 29}
    required_gate_suffixes = {
        "ci_seed_success",
        "ci_truth_amp_pearson",
        "ci_truth_amp_ssim",
        "ci_absolute_amp_nrmse",
        "ci_physical_count_error",
        "ci_model_to_poisson_oracle",
        "ci_scan_utilization",
        "ci_canvas_coverage",
        "ci_reload_allclose",
        "ci_manual_visual_review",
    }
    assert {
        f"{family}_{suffix}"
        for family in ("deadleaves", "lines")
        for suffix in required_gate_suffixes
    } <= {gate.id for gate in checked.gates}
    assert not any(gate.operator == "finite" for gate in checked.gates)
    gates_by_id = {gate.id: gate for gate in checked.gates}
    for family in ("deadleaves", "lines"):
        phase_gate = gates_by_id[f"{family}_ci_post_varpro_phase_ssim"]
        assert phase_gate.operator == "ge"
        assert phase_gate.metric == "truth_quality.post_varpro.phase_ssim"
        assert phase_gate.aggregation == "all_successful"
        assert phase_gate.threshold == 0.90
        assert phase_gate.min_successful == 2
        convergence_gate = gates_by_id[
            f"{family}_ci_validation_tail_normalized_slope"
        ]
        assert convergence_gate.operator == "le"
        assert (
            convergence_gate.metric
            == "stability.validation_loss_tail_normalized_slope"
        )
        assert convergence_gate.aggregation == "all_successful"
        assert convergence_gate.threshold == 0.001
        assert convergence_gate.min_successful == 2
    for family in ("deadleaves", "lines"):
        assert {
            f"{family}_cnn_ci_seed_success",
            f"{family}_cnn_ci_post_varpro_amp_ssim",
            f"{family}_cnn_ci_amp_variance",
            f"{family}_cnn_ci_phase_variance",
            f"{family}_cnn_ci_real_head_lower_saturation",
            f"{family}_cnn_ci_real_head_upper_saturation",
            f"{family}_cnn_ci_imag_head_lower_saturation",
            f"{family}_cnn_ci_imag_head_upper_saturation",
        } <= {gate.id for gate in checked.gates}
    diagnostic_comparison_suffixes = {
        "hybrid_ci_vs_legacy_nll",
        "cnn_ci_vs_legacy_nll",
        "hybrid_ci_vs_legacy_mae",
        "hybrid_vs_cnn_ci",
    }
    assert {comparison.id for comparison in checked.comparisons} == {
        f"{family}_{suffix}"
        for family in ("deadleaves", "lines")
        for suffix in diagnostic_comparison_suffixes
    } | {
        f"{family}_{architecture}_ci_legacy_mae_amp_ssim_ratio"
        for family in ("deadleaves", "lines")
        for architecture in ("hybrid", "cnn")
    }
    floors = tuple(
        comparison
        for comparison in checked.comparisons
        if comparison.id.endswith("amp_ssim_ratio")
    )
    assert len(floors) == 4
    assert all(comparison.threshold == 0.85 for comparison in floors)
    assert all(comparison.metric == "truth_quality.amp_ssim" for comparison in floors)
    assert all(comparison.min_pairs == 2 for comparison in floors)
    assert all(comparison.aggregation == "median" for comparison in floors)
    assert all(comparison.diagnostic is False for comparison in floors)
    assert all(
        comparison.diagnostic is True
        for comparison in checked.comparisons
        if comparison not in floors
    )

    for arm in resolved.arms:
        configs = resolve_torch_configs(dict(arm.overrides))
        family = arm.dimensions["object_family"]
        profile = arm.dimensions["physics_profile"]
        expected_dataset = (
            f"{family}_ci_3p5m" if profile == "ci_nll" else f"{family}_legacy_amp"
        )
        assert arm.dataset_id == expected_dataset
        assert configs.dataset_id == expected_dataset
        assert configs.training_config.epochs == 80
        assert configs.data_config.N == 64
        assert configs.data_config.C == 4
        assert configs.data_config.grid_size == (2, 2)
        assert configs.model_config.C_model == 4
        assert configs.model_config.C_forward == 4
        assert configs.model_config.object_big is True
        assert configs.model_config.probe_big is True
        assert configs.model_config.training_patch_weighting == "probe"
        assert configs.model_config.probe_mask is False
        assert configs.inference_config.middle_trim == 32
        assert configs.inference_config.patch_weighting == "probe"
        assert configs.execution_config.accelerator == "cuda"
        assert configs.execution_config.devices == 1
        assert configs.execution_config.precision == "32-true"
        if profile == "ci_nll":
            assert configs.data_config.scale_contract_version == "ci_intensity_v2"
            assert configs.data_config.measurement_domain == "count_intensity"
            assert configs.model_config.physics_forward_mode == "rectangular_scaled"
            assert configs.model_config.rect_s1s2_trainable is True
            assert configs.training_config.torch_loss_mode == "poisson"
            assert configs.training_config.nll is True
            assert configs.inference_config.varpro_scaling is True
        else:
            assert arm.dataset_id == f"{family}_legacy_amp"
            assert configs.data_config.scale_contract_version == "legacy_v1"
            assert configs.data_config.measurement_domain == "normalized_amplitude"
            assert configs.data_config.normalize == "Batch"
            assert configs.data_config.data_scaling == "Parseval"
            assert configs.model_config.physics_forward_mode == "amplitude"
            assert configs.model_config.rect_s1s2_trainable is False
            assert configs.inference_config.varpro_scaling is False
            if profile == "legacy_nll":
                assert configs.training_config.torch_loss_mode == "poisson"
                assert configs.training_config.nll is True
            else:
                assert configs.training_config.torch_loss_mode == "mae"
                assert configs.training_config.nll is False


def test_checked_manifest_binds_v3_descriptors_without_reading_ignored_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = CI_COMPATIBILITY_SPEC.resolve()
    original_read_bytes = Path.read_bytes
    original_read_text = Path.read_text

    def reject_artifact_read(path: Path, *args, **kwargs):
        if ".artifacts" in path.parts:
            raise AssertionError(f"unit test read ignored artifact {path}")
        method = original_read_bytes if args == ("bytes",) else original_read_text
        return method(path, **kwargs)

    monkeypatch.setattr(
        Path,
        "read_bytes",
        lambda path: reject_artifact_read(path, "bytes"),
    )
    monkeypatch.setattr(Path, "read_text", reject_artifact_read)
    monkeypatch.chdir(tmp_path)

    checked = manifest.load_manifest(spec)
    expected = {
        "deadleaves_ci_3p5m": (
            "bfd63fee5dbb60afaeff8f90b18aa8f712388cc4aae623a011edc25975e5fc67",
            "ea41285534c3d1c92d2e3d60505d880628080330e99bd5456f79ddd130056604",
            "8ce80eb1a01ee9c7164e97eb7c2402c361d6dde24992d1b06ac2217374e6a2ed",
            "583ebb73293dd5fc77836d10d2c70e5b06fc9edbd97ca8920b815731c59af1e6",
        ),
        "deadleaves_legacy_amp": (
            "4096944c8f6d3a8e324df874a7eed884e9926d5c1bdba2bacffab8e5037a22c3",
            "469f79f852a994ee5ef06e762e38967886eb800c4d441a4a96f23ca25c02c41e",
            "fbdf985d265544cd0c20fe75c19a0299e3f13c601bbd599218b2bd3c9ec4dd73",
            None,
        ),
        "lines_ci_3p5m": (
            "316c4ac841eb45184fd84f5430a08be6f80ef919963d9aeaa2406d5d1a0e239b",
            "b120bd1364053969fdaf8b1dc7d6651a54c097f52116e1704579c81909b6c747",
            "f26e41a4d6be8ff99cf82906e608b39974f422ced361ad54aacc7e2e318d0e15",
            "09280f5a9543ba44ec07ad5d6fb181674bde94513fecf89146b6f6ac587e4535",
        ),
        "lines_legacy_amp": (
            "97e3933abf1ff27e443d1d0541e776ebb5e52c0d6edb2f2e3f2e3a744bdbf38f",
            "19f9c98e2c40f8136957ba30aca28cf97c05cdacbcb96380aaec7599759ea6cc",
            "fbdf985d265544cd0c20fe75c19a0299e3f13c601bbd599218b2bd3c9ec4dd73",
            None,
        ),
    }
    assert {dataset.id for dataset in checked.datasets} == set(expected)
    for dataset in checked.datasets:
        metadata = manifest._thaw(dataset.metadata)
        train_sha256, test_sha256, probe_train_or_sha, probe_test = expected[
            dataset.id
        ]
        assert metadata["train"].startswith(
            ".artifacts/ci_compatibility/datasets_v3/"
        )
        assert metadata["test"].startswith(
            ".artifacts/ci_compatibility/datasets_v3/"
        )
        assert metadata["provenance"] == (
            ".artifacts/ci_compatibility/datasets_v3/"
            "ci_compatibility_provenance.json"
        )
        assert metadata["train_sha256"] == train_sha256
        assert metadata["test_sha256"] == test_sha256
        assert metadata["provenance_sha256"] == (
            "dd88d0b5892537d79317772124115d10dc34635ea65848826a6f06eaf0711742"
        )
        if probe_test is None:
            assert metadata["probe"]["sha256"] == probe_train_or_sha
        else:
            assert metadata["probe"]["train_sha256"] == probe_train_or_sha
            assert metadata["probe"]["test_sha256"] == probe_test


def test_checked_claim_grade_protocol_is_locked_to_corrected_manifest() -> None:
    from scripts.studies.ablation.runtime_planning import StudyRequest, load_study
    from scripts.studies.ablation.runtime_study import _preflight_selected_configs

    loaded = load_study(StudyRequest(spec=CI_COMPATIBILITY_SPEC, dry_run=True))
    _, configs = _preflight_selected_configs(loaded)
    profiles = {dataset.id: "claim_grade" for dataset in loaded.manifest.datasets}
    assert loaded.manifest.base_overrides["training.epochs"] == 80
    assert loaded.manifest.budget_threshold_contract_locked is True

    actual = manifest.protocol_fingerprint(
        loaded.manifest,
        loaded.study.runs,
        configs,
        dataset_profiles=profiles,
    )

    assert actual == loaded.manifest.expected_protocol_sha256
    from scripts.studies.ablation.verdicts import (
        IntegrationBridgeEvidence,
        evaluate_integration_bridge,
    )

    requirement = loaded.manifest.integration_bridge_requirement
    assert requirement is not None
    exact_payload = _exact_bridge_evidence_payload(requirement)
    exact_evidence = IntegrationBridgeEvidence.from_sealed_artifact_bytes(
        json.dumps(exact_payload, sort_keys=True, separators=(",", ":")).encode()
    )
    assert evaluate_integration_bridge(requirement, exact_evidence).verdict.value == "pass"
    eligible, reasons = manifest.claim_grade_eligibility(
        loaded.manifest,
        loaded.study.runs,
        loaded.selected,
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
        resolved_run_configs=configs,
        dataset_profiles=profiles,
        integration_bridge_evidence=exact_evidence,
    )
    assert eligible is True
    assert reasons == ()

    mismatched = exact_evidence.contract.to_mapping()
    mismatched["loader_kind"] = "mmap"
    mismatched_evidence = {
        **{
            name: getattr(exact_evidence, name)
            for name, item in exact_evidence.__dataclass_fields__.items()
            if name != "contract" and item.init
        },
        "contract": mismatched,
    }
    eligible, reasons = manifest.claim_grade_eligibility(
        loaded.manifest,
        loaded.study.runs,
        loaded.selected,
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
        resolved_run_configs=configs,
        dataset_profiles=profiles,
        integration_bridge_evidence=(
            IntegrationBridgeEvidence.from_sealed_artifact_bytes(
                json.dumps(
                    mismatched_evidence, sort_keys=True, separators=(",", ":")
                ).encode()
            )
        ),
    )
    assert eligible is False
    assert reasons == ("integration_bridge_prerequisite",)
    assert not hasattr(manifest, "_CLAIM_GRADE_EPOCHS")


def test_checked_claim_grade_study_binds_supplied_bridge_evidence(
    tmp_path: Path,
) -> None:
    from scripts.studies.ablation.runtime_planning import StudyRequest, load_study

    checked = manifest.load_manifest(CI_COMPATIBILITY_SPEC)
    requirement = checked.integration_bridge_requirement
    assert requirement is not None
    evidence_bytes = json.dumps(
        _exact_bridge_evidence_payload(requirement),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    evidence_path = tmp_path / "bridge-evidence.json"
    evidence_path.write_bytes(evidence_bytes)

    loaded = load_study(
        StudyRequest(
            spec=CI_COMPATIBILITY_SPEC,
            dry_run=True,
            integration_bridge_evidence=evidence_path,
        )
    )

    assert loaded.integration_bridge_evidence is not None
    assert loaded.integration_bridge_evidence_sha256 == hashlib.sha256(
        evidence_bytes
    ).hexdigest()


def test_claim_grade_manifest_is_disqualified_until_integration_bridge_passes() -> None:
    parsed = manifest.load_manifest(CI_COMPATIBILITY_SPEC)
    runs = manifest.expand_runs(parsed)

    eligible, reasons = manifest.claim_grade_eligibility(
        parsed,
        runs,
        runs,
        epochs_override=False,
        seeds_override=False,
        matrix_filter=False,
        dataset_override=False,
        external_dataset_spec=False,
        dirty_checkout=False,
    )

    assert eligible is False
    assert "integration_bridge_prerequisite" in reasons
    assert "unlocked_budget_threshold_contract" not in reasons


def test_checked_ci_compatibility_manifest_routes_experimental_no_reference_gates(
    tmp_path: Path,
) -> None:
    from scripts.studies.ablation.runtime_planning import StudyRequest, load_study

    descriptor = tmp_path / "experimental_no_reference.toml"
    descriptor.write_text(
        """[schema]
version = 1

[dataset]
id = "experimental_no_reference"
kind = "experimental"
format = "npz_mmap"
scale_contract_version = "ci_intensity_v2"
measurement_domain = "count_intensity"
truth = "none"
truth_location = "none"
measurement_key = "diff3d"
probe_key = "probeGuess"
x_key = "xcoords"
y_key = "ycoords"
coords_convention = "xy_pixels"
detector_shape = [64, 64]
grouping_max_C = 4
probe_modes = 1
train = "/data/experimental_no_reference_train.npz"
test = "/data/experimental_no_reference_test.npz"
provenance = "/data/experimental_no_reference_provenance.json"
train_sha256 = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
test_sha256 = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
provenance_sha256 = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"

[dataset.probe]
source = "iterative_reconstruction"
calibration = "count_amplitude"
gauge = "physical_count_amplitude"
mask_policy = "model_config"
sha256 = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"

[dataset.dose.train]
counts_mean = 864.0
photons_per_image_min = 1000000.0
photons_per_image_mean = 3538944.0
max_observed_count = 7600
dtype_max = 65535
saturation_fraction = 0.0

[dataset.dose.test]
counts_mean = 864.0
photons_per_image_min = 1000000.0
photons_per_image_mean = 3538944.0
max_observed_count = 7600
dtype_max = 65535
saturation_fraction = 0.0
""",
        encoding="utf-8",
    )

    experimental = load_study(
        StudyRequest(
            spec=CI_COMPATIBILITY_SPEC,
            dataset_spec=descriptor,
            dataset="experimental_no_reference",
            only="object_family=deadleaves,physics_profile=ci_nll",
            dry_run=True,
        )
    )

    assert len(experimental.selected) == 6
    assert all(
        run.id.startswith("hybrid-resnet-ci-compatibility--experimental_no_reference--")
        for run in experimental.selected
    )
    gate_states = {
        gate.id: gate.applicability.value for gate in experimental.study.gates
    }
    assert gate_states["deadleaves_ci_truth_amp_pearson"] == "not_applicable"
    assert gate_states["deadleaves_ci_absolute_amp_nrmse"] == "not_applicable"
    assert gate_states["deadleaves_ci_physical_count_error"] == "active"
    assert gate_states["deadleaves_ci_model_to_poisson_oracle"] == "not_applicable"
    assert gate_states["deadleaves_ci_manual_visual_review"] == "active"


def test_manifest_exposes_one_public_immutable_metric_registry():
    assert isinstance(manifest.METRIC_PATHS, frozenset)
    assert manifest._METRIC_PATHS is manifest.METRIC_PATHS


def _closed_manifest_dataset(dataset_id: str, *, kind: str, truth: str) -> str:
    truth_fields = {
        "object_truth": 'truth_location = "embedded_test"\ntruth_key = "objectGuess"',
        "reference_reconstruction": (
            'truth_location = "external_npz"\ntruth_key = "object"\n'
            f'reference = "fixtures/{dataset_id}_reference.npz"\n'
            f'reference_sha256 = "{"d" * 64}"'
        ),
        "none": 'truth_location = "none"',
    }[truth]
    return f"""[datasets.{dataset_id}]
kind = "{kind}"
truth = "{truth}"
format = "npz_mmap"
scale_contract_version = "legacy_v1"
measurement_domain = "normalized_amplitude"
{truth_fields}
measurement_key = "diff3d"
probe_key = "probeGuess"
x_key = "xcoords"
y_key = "ycoords"
coords_convention = "xy_pixels"
detector_shape = [2, 2]
grouping_max_C = 4
probe_modes = 1
train = "fixtures/{dataset_id}_train.npz"
test = "fixtures/{dataset_id}_test.npz"
provenance = "fixtures/{dataset_id}_provenance.json"
train_sha256 = "{"a" * 64}"
test_sha256 = "{"b" * 64}"
provenance_sha256 = "{"c" * 64}"

[datasets.{dataset_id}.probe]
source = "manifest_test_fixture"
calibration = "legacy_normalized"
gauge = "legacy_normalized"
mask_policy = "model_config"
sha256 = "{"e" * 64}"
"""


def _replace_manifest_dataset(
    text: str, dataset_id: str, *, kind: str, truth: str
) -> str:
    pattern = (
        rf"\[datasets\.{re.escape(dataset_id)}\]\n.*?"
        r"(?=\n\[datasets\.[^\n]+\]\nkind = |\n\[\[matrix\.)"
    )
    return re.sub(
        pattern,
        _closed_manifest_dataset(dataset_id, kind=kind, truth=truth).rstrip(),
        text,
        count=1,
        flags=re.S,
    )


BASE_MANIFEST = f"""
[schema]
version = 1

[study]
id = "compatibility"
seeds = [3, 11]
output_root = "artifacts/default"

[base.overrides]
"dataset.id" = "synthetic"
"training.epochs" = 10
"model.width" = 32

{_closed_manifest_dataset("synthetic", kind="synthetic", truth="object_truth")}
{_closed_manifest_dataset("experimental", kind="experimental", truth="none")}

[[matrix.dimensions]]
name = "architecture"

[[matrix.dimensions.values]]
id = "hybrid"
[matrix.dimensions.values.overrides]
"model.architecture" = "hybrid_resnet"

[[matrix.dimensions.values]]
id = "cnn"
[matrix.dimensions.values.overrides]
"model.architecture" = "cnn"

[[matrix.dimensions]]
name = "physics_profile"

[[matrix.dimensions.values]]
id = "ci"
[matrix.dimensions.values.overrides]
"training.loss" = "poisson"

[[matrix.dimensions.values]]
id = "legacy"
[matrix.dimensions.values.overrides]
"training.loss" = "mae"
"""


TWIN_DATASET_MANIFEST = f"""
[schema]
version = 1

[study]
id = "twin-comparison"
seeds = [3]

[base.overrides]
"dataset.id" = "ci_twin"
"training.epochs" = 1

{_closed_manifest_dataset("ci_twin", kind="synthetic", truth="object_truth")}
{_closed_manifest_dataset("legacy_twin", kind="synthetic", truth="object_truth")}
{_closed_manifest_dataset("replacement", kind="experimental", truth="none")}

[[matrix.dimensions]]
name = "architecture"

[[matrix.dimensions.values]]
id = "hybrid"
[matrix.dimensions.values.overrides]
"model.architecture" = "hybrid_resnet"

[[matrix.dimensions]]
name = "physics_profile"

[[matrix.dimensions.values]]
id = "ci"
[matrix.dimensions.values.overrides]
"dataset.id" = "ci_twin"
"training.loss" = "poisson"

[[matrix.dimensions.values]]
id = "legacy"
[matrix.dimensions.values.overrides]
"dataset.id" = "legacy_twin"
"training.loss" = "mae"
"""


def _write(tmp_path: Path, text: str = BASE_MANIFEST) -> Path:
    path = tmp_path / "study.toml"
    path.write_text(text, encoding="utf-8")
    return path


def _api():
    from scripts.studies.ablation import manifest

    return manifest


def _with_dataset_dimension(text: str) -> str:
    return text.replace('"dataset.id" = "synthetic"\n', "").replace(
        '[[matrix.dimensions]]\nname = "architecture"',
        """[[matrix.dimensions]]
name = "dataset"

[[matrix.dimensions.values]]
id = "synthetic"

[[matrix.dimensions.values]]
id = "experimental"

[[matrix.dimensions]]
name = "architecture"
""",
        1,
    )


def test_loads_v1_with_compact_canonical_json_and_frozen_values(tmp_path):
    api = _api()
    text = BASE_MANIFEST.replace(
        '"model.width" = 32',
        '"model.width" = 32\n"model.label" = "__import__(\'pathlib\').Path(\'owned\').touch()"',
    )

    manifest = api.load_manifest(_write(tmp_path, text))

    assert manifest.schema_version == 1
    assert manifest.base_overrides["model.label"].startswith("__import__")
    assert not (tmp_path / "owned").exists()
    assert manifest.canonical_json == json.dumps(
        manifest.to_dict(),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    with pytest.raises(TypeError):
        manifest.base_overrides["model.width"] = 64


def test_frozen_dict_rejects_non_string_keys():
    api = _api()

    with pytest.raises(TypeError, match="string"):
        api.FrozenDict({1: "value"})


def test_manifest_and_resolved_backing_state_is_assignment_proof(tmp_path):
    api = _api()
    manifest = api.load_manifest(_write(tmp_path))
    resolved = api.resolve_manifest(manifest)
    run = resolved.runs[0]
    canonical_before = manifest.canonical_json
    manifest_before = manifest.to_dict()
    overrides_before = dict(run.overrides)
    ids_before = (run.arm_id, run.id)

    with pytest.raises(AttributeError):
        manifest._raw._data = MappingProxyType({"schema": {"version": 99}})
    with pytest.raises(AttributeError):
        manifest._raw["base"]["overrides"]._data = MappingProxyType(
            {"dataset.id": "experimental"}
        )
    with pytest.raises(TypeError):
        manifest._raw["study"]["seeds"][0] = 99
    with pytest.raises(AttributeError):
        run.overrides._data = MappingProxyType({"training.epochs": 999})
    with pytest.raises(TypeError):
        run.dimension_options["architecture"][0] = "changed"

    assert manifest.canonical_json == canonical_before
    assert manifest.to_dict() == manifest_before
    assert dict(run.overrides) == overrides_before
    assert (run.arm_id, run.id) == ids_before


@pytest.mark.parametrize(
    "text, match",
    [
        (BASE_MANIFEST.replace("version = 1", "version = 2"), "schema.version"),
        (BASE_MANIFEST.replace("version = 1", "version = 1.0"), "schema.version"),
        (BASE_MANIFEST.replace("version = 1", "version = true"), "schema.version"),
        ("unknown = 1\n" + BASE_MANIFEST, "unknown"),
        (BASE_MANIFEST.replace('id = "compatibility"', "id = 4"), "study.id"),
        (BASE_MANIFEST.replace("seeds = [3, 11]", 'seeds = "3"'), "study.seeds"),
    ],
)
def test_rejects_unknown_versions_and_closed_or_ill_typed_fields(tmp_path, text, match):
    api = _api()

    with pytest.raises(api.ManifestError, match=match):
        api.load_manifest(_write(tmp_path, text))


@pytest.mark.parametrize(
    "replacement, match",
    [
        ('"model.width" = [32]', "scalar"),
        ('"model.width" = nan', "finite"),
        ('"model.width" = { value = 32 }', "scalar"),
    ],
)
def test_rejects_non_json_scalar_overrides(tmp_path, replacement, match):
    api = _api()
    text = BASE_MANIFEST.replace('"model.width" = 32', replacement)

    with pytest.raises(api.ManifestError, match=match):
        api.load_manifest(_write(tmp_path, text))


def test_expands_cartesian_product_in_declaration_order_and_matches_all_exclude_keys(
    tmp_path,
):
    api = _api()
    text = (
        BASE_MANIFEST
        + """

[[matrix.exclude]]
architecture = "cnn"
physics_profile = "legacy"
"""
    )

    runs = api.expand_runs(api.load_manifest(_write(tmp_path, text)))

    assert [run.id for run in runs] == [
        "compatibility--synthetic--hybrid--ci--seed-3",
        "compatibility--synthetic--hybrid--ci--seed-11",
        "compatibility--synthetic--hybrid--legacy--seed-3",
        "compatibility--synthetic--hybrid--legacy--seed-11",
        "compatibility--synthetic--cnn--ci--seed-3",
        "compatibility--synthetic--cnn--ci--seed-11",
    ]


def test_include_requires_complete_assignment_and_can_readd_excluded_assignment(
    tmp_path,
):
    api = _api()
    incomplete = (
        BASE_MANIFEST
        + """

[[matrix.include]]
architecture = "cnn"
"""
    )
    with pytest.raises(api.ManifestError, match="every dimension"):
        api.load_manifest(_write(tmp_path, incomplete))

    complete = (
        BASE_MANIFEST
        + """

[[matrix.exclude]]
architecture = "cnn"
physics_profile = "legacy"

[[matrix.include]]
architecture = "cnn"
physics_profile = "legacy"
[matrix.include.overrides]
"execution.deterministic" = true
"""
    )
    runs = api.expand_runs(api.load_manifest(_write(tmp_path, complete)))

    assert len(runs) == 8
    assert runs[-1].dimensions == {"architecture": "cnn", "physics_profile": "legacy"}
    assert runs[-1].overrides["execution.deterministic"] is True


def test_include_rejects_duplicate_resulting_assignment(tmp_path):
    api = _api()
    text = (
        BASE_MANIFEST
        + """

[[matrix.include]]
architecture = "hybrid"
physics_profile = "ci"
"""
    )

    with pytest.raises(api.ManifestError, match="duplicate.*assignment"):
        api.expand_runs(api.load_manifest(_write(tmp_path, text)))


@pytest.mark.parametrize(
    "text, match",
    [
        (BASE_MANIFEST.replace("seeds = [3, 11]", "seeds = [3, 3]"), "run.*id|seed"),
        (
            BASE_MANIFEST.replace(
                'name = "physics_profile"', 'name = "architecture"', 1
            ),
            "dimension.*architecture",
        ),
        (
            BASE_MANIFEST.replace('id = "legacy"', 'id = "ci"', 1),
            "value.*ci",
        ),
    ],
)
def test_rejects_duplicate_run_dimension_and_value_ids(tmp_path, text, match):
    api = _api()

    with pytest.raises(api.ManifestError, match=match):
        api.load_manifest(_write(tmp_path, text))


@pytest.mark.parametrize("invalid_seed", ["true", "1.0"])
def test_manifest_seeds_reject_bool_and_float(tmp_path, invalid_seed):
    api = _api()
    text = BASE_MANIFEST.replace("seeds = [3, 11]", f"seeds = [{invalid_seed}]")

    with pytest.raises(api.ManifestError, match="seed|integer"):
        api.load_manifest(_write(tmp_path, text))


def test_manifest_and_cli_seeds_accept_zero_with_exact_run_id(tmp_path):
    api = _api()
    manifest_seed_zero = api.load_manifest(
        _write(tmp_path, BASE_MANIFEST.replace("seeds = [3, 11]", "seeds = [0]"))
    )

    manifest_runs = api.expand_runs(manifest_seed_zero)
    cli_runs = api.expand_runs(
        api.load_manifest(_write(tmp_path, BASE_MANIFEST)), seeds=[0]
    )

    assert manifest_runs[0].id == "compatibility--synthetic--hybrid--ci--seed-0"
    assert cli_runs[0].id == "compatibility--synthetic--hybrid--ci--seed-0"


@pytest.mark.parametrize("invalid_seed", [True, 1.0])
def test_cli_seeds_reject_bool_and_float(tmp_path, invalid_seed):
    api = _api()
    manifest = api.load_manifest(_write(tmp_path))

    with pytest.raises(api.ManifestError, match="seed|integer"):
        api.expand_runs(manifest, seeds=[invalid_seed])


def test_tomllib_rejects_duplicate_study_id(tmp_path):
    api = _api()
    text = BASE_MANIFEST.replace(
        'id = "compatibility"', 'id = "compatibility"\nid = "other"', 1
    )

    with pytest.raises(api.ManifestError, match="TOML|overwrite"):
        api.load_manifest(_write(tmp_path, text))


def test_base_override_may_be_specialized_by_dimension(tmp_path):
    api = _api()
    text = BASE_MANIFEST.replace(
        '"model.width" = 32',
        '"model.width" = 32\n"model.architecture" = "default"',
    )

    runs = api.expand_runs(api.load_manifest(_write(tmp_path, text)))

    assert runs[0].overrides["model.architecture"] == "hybrid_resnet"
    assert runs[-1].overrides["model.architecture"] == "cnn"


def test_rejects_conflicting_dimension_and_include_overrides(tmp_path):
    api = _api()
    dimension_collision = BASE_MANIFEST.replace(
        '"training.loss" = "poisson"',
        '"training.loss" = "poisson"\n"model.architecture" = "other"',
    )
    with pytest.raises(api.ManifestError, match="conflicting.*model.architecture"):
        api.expand_runs(api.load_manifest(_write(tmp_path, dimension_collision)))

    include_collision = (
        BASE_MANIFEST
        + """

[[matrix.exclude]]
architecture = "cnn"
physics_profile = "legacy"

[[matrix.include]]
architecture = "cnn"
physics_profile = "legacy"
[matrix.include.overrides]
"training.loss" = "poisson"
"""
    )
    with pytest.raises(api.ManifestError, match="conflicting.*training.loss"):
        api.expand_runs(api.load_manifest(_write(tmp_path, include_collision)))


def test_conflict_error_reports_original_and_later_assignment_sources(tmp_path):
    api = _api()
    dimension_collision = BASE_MANIFEST.replace(
        '"training.loss" = "poisson"',
        '"training.loss" = "poisson"\n"model.architecture" = "other"',
    )
    with pytest.raises(api.ManifestError) as dimension_error:
        api.expand_runs(api.load_manifest(_write(tmp_path, dimension_collision)))
    assert "dimension architecture=hybrid" in str(dimension_error.value)
    assert "dimension physics_profile=ci" in str(dimension_error.value)

    include_collision = (
        BASE_MANIFEST
        + """
[[matrix.exclude]]
architecture = "hybrid"
physics_profile = "ci"

[[matrix.include]]
architecture = "hybrid"
physics_profile = "ci"
[matrix.include.overrides]
"model.architecture" = "other"
"""
    )
    with pytest.raises(api.ManifestError) as include_error:
        api.expand_runs(api.load_manifest(_write(tmp_path, include_collision)))
    assert "dimension architecture=hybrid" in str(include_error.value)
    assert "matrix include" in str(include_error.value)


@pytest.mark.parametrize(
    "left_value,right_value", [("true", "1"), ("1", "1.0"), ("-0.0", "0.0")]
)
def test_dimension_override_collisions_are_json_type_sensitive(
    tmp_path, left_value, right_value
):
    api = _api()
    text = BASE_MANIFEST.replace(
        '"model.architecture" = "hybrid_resnet"',
        f'"model.architecture" = "hybrid_resnet"\n"model.shared" = {left_value}',
    ).replace(
        '"training.loss" = "poisson"',
        f'"training.loss" = "poisson"\n"model.shared" = {right_value}',
    )

    with pytest.raises(api.ManifestError, match="conflicting.*model.shared"):
        api.expand_runs(api.load_manifest(_write(tmp_path, text)))


@pytest.mark.parametrize(
    "left_value,right_value", [("true", "1"), ("1", "1.0"), ("-0.0", "0.0")]
)
def test_include_override_collisions_are_json_type_sensitive(
    tmp_path, left_value, right_value
):
    api = _api()
    text = BASE_MANIFEST.replace(
        '"model.architecture" = "hybrid_resnet"',
        f'"model.architecture" = "hybrid_resnet"\n"model.shared" = {left_value}',
    )
    text += f"""
[[matrix.exclude]]
architecture = "hybrid"
physics_profile = "ci"

[[matrix.include]]
architecture = "hybrid"
physics_profile = "ci"
[matrix.include.overrides]
"model.shared" = {right_value}
"""

    with pytest.raises(api.ManifestError, match="conflicting.*model.shared"):
        api.expand_runs(api.load_manifest(_write(tmp_path, text)))


def test_excluded_assignment_is_removed_before_override_collision_checks(tmp_path):
    api = _api()
    text = BASE_MANIFEST.replace(
        '"training.loss" = "poisson"',
        '"training.loss" = "poisson"\n"model.architecture" = "conflict"',
    )
    text += """
[[matrix.exclude]]
physics_profile = "ci"
"""

    runs = api.expand_runs(api.load_manifest(_write(tmp_path, text)))

    assert len(runs) == 4
    assert {run.dimensions["physics_profile"] for run in runs} == {"legacy"}


def test_cli_dataset_epochs_seeds_and_output_root_apply_last_without_generic_mutation(
    tmp_path,
):
    api = _api()
    manifest = api.load_manifest(_write(tmp_path))

    runs = api.expand_runs(
        manifest,
        dataset="experimental",
        epochs=4,
        seeds=[29],
        output_root=tmp_path / "cli-output",
    )

    assert {run.dataset_id for run in runs} == {"experimental"}
    assert {run.seed for run in runs} == {29}
    assert {run.overrides["training.epochs"] for run in runs} == {4}
    assert {run.output_root for run in runs} == {str(tmp_path / "cli-output")}
    assert all("synthetic" not in run.id for run in runs)
    with pytest.raises(TypeError):
        api.expand_runs(manifest, learning_rate=0.1)


def test_dataset_cli_filters_dataset_dimension_and_dataset_appears_once_in_ids(
    tmp_path,
):
    api = _api()
    text = _with_dataset_dimension(BASE_MANIFEST)

    runs = api.expand_runs(
        api.load_manifest(_write(tmp_path, text)), dataset="experimental", seeds=[5]
    )

    assert len(runs) == 4
    assert runs[0].id == "compatibility--experimental--hybrid--ci--seed-5"
    assert runs[0].id.count("experimental") == 1


def test_only_selects_exact_arm_or_run_and_dimension_conjunction(tmp_path):
    api = _api()
    runs = api.expand_runs(api.load_manifest(_write(tmp_path)))

    arm = api.select_runs(runs, "compatibility--synthetic--hybrid--ci")
    one_run = api.select_runs(runs, "compatibility--synthetic--hybrid--ci--seed-3")
    conjunction = api.select_runs(runs, "architecture=hybrid,physics_profile=ci")

    assert [run.seed for run in arm] == [3, 11]
    assert [run.seed for run in one_run] == [3]
    assert {run.dimensions["architecture"] for run in conjunction} == {"hybrid"}


@pytest.mark.parametrize(
    "only",
    [
        "unknown=value",
        "architecture=unknown",
        "architecture",
        "architecture=hybrid,",
        "architecture=hybrid,architecture=cnn",
        "architecture==hybrid",
        "does-not-exist",
    ],
)
def test_only_rejects_unknown_or_malformed_selectors(tmp_path, only):
    api = _api()
    runs = api.expand_runs(api.load_manifest(_write(tmp_path)))

    with pytest.raises(api.ManifestError):
        api.select_runs(runs, only)


VALID_RULES = """
[[gates]]
id = "successes"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "status_count_ge"
status = "success"
threshold = 1
requested = 2

[[gates]]
id = "pearson"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "ge"
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.5
min_successful = 1
requires = ["has_object_truth"]
when_dataset_kind = "synthetic"

[[gates]]
id = "dose_cv"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "le"
metric = "measurement_consistency.dose.object_scale"
aggregation = "cv"
threshold = 0.15
min_successful = 1

[[comparisons]]
id = "relative"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "hybrid", physics_profile = "legacy" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "mean"
threshold = 0.7
min_pairs = 1
"""


def test_gate_and_comparison_targets_resolve_after_dataset_replacement(tmp_path):
    api = _api()
    manifest = api.load_manifest(_write(tmp_path, BASE_MANIFEST + VALID_RULES))

    resolved = api.resolve_manifest(manifest, dataset="experimental", seeds=[3])

    assert resolved.gates[0].target_arm_id == "compatibility--experimental--hybrid--ci"
    assert resolved.comparisons[0].left_arm_id.startswith("compatibility--experimental")
    assert resolved.comparisons[0].right_arm_id.startswith(
        "compatibility--experimental"
    )
    assert resolved.gates[1].applicability is api.RuleApplicability.NOT_APPLICABLE
    assert resolved.gates[1].reason == "dataset_kind"


def test_default_comparison_selectors_pair_distinct_twin_datasets(tmp_path):
    api = _api()
    comparison = """
[[comparisons]]
id = "ci_vs_legacy"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "hybrid", physics_profile = "legacy" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.7
min_pairs = 1
"""

    resolved = api.resolve_manifest(
        api.load_manifest(_write(tmp_path, TWIN_DATASET_MANIFEST + comparison))
    )

    paired = resolved.comparisons[0]
    assert paired.left_dataset_id == "ci_twin"
    assert paired.right_dataset_id == "legacy_twin"
    assert paired.left_arm_id == "twin-comparison--ci_twin--hybrid--ci"
    assert paired.right_arm_id == "twin-comparison--legacy_twin--hybrid--legacy"


def test_explicit_dataset_selectors_pair_distinct_twin_datasets(tmp_path):
    api = _api()
    comparison = """
[[comparisons]]
id = "ci_vs_legacy"
left = { dataset = "ci_twin", architecture = "hybrid", physics_profile = "ci" }
right = { dataset = "legacy_twin", architecture = "hybrid", physics_profile = "legacy" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.7
min_pairs = 1
"""

    resolved = api.resolve_manifest(
        api.load_manifest(_write(tmp_path, TWIN_DATASET_MANIFEST + comparison))
    )

    paired = resolved.comparisons[0]
    assert (paired.left_dataset_id, paired.right_dataset_id) == (
        "ci_twin",
        "legacy_twin",
    )


def test_dataset_replacement_applies_before_default_comparison_resolution(tmp_path):
    api = _api()
    comparison = """
[[comparisons]]
id = "ci_vs_legacy"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "hybrid", physics_profile = "legacy" }
operator = "paired_ratio_ge"
metric = "measurement_consistency.relative_l2_intensity_error"
aggregation = "median"
threshold = 0.7
min_pairs = 1
"""
    manifest = api.load_manifest(_write(tmp_path, TWIN_DATASET_MANIFEST + comparison))

    resolved = api.resolve_manifest(manifest, dataset="replacement")

    paired = resolved.comparisons[0]
    assert paired.left_dataset_id == "replacement"
    assert paired.right_dataset_id == "replacement"
    assert "--replacement--" in paired.left_arm_id
    assert "--replacement--" in paired.right_arm_id


def test_globally_unique_gate_resolves_with_active_legacy_twin(tmp_path):
    api = _api()
    gate = """
[[gates]]
id = "ci_only"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "manual_review"
"""

    resolved = api.resolve_manifest(
        api.load_manifest(_write(tmp_path, TWIN_DATASET_MANIFEST + gate))
    )

    assert len(resolved.gates) == 1
    assert resolved.gates[0].dataset_id == "ci_twin"
    assert resolved.gates[0].target_arm_id.endswith("--hybrid--ci")


def test_dataset_grouped_comparison_resolves_one_pair_per_dataset(tmp_path):
    api = _api()
    comparison = """
[[comparisons]]
id = "architecture_pair"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "cnn", physics_profile = "ci" }
operator = "paired_ratio_ge"
metric = "measurement_consistency.relative_l2_intensity_error"
aggregation = "median"
threshold = 0.7
min_pairs = 1
"""
    manifest = api.load_manifest(
        _write(tmp_path, _with_dataset_dimension(BASE_MANIFEST) + comparison)
    )

    resolved = api.resolve_manifest(manifest)

    assert [item.dataset_id for item in resolved.comparisons] == [
        "synthetic",
        "experimental",
    ]
    assert all(
        item.left_dataset_id == item.right_dataset_id for item in resolved.comparisons
    )


def test_globally_unique_comparison_rejects_self_ratio(tmp_path):
    api = _api()
    comparison = """
[[comparisons]]
id = "self_ratio"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "hybrid", physics_profile = "ci" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.7
min_pairs = 1
"""
    manifest = api.load_manifest(_write(tmp_path, TWIN_DATASET_MANIFEST + comparison))

    with pytest.raises(api.ManifestError, match="self-ratio|same arm"):
        api.resolve_manifest(manifest)


def test_dataset_grouped_comparison_rejects_self_ratios(tmp_path):
    api = _api()
    comparison = """
[[comparisons]]
id = "self_ratio"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "hybrid", physics_profile = "ci" }
operator = "paired_ratio_ge"
metric = "measurement_consistency.relative_l2_intensity_error"
aggregation = "median"
threshold = 0.7
min_pairs = 1
"""
    manifest = api.load_manifest(
        _write(tmp_path, _with_dataset_dimension(BASE_MANIFEST) + comparison)
    )

    with pytest.raises(api.ManifestError, match="self-ratio|same arm"):
        api.resolve_manifest(manifest)


def test_explicit_twin_selector_fails_after_dataset_replacement(tmp_path):
    api = _api()
    comparison = """
[[comparisons]]
id = "ci_vs_legacy"
left = { dataset = "ci_twin", architecture = "hybrid", physics_profile = "ci" }
right = { dataset = "legacy_twin", architecture = "hybrid", physics_profile = "legacy" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.7
min_pairs = 1
"""
    manifest = api.load_manifest(_write(tmp_path, TWIN_DATASET_MANIFEST + comparison))

    with pytest.raises(api.ManifestError, match="comparison 'ci_vs_legacy' left.*zero"):
        api.resolve_manifest(manifest, dataset="replacement")


def _paired_rule(*, left_profile: str, right_profile: str, conditions: str) -> str:
    return f"""
[[comparisons]]
id = "ordered_pair"
left = {{ architecture = "hybrid", physics_profile = "{left_profile}" }}
right = {{ architecture = "hybrid", physics_profile = "{right_profile}" }}
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.7
min_pairs = 1
{conditions}
"""


def test_comparison_kind_applicability_is_operand_order_independent(tmp_path):
    api = _api()
    manifest_text = _replace_manifest_dataset(
        TWIN_DATASET_MANIFEST, "ci_twin", kind="experimental", truth="none"
    )
    manifest_text = _replace_manifest_dataset(
        manifest_text, "legacy_twin", kind="experimental", truth="none"
    )
    conditions = 'requires = ["has_object_truth"]\nwhen_dataset_kind = "synthetic"'

    outcomes = []
    for left, right in (("ci", "legacy"), ("legacy", "ci")):
        manifest = api.load_manifest(
            _write(
                tmp_path,
                manifest_text
                + _paired_rule(
                    left_profile=left,
                    right_profile=right,
                    conditions=conditions,
                ),
            )
        )
        item = api.resolve_manifest(manifest).comparisons[0]
        outcomes.append((item.applicability, item.reason))

    assert outcomes == [
        (api.RuleApplicability.NOT_APPLICABLE, "dataset_kind"),
        (api.RuleApplicability.NOT_APPLICABLE, "dataset_kind"),
    ]


def test_comparison_missing_capability_is_operand_order_independent(tmp_path):
    api = _api()
    manifest_text = _replace_manifest_dataset(
        TWIN_DATASET_MANIFEST,
        "legacy_twin",
        kind="experimental",
        truth="reference_reconstruction",
    )
    conditions = (
        'requires = ["has_object_truth", "has_reference"]\n'
        'on_missing_capability = "not_applicable"'
    )

    outcomes = []
    for left, right in (("ci", "legacy"), ("legacy", "ci")):
        manifest = api.load_manifest(
            _write(
                tmp_path,
                manifest_text
                + _paired_rule(
                    left_profile=left,
                    right_profile=right,
                    conditions=conditions,
                ),
            )
        )
        item = api.resolve_manifest(manifest).comparisons[0]
        outcomes.append((item.applicability, item.reason))

    assert outcomes == [
        (
            api.RuleApplicability.NOT_APPLICABLE,
            "missing_capability:has_object_truth",
        ),
        (
            api.RuleApplicability.NOT_APPLICABLE,
            "missing_capability:has_object_truth",
        ),
    ]


def test_comparison_missing_capability_error_is_operand_order_independent(tmp_path):
    api = _api()
    manifest_text = _replace_manifest_dataset(
        TWIN_DATASET_MANIFEST,
        "legacy_twin",
        kind="experimental",
        truth="reference_reconstruction",
    )
    conditions = 'requires = ["has_object_truth", "has_reference"]'

    errors = []
    for left, right in (("ci", "legacy"), ("legacy", "ci")):
        manifest = api.load_manifest(
            _write(
                tmp_path,
                manifest_text
                + _paired_rule(
                    left_profile=left,
                    right_profile=right,
                    conditions=conditions,
                ),
            )
        )
        with pytest.raises(api.ManifestError) as exc_info:
            api.resolve_manifest(manifest)
        errors.append(str(exc_info.value))

    assert errors[0] == errors[1]
    assert "has_object_truth" in errors[0]
    assert "legacy_twin" in errors[0]


def test_reference_gate_can_be_typed_not_applicable_for_no_reference_experimental_data(
    tmp_path,
):
    api = _api()
    rule = """
[[gates]]
id = "reference_quality"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "ge"
metric = "reference_agreement.amp_pearson"
aggregation = "median"
threshold = 0.5
min_successful = 1
requires = ["has_reference"]
when_dataset_kind = "experimental"
on_missing_capability = "not_applicable"
"""

    resolved = api.resolve_manifest(
        api.load_manifest(_write(tmp_path, BASE_MANIFEST + rule)),
        dataset="experimental",
    )

    gate = resolved.gates[0]
    assert gate.applicability is api.RuleApplicability.NOT_APPLICABLE
    assert gate.reason == "missing_capability:has_reference"


def test_missing_capability_defaults_to_error(tmp_path):
    api = _api()
    rule = """
[[gates]]
id = "reference_quality"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "finite"
metric = "reference_agreement.amp_pearson"
min_successful = 1
requires = ["has_reference"]
when_dataset_kind = "experimental"
"""
    manifest = api.load_manifest(_write(tmp_path, BASE_MANIFEST + rule))

    with pytest.raises(api.ManifestError, match="has_reference"):
        api.resolve_manifest(manifest, dataset="experimental")


@pytest.mark.parametrize(
    "rule, match",
    [
        (
            """[[gates]]
id = "bad"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "ge"
metric = "truth_quality.not_registered"
aggregation = "median"
threshold = 1.0
min_successful = 1
""",
            "metric",
        ),
        (
            """[[gates]]
id = "bad"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "ge"
metric = "truth_quality.amp_pearson"
aggregation = "sum"
threshold = 1.0
min_successful = 1
""",
            "aggregation",
        ),
        (
            """[[gates]]
id = "bad"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "status_count_ge"
status = "success"
threshold = 1
""",
            "requested",
        ),
        (
            """[[gates]]
id = "bad"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "manual_review"
threshold = 1
""",
            "threshold|irrelevant|unknown",
        ),
        (
            """[[gates]]
id = "bad"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "unknown"
""",
            "operator",
        ),
        (
            """[[gates]]
id = "bad"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "manual_review"
on_missing_capability = "skip"
""",
            "on_missing_capability",
        ),
        (
            """[[comparisons]]
id = "bad"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "hybrid", physics_profile = "legacy" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "cv"
threshold = 0.7
""",
            "min_pairs",
        ),
    ],
)
def test_rule_schema_is_closed_and_operator_specific(tmp_path, rule, match):
    api = _api()

    with pytest.raises(api.ManifestError, match=match):
        api.load_manifest(_write(tmp_path, BASE_MANIFEST + rule))


@pytest.mark.parametrize(
    "text",
    [
        BASE_MANIFEST.replace('kind = "synthetic"', "kind = []", 1),
        BASE_MANIFEST.replace('truth = "object_truth"', "truth = {}", 1),
        BASE_MANIFEST
        + """
[[gates]]
id = "bad_type"
target = { architecture = "hybrid" }
operator = []
""",
        BASE_MANIFEST
        + """
[[gates]]
id = "bad_type"
target = { architecture = "hybrid" }
operator = "ge"
metric = {}
aggregation = "median"
threshold = 0.5
min_successful = 1
""",
        BASE_MANIFEST
        + """
[[gates]]
id = "bad_type"
target = { architecture = "hybrid" }
operator = "ge"
metric = "truth_quality.amp_pearson"
aggregation = []
threshold = 0.5
min_successful = 1
""",
        BASE_MANIFEST
        + """
[[gates]]
id = "bad_type"
target = { architecture = "hybrid" }
operator = "manual_review"
when_dataset_kind = []
""",
        BASE_MANIFEST
        + """
[[gates]]
id = "bad_type"
target = { architecture = "hybrid" }
operator = "manual_review"
on_missing_capability = {}
""",
        BASE_MANIFEST
        + """
[[gates]]
id = "bad_type"
target = { architecture = "hybrid" }
operator = "manual_review"
requires = {}
""",
        BASE_MANIFEST
        + """
[[gates]]
id = "bad_type"
target = { architecture = ["hybrid"] }
operator = "manual_review"
""",
        BASE_MANIFEST
        + """
[[gates]]
id = "bad_type"
target = { architecture = "hybrid" }
operator = "status_count_ge"
status = []
threshold = 1
requested = 2
""",
        BASE_MANIFEST
        + """
[[comparisons]]
id = "bad_type"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "cnn", physics_profile = "ci" }
operator = []
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.5
min_pairs = 1
""",
        BASE_MANIFEST
        + """
[[comparisons]]
id = "bad_type"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "cnn", physics_profile = "ci" }
operator = "paired_ratio_ge"
metric = []
aggregation = "median"
threshold = 0.5
min_pairs = 1
""",
        BASE_MANIFEST
        + """
[[comparisons]]
id = "bad_type"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "cnn", physics_profile = "ci" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = {}
threshold = 0.5
min_pairs = 1
""",
    ],
)
def test_closed_fields_reject_malformed_types_as_manifest_errors(tmp_path, text):
    api = _api()

    with pytest.raises(api.ManifestError):
        api.load_manifest(_write(tmp_path, text))


def test_rejects_duplicate_gate_and_comparison_ids(tmp_path):
    api = _api()
    duplicate = VALID_RULES.replace('id = "relative"', 'id = "successes"')

    with pytest.raises(api.ManifestError, match="duplicate.*successes"):
        api.load_manifest(_write(tmp_path, BASE_MANIFEST + duplicate))


@pytest.mark.parametrize(
    "target, match",
    [
        ('{ architecture = "missing" }', "unknown.*value"),
        ('{ unknown = "hybrid" }', "unknown.*dimension"),
        ("{ architecture = 4 }", "selector"),
    ],
)
def test_rejects_malformed_rule_selectors(tmp_path, target, match):
    api = _api()
    rule = f"""
[[gates]]
id = "bad_target"
target = {target}
operator = "manual_review"
"""

    with pytest.raises(api.ManifestError, match=match):
        api.load_manifest(_write(tmp_path, BASE_MANIFEST + rule))


def test_rule_target_must_resolve_exactly_one_arm_per_dataset(tmp_path):
    api = _api()
    multiple = """
[[gates]]
id = "multi"
target = { architecture = "hybrid" }
operator = "manual_review"
"""
    with pytest.raises(api.ManifestError, match="multiple|exactly one"):
        api.resolve_manifest(
            api.load_manifest(_write(tmp_path, BASE_MANIFEST + multiple))
        )

    zero = """
[[matrix.exclude]]
architecture = "hybrid"
physics_profile = "legacy"

[[gates]]
id = "zero"
target = { architecture = "hybrid", physics_profile = "legacy" }
operator = "manual_review"
"""
    with pytest.raises(api.ManifestError, match="zero|exactly one"):
        api.resolve_manifest(api.load_manifest(_write(tmp_path, BASE_MANIFEST + zero)))


def test_comparison_targets_must_each_resolve_exactly_one_arm(tmp_path):
    api = _api()
    multiple = """
[[comparisons]]
id = "multi"
left = { architecture = "hybrid" }
right = { architecture = "cnn", physics_profile = "ci" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.7
min_pairs = 1
"""

    with pytest.raises(api.ManifestError, match="multiple|exactly one"):
        api.resolve_manifest(
            api.load_manifest(_write(tmp_path, BASE_MANIFEST + multiple))
        )


@pytest.mark.parametrize(
    "rule",
    [
        """
[[gates]]
id = "all_excluded_gate"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "manual_review"
""",
        """
[[comparisons]]
id = "all_excluded_comparison"
left = { architecture = "hybrid", physics_profile = "ci" }
right = { architecture = "cnn", physics_profile = "ci" }
operator = "paired_ratio_ge"
metric = "truth_quality.amp_pearson"
aggregation = "median"
threshold = 0.5
min_pairs = 1
""",
    ],
)
@pytest.mark.parametrize("filter_dataset_dimension", [False, True])
def test_all_excluded_rules_fail_zero_match_even_after_dataset_filtering(
    tmp_path, rule, filter_dataset_dimension
):
    api = _api()
    manifest_text = BASE_MANIFEST
    selected_dataset = "experimental"
    if filter_dataset_dimension:
        manifest_text = _with_dataset_dimension(manifest_text)
    manifest_text += """
[[matrix.exclude]]
architecture = "hybrid"

[[matrix.exclude]]
architecture = "cnn"
"""
    manifest = api.load_manifest(_write(tmp_path, manifest_text + rule))

    with pytest.raises(api.ManifestError, match="empty.*arm|no.*arm"):
        api.resolve_manifest(manifest, dataset=selected_dataset)


def test_all_excluded_manifest_without_rules_is_rejected(tmp_path):
    api = _api()
    text = (
        BASE_MANIFEST
        + """
[[matrix.exclude]]
architecture = "hybrid"

[[matrix.exclude]]
architecture = "cnn"
"""
    )

    with pytest.raises(api.ManifestError, match="empty.*exclude|no.*arm"):
        api.expand_runs(api.load_manifest(_write(tmp_path, text)))


def test_dataset_filter_producing_no_assignments_is_rejected(tmp_path):
    api = _api()
    text = (
        _with_dataset_dimension(BASE_MANIFEST)
        + """
[[matrix.exclude]]
dataset = "experimental"
"""
    )
    manifest = api.load_manifest(_write(tmp_path, text))

    with pytest.raises(
        api.ManifestError, match="empty.*dataset.*experimental|no.*arm.*experimental"
    ):
        api.expand_runs(manifest, dataset="experimental")


def test_fully_excluded_dataset_is_not_active_when_other_dataset_survives(tmp_path):
    api = _api()
    text = _with_dataset_dimension(BASE_MANIFEST)
    text += """
[[matrix.exclude]]
dataset = "synthetic"

[[gates]]
id = "surviving_dataset"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "manual_review"
"""

    resolved = api.resolve_manifest(api.load_manifest(_write(tmp_path, text)))

    assert {arm.dataset_id for arm in resolved.arms} == {"experimental"}
    assert len(resolved.gates) == 1
    assert resolved.gates[0].dataset_id == "experimental"


def test_surviving_include_does_not_require_pre_exclusion_dataset_resolution(tmp_path):
    api = _api()
    text = BASE_MANIFEST.replace('"dataset.id" = "synthetic"\n', "")
    text += """
[[matrix.exclude]]
architecture = "hybrid"

[[matrix.exclude]]
architecture = "cnn"

[[matrix.include]]
architecture = "hybrid"
physics_profile = "ci"
[matrix.include.overrides]
"dataset.id" = "synthetic"
"""

    resolved = api.resolve_manifest(api.load_manifest(_write(tmp_path, text)))

    assert [arm.id for arm in resolved.arms] == ["compatibility--synthetic--hybrid--ci"]


def test_all_excluded_matrix_retains_pre_exclusion_dataset_context(tmp_path):
    api = _api()
    text = (
        BASE_MANIFEST
        + """
[[matrix.exclude]]
architecture = "hybrid"

[[matrix.exclude]]
architecture = "cnn"

[[gates]]
id = "all_gone"
target = { architecture = "hybrid", physics_profile = "ci" }
operator = "manual_review"
"""
    )

    with pytest.raises(api.ManifestError, match="empty.*arm|no.*arm"):
        api.resolve_manifest(api.load_manifest(_write(tmp_path, text)))


def test_dataset_cli_requires_value_present_in_dataset_dimension(tmp_path):
    api = _api()
    text = _with_dataset_dimension(BASE_MANIFEST).replace(
        '[[matrix.dimensions]]\nname = "dataset"',
        f"""{_closed_manifest_dataset("declared_but_absent", kind="experimental", truth="none")}

[[matrix.dimensions]]
name = "dataset"
""",
        1,
    )
    manifest = api.load_manifest(_write(tmp_path, text))

    with pytest.raises(
        api.ManifestError, match="dataset dimension.*declared_but_absent"
    ):
        api.expand_runs(manifest, dataset="declared_but_absent")


@pytest.mark.parametrize(
    "text",
    [
        BASE_MANIFEST.replace('id = "compatibility"', 'id = "compatibility--seed-3"'),
        BASE_MANIFEST.replace("[datasets.synthetic]", "[datasets.synthetic--seed-3]"),
        BASE_MANIFEST.replace(
            'name = "architecture"', 'name = "architecture--seed-3"', 1
        ),
        BASE_MANIFEST.replace('id = "hybrid"', 'id = "hybrid--seed-3"', 1),
        BASE_MANIFEST.replace('id = "compatibility"', 'id = "seed-3"'),
    ],
)
def test_arm_id_components_reject_reserved_run_namespace_patterns(tmp_path, text):
    api = _api()

    with pytest.raises(api.ManifestError, match="reserved|component"):
        api.load_manifest(_write(tmp_path, text))


def test_generated_arm_and_run_namespaces_are_disjoint(tmp_path):
    api = _api()
    resolved = api.resolve_manifest(api.load_manifest(_write(tmp_path)))

    arm_ids = {arm.id for arm in resolved.arms}
    run_ids = {run.id for run in resolved.runs}

    assert arm_ids.isdisjoint(run_ids)
    assert len(api.select_runs(resolved.runs, resolved.arms[0].id)) == 2
    assert len(api.select_runs(resolved.runs, resolved.runs[0].id)) == 1


def test_exact_run_selection_rejects_ambiguous_arm_and_run_namespace(tmp_path):
    api = _api()
    runs = api.expand_runs(api.load_manifest(_write(tmp_path)))
    ambiguous = (
        replace(runs[0], id="shared-id"),
        replace(runs[1], arm_id="shared-id"),
    )

    with pytest.raises(api.ManifestError, match="ambiguous"):
        api.select_runs(ambiguous, "shared-id")


def test_dataset_narrowed_target_resolves_only_selected_dataset(tmp_path):
    api = _api()
    rule = """
[[gates]]
id = "experimental_manual"
target = { dataset = "experimental", architecture = "hybrid", physics_profile = "ci" }
operator = "manual_review"
"""
    manifest = api.load_manifest(_write(tmp_path, BASE_MANIFEST + rule))

    resolved = api.resolve_manifest(manifest, dataset="experimental")

    assert resolved.gates[0].dataset_id == "experimental"
    with pytest.raises(api.ManifestError, match="selected dataset|zero"):
        api.resolve_manifest(manifest, dataset="synthetic")
