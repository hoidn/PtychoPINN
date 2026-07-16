from __future__ import annotations

import ast
import json
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import lightning as L
import numpy as np
import pytest
import torch

from ptycho.config.config import PyTorchExecutionConfig
import ptycho_torch.config_factory as legacy_config_factory
import ptycho_torch.config_params as config_params
from ptycho_torch.config_params import (
    DataConfig,
    DatagenConfig,
    InferenceConfig,
    ModelConfig,
    TrainingConfig,
)
from ptycho_torch.model import (
    Autoencoder,
    PtychoPINN,
    PtychoPINN_Lightning,
    Ptycho_Supervised,
)
from ptycho_torch.patch_generator import group_coords
from ptycho_torch.train_utils import EncoderFreezeCallback
from ptycho_torch.train_lightning_only import (
    EarlyStopping,
    ModelCheckpoint,
    _resolve_checkpoint_monitor,
    _trainer_accelerator,
    _trainer_strategy,
)
from scripts.studies.ablation.configuration import (
    ALLOWLISTS,
    ARCHITECTURE_POLICIES,
    EXECUTION_PLATFORM_POLICIES,
    EXECUTION_TO_TRAINING_ALIASES,
    NAMESPACE_OWNERS,
    SIMULATION_PATHS,
    TRAINING_TO_EXECUTION_ALIASES,
    ConfigResolutionError,
    ResolvedTorchConfigs,
    resolve_torch_configs,
    resolve_simulation_namespace,
)
from scripts.studies.ablation.datasets import (
    load_checked_dataset,
    load_standalone_dataset,
)
from tests.studies.ablation_dataset_fixtures import (
    _bundle,
    _dose as _fixture_dose,
    _file_sha256 as _fixture_file_sha256,
    _refresh_provenance,
)


def _ci_overrides(*, architecture: str = "hybrid_resnet") -> dict[str, Any]:
    output_field = (
        "model.cnn_output_mode"
        if architecture == "cnn"
        else "model.generator_output_mode"
    )
    return {
        "data.N": 64,
        "data.C": 4,
        "data.grid_size": [2, 2],
        "data.scale_contract_version": "ci_intensity_v2",
        "data.measurement_domain": "count_intensity",
        "model.mode": "Unsupervised",
        "model.architecture": architecture,
        output_field: "real_imag",
        "model.C_model": 4,
        "model.C_forward": 4,
        "model.physics_forward_mode": "rectangular_scaled",
        "model.rect_s1s2_trainable": True,
        "model.loss_function": "Poisson",
        "training.torch_loss_mode": "poisson",
        "training.learning_rate": 2e-4,
        "training.epochs": 3,
        "training.batch_size": 2,
        "training.scheduler": "WarmupCosine",
        "training.lr_warmup_epochs": 1,
        "training.gradient_clip_val": 0.5,
        "training.gradient_clip_algorithm": "norm",
        "training.accum_steps": 2,
        "inference.patch_weighting": "probe",
        "inference.varpro_scaling": True,
        "execution.accelerator": "cpu",
        "execution.devices": 1,
        "execution.strategy": "auto",
        "execution.precision": "32-true",
        "execution.num_workers": 2,
        "execution.pin_memory": False,
        "execution.enable_checkpointing": True,
    }


def test_namespace_registry_has_one_explicit_owner_per_accepted_path() -> None:
    assert NAMESPACE_OWNERS == {
        "dataset": "immutable dataset descriptor",
        "data": DataConfig,
        "model": ModelConfig,
        "training": TrainingConfig,
        "inference": InferenceConfig,
        "execution": PyTorchExecutionConfig,
    }
    accepted = [
        f"{namespace}.{field}"
        for namespace, fields in ALLOWLISTS.items()
        for field in fields
    ]
    assert len(accepted) == len(set(accepted))
    assert "probe_scale" in ALLOWLISTS["data"]
    assert "physics_forward_mode" in ALLOWLISTS["model"]
    assert "torch_loss_mode" in ALLOWLISTS["training"]
    assert "varpro_scaling" in ALLOWLISTS["inference"]
    assert "precision" in ALLOWLISTS["execution"]
    assert "objects_per_probe" not in set().union(*ALLOWLISTS.values())
    assert set(ARCHITECTURE_POLICIES) == {"cnn", "fno", "hybrid_resnet"}
    assert "model.fno_blocks" in ARCHITECTURE_POLICIES["fno"].applicable_paths
    assert "model.fno_blocks" in ARCHITECTURE_POLICIES["hybrid_resnet"].applicable_paths
    with pytest.raises(TypeError):
        ARCHITECTURE_POLICIES["cnn"] = ARCHITECTURE_POLICIES["fno"]  # type: ignore[index]


def test_immutable_simulation_namespace_resolves_recursive_paths_separately() -> None:
    assert "simulation.probe.transform_pipeline" in SIMULATION_PATHS
    assert "simulation.object.kind" in SIMULATION_PATHS
    simulation = resolve_simulation_namespace(
        {
            "simulation.N": 128,
            "simulation.seed": 3,
            "simulation.probe.source": "custom",
            "simulation.probe.source_path": "probe.npz",
            "simulation.probe.transform_pipeline": (
                "smooth:0.5|pad_extrapolate_boundary_matched:128"
            ),
            "simulation.object.kind": "lines",
            "simulation.scan.grid_size": [2, 2],
            "simulation.detector.photons_per_pattern": 1e8,
        }
    )
    assert simulation.N == 128
    assert simulation.scan.grid_size == (2, 2)
    assert simulation.probe.source_path == Path("probe.npz")
    with pytest.raises(ConfigResolutionError, match="simulation.training.epochs"):
        resolve_simulation_namespace({"simulation.training.epochs": 10})


def test_every_allowlisted_config_field_has_a_canonical_consumer_read_site() -> None:
    source_paths = {
        "data": (
            "ptycho_torch/dataloader.py",
            "ptycho_torch/patch_generator.py",
            "ptycho_torch/helper.py",
            "ptycho_torch/model.py",
            "ptycho_torch/scaling_contract.py",
        ),
        "model": (
            "ptycho_torch/model.py",
            "ptycho_torch/helper.py",
            "ptycho_torch/generators",
        ),
        "training": (
            "ptycho_torch/model.py",
            "ptycho_torch/train_lightning_only.py",
            "ptycho_torch/train_utils.py",
            "ptycho_torch/lightning_utils.py",
        ),
        "inference": ("ptycho_torch/reassembly.py",),
        "execution": (
            "ptycho_torch/train_lightning_only.py",
            "ptycho_torch/train_utils.py",
        ),
    }

    for namespace, paths in source_paths.items():
        source_files = []
        for raw_path in paths:
            path = Path(raw_path)
            source_files.extend(sorted(path.glob("*.py")) if path.is_dir() else [path])
        qualified_reads: set[str] = set()
        qualifier = f"{namespace}_config"
        for source_file in source_files:
            tree = ast.parse(source_file.read_text(), filename=str(source_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute):
                    owner = ast.unparse(node.value)
                    if qualifier in owner:
                        qualified_reads.add(node.attr)
                elif (
                    isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == "getattr"
                    and len(node.args) >= 2
                    and qualifier in ast.unparse(node.args[0])
                    and isinstance(node.args[1], ast.Constant)
                    and isinstance(node.args[1].value, str)
                ):
                    qualified_reads.add(node.args[1].value)
        missing = sorted(
            field for field in ALLOWLISTS[namespace] if field not in qualified_reads
        )
        assert not missing, f"{namespace} allowlist fields lack read sites: {missing}"


_REMOVED_INERT_PATHS = (
    "data.nphotons",
    "data.subsample_seed",
    "data.K",
    "data.K_quadrant",
    "data.neighbor_function",
    "data.min_neighbor_distance",
    "data.max_neighbor_distance",
    "data.scan_pattern",
    "model.edge_pad",
    "model.pad_object",
    "model.gaussian_smoothing_sigma",
    "model.eca_encoder",
    "model.eca_decoder",
    "model.cbam_decoder",
    "model.spatial_decoder",
    "model.decoder_spatial_kernel",
    "training.nll",
    "inference.pad_eval",
    "inference.window",
    "inference.log_patch_stats",
    "inference.patch_stats_limit",
    "execution.logger_backend",
    "execution.recon_log_every_n_epochs",
    "execution.recon_log_num_patches",
    "execution.recon_log_fixed_indices",
    "execution.recon_log_stitch",
    "execution.recon_log_max_stitch_samples",
)


@pytest.mark.parametrize("path", _REMOVED_INERT_PATHS)
def test_inert_or_unhonored_controls_are_not_allowlisted(path: str) -> None:
    namespace, field = path.split(".")

    assert field not in ALLOWLISTS[namespace]
    with pytest.raises(ConfigResolutionError, match=rf"{path}.*(?:inert|unsupported)"):
        resolve_torch_configs({path: True})


def test_representative_overrides_construct_configs_and_derive_aliases() -> None:
    overrides = _ci_overrides()
    overrides.update(
        {
            "data.probe_scale": 3,
            "data.probe_normalize": False,
            "model.fno_modes": 10,
            "model.hybrid_resnet_blocks": 4,
            "inference.middle_trim": 16,
            "inference.batch_size": 8,
            "execution.deterministic": "warn",
            "execution.persistent_workers": True,
            "execution.prefetch_factor": 3,
            "execution.checkpoint_save_top_k": 2,
            "execution.checkpoint_monitor_metric": "val_loss",
        }
    )

    resolved = resolve_torch_configs(overrides)

    assert isinstance(resolved, ResolvedTorchConfigs)
    assert resolved.data_config.probe_scale == 3.0
    assert resolved.data_config.probe_normalize is False
    assert resolved.model_config.fno_modes == 10
    assert resolved.training_config.scheduler == "WarmupCosine"
    assert resolved.inference_config.middle_trim == 16
    assert resolved.execution_config.precision == "32-true"
    assert (
        resolved.execution_config.learning_rate
        == resolved.training_config.learning_rate
    )
    assert resolved.execution_config.scheduler == resolved.training_config.scheduler
    assert resolved.execution_config.gradient_clip_val == 0.5
    assert resolved.execution_config.accum_steps == 2
    assert resolved.training_config.device == "cpu"
    assert resolved.training_config.strategy == "auto"
    assert resolved.training_config.n_devices == 1
    assert resolved.training_config.num_workers == 2
    snapshot = resolved.snapshot
    for training_name, execution_name in TRAINING_TO_EXECUTION_ALIASES.items():
        assert (
            snapshot["training"][training_name] == snapshot["execution"][execution_name]
        )
    for execution_name, training_name in EXECUTION_TO_TRAINING_ALIASES.items():
        assert (
            snapshot["execution"][execution_name] == snapshot["training"][training_name]
        )
    assert not (
        set(snapshot["model"])
        & {
            "accelerator",
            "devices",
            "strategy",
            "precision",
            "num_workers",
        }
    )


@pytest.mark.parametrize(
    ("loss_mode", "expected_monitor"),
    [("poisson", "poisson_val_loss"), ("mae", "mae_val_loss")],
)
def test_checkpoint_sentinel_resolves_to_emitted_metric_and_builds_callbacks(
    tmp_path: Path,
    loss_mode: str,
    expected_monitor: str,
) -> None:
    overrides = (
        _ci_overrides()
        if loss_mode == "poisson"
        else _legacy_overrides(loss_mode="mae")
    )
    overrides["execution.checkpoint_monitor_metric"] = "val_loss"
    resolved = resolve_torch_configs(overrides)
    model = PtychoPINN_Lightning(
        resolved.model_config,
        resolved.data_config,
        resolved.training_config,
        resolved.inference_config,
    )

    monitor = _resolve_checkpoint_monitor(resolved.execution_config, model)
    checkpoint = ModelCheckpoint(
        dirpath=str(tmp_path),
        monitor=monitor,
        mode=resolved.execution_config.checkpoint_mode,
        save_top_k=resolved.execution_config.checkpoint_save_top_k,
    )
    early_stopping = EarlyStopping(
        monitor=monitor,
        mode=resolved.execution_config.checkpoint_mode,
        patience=resolved.execution_config.early_stop_patience,
        strict=True,
    )

    assert model.val_loss_name == expected_monitor
    assert checkpoint.monitor == expected_monitor
    assert early_stopping.monitor == expected_monitor


def test_checkpoint_monitor_rejects_unproven_custom_metric() -> None:
    overrides = _ci_overrides()
    overrides["execution.checkpoint_monitor_metric"] = "poisson_val"

    with pytest.raises(
        ConfigResolutionError,
        match=r"checkpoint_monitor_metric.*val_loss",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("execution.enable_checkpointing", False, "enable_checkpointing.*true"),
        ("execution.checkpoint_save_top_k", 0, "checkpoint_save_top_k.*at least 1"),
        ("execution.checkpoint_mode", "max", "checkpoint_mode.*min"),
    ],
)
def test_claim_grade_checkpoint_contract_rejects_unsafe_settings(
    tmp_path: Path,
    path: str,
    value: object,
    message: str,
) -> None:
    overrides = _claim_grade_overrides("fno")
    overrides[path] = value

    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(
            overrides,
            dataset=_validated_fixture(tmp_path),
            require_all_explicit=True,
        )


def test_generic_checkpoint_contract_retains_runtime_supported_flexibility() -> None:
    disabled = _ci_overrides()
    disabled["execution.enable_checkpointing"] = False
    resolved_disabled = resolve_torch_configs(disabled)
    assert resolved_disabled.execution_config.enable_checkpointing is False

    maximizing = _ci_overrides()
    maximizing.update(
        {
            "execution.checkpoint_save_top_k": 0,
            "execution.checkpoint_mode": "max",
            "execution.checkpoint_monitor_metric": "val_loss",
        }
    )
    resolved_max = resolve_torch_configs(maximizing)
    assert resolved_max.execution_config.checkpoint_save_top_k == 0
    assert resolved_max.execution_config.checkpoint_mode == "max"


def test_execution_platform_policy_is_closed_and_immutable() -> None:
    assert set(EXECUTION_PLATFORM_POLICIES) == {"cpu", "cuda", "gpu"}
    assert EXECUTION_PLATFORM_POLICIES["cpu"].strategies == {"auto", "ddp"}
    assert all(
        policy.requires_single_device
        for policy in EXECUTION_PLATFORM_POLICIES.values()
    )
    assert EXECUTION_PLATFORM_POLICIES["cpu"].precisions == {
        "32-true",
        "bf16-mixed",
    }
    with pytest.raises(TypeError):
        EXECUTION_PLATFORM_POLICIES["cpu"] = EXECUTION_PLATFORM_POLICIES["cuda"]  # type: ignore[index]


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"execution.strategy": "fsdp"}, r"execution\.strategy='fsdp'.*unsupported"),
        (
            {"execution.strategy": "deepspeed"},
            r"execution\.strategy='deepspeed'.*unsupported",
        ),
        (
            {"execution.accelerator": "cpu", "execution.precision": "16-mixed"},
            r"execution\.precision='16-mixed'.*cpu.*rewrites",
        ),
        (
            {"execution.accelerator": "mps", "execution.devices": 2},
            r"execution\.accelerator='mps'.*unsupported.*float64.*reassembly.*count",
        ),
        (
            {"execution.accelerator": "mps", "execution.devices": "auto"},
            r"execution\.accelerator='mps'.*unsupported.*float64.*reassembly.*count",
        ),
        (
            {"execution.accelerator": "mps", "execution.strategy": "ddp"},
            r"execution\.accelerator='mps'.*unsupported.*float64.*reassembly.*count",
        ),
        (
            {"execution.accelerator": "mps", "execution.precision": "bf16-mixed"},
            r"execution\.accelerator='mps'.*unsupported.*float64.*reassembly.*count",
        ),
        (
            {"execution.accelerator": "cpu", "execution.strategy": "ddp_spawn"},
            r"execution\.strategy='ddp_spawn'.*unsupported",
        ),
    ],
)
def test_execution_platform_policy_rejects_unproven_combinations(
    updates: dict[str, object],
    message: str,
) -> None:
    overrides = _ci_overrides()
    overrides.update(updates)

    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("accelerator", ["cuda", "gpu"])
@pytest.mark.parametrize("precision", ["32-true", "16-mixed", "bf16-mixed"])
@pytest.mark.parametrize("strategy", ["auto", "ddp"])
def test_cuda_family_policy_accepts_one_device_without_allocating_gpu(
    accelerator: str,
    precision: str,
    strategy: str,
) -> None:
    overrides = _ci_overrides()
    overrides.update(
        {
            "execution.accelerator": accelerator,
            "execution.devices": 1,
            "execution.strategy": strategy,
            "execution.precision": precision,
        }
    )

    resolved = resolve_torch_configs(overrides)

    assert resolved.execution_config.accelerator == accelerator
    assert resolved.execution_config.precision == precision


@pytest.mark.parametrize("accelerator", ["cpu", "cuda", "gpu"])
@pytest.mark.parametrize("devices", [2, "auto"])
def test_canonical_ablation_rejects_non_single_device_execution(
    accelerator: str,
    devices: int | str,
) -> None:
    overrides = _ci_overrides()
    overrides.update(
        {
            "execution.accelerator": accelerator,
            "execution.devices": devices,
        }
    )

    with pytest.raises(
        ConfigResolutionError,
        match=(
            r"canonical ablation.*execution\.devices=1.*held-out mmap/reassembly.*"
            r"framework peak-memory evidence.*single-device-only"
        ),
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("strategy", ["auto", "ddp"])
def test_canonical_cpu_accepts_one_device_supported_strategies(strategy: str) -> None:
    overrides = _ci_overrides()
    overrides["execution.strategy"] = strategy

    resolved = resolve_torch_configs(overrides)

    assert resolved.execution_config.devices == 1
    assert resolved.execution_config.strategy == strategy


def test_generic_execution_config_remains_multi_device_capable() -> None:
    generic = PyTorchExecutionConfig(
        accelerator="cuda",
        devices=2,
        strategy="ddp",
        precision="32-true",
    )

    assert generic.devices == 2
    assert generic.strategy == "ddp"


def test_ablation_rejects_mps_but_generic_execution_config_retains_it() -> None:
    generic = PyTorchExecutionConfig(
        accelerator="mps",
        devices=1,
        strategy="auto",
        precision="32-true",
    )
    assert generic.accelerator == "mps"

    overrides = _ci_overrides()
    overrides.update(
        {
            "execution.accelerator": "mps",
            "execution.devices": 1,
            "execution.strategy": "auto",
            "execution.precision": "32-true",
        }
    )

    with pytest.raises(
        ConfigResolutionError,
        match=r"execution\.accelerator='mps'.*unsupported.*float64.*reassembly.*count",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("cuda_available", [False, True])
def test_auto_accelerator_is_validated_after_execution_config_resolution(
    monkeypatch: pytest.MonkeyPatch,
    cuda_available: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda_available)
    overrides = _ci_overrides()
    overrides["execution.accelerator"] = "auto"

    resolved = resolve_torch_configs(overrides)

    assert resolved.execution_config.accelerator == (
        "cuda" if cuda_available else "cpu"
    )


def test_auto_accelerator_rejects_precision_after_resolving_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    overrides = _ci_overrides()
    overrides.update(
        {
            "execution.accelerator": "auto",
            "execution.precision": "16-mixed",
        }
    )

    with pytest.raises(
        ConfigResolutionError,
        match=r"execution\.precision='16-mixed'.*cpu.*rewrites",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("strategy", ["auto", "ddp"])
@pytest.mark.parametrize("precision", ["32-true", "bf16-mixed"])
def test_claim_cpu_platform_combinations_construct_exact_task1_trainer(
    tmp_path: Path,
    strategy: str,
    precision: str,
) -> None:
    overrides = _claim_grade_overrides("fno")
    overrides.update(
        {
            "execution.accelerator": "cpu",
            "execution.devices": 1,
            "execution.strategy": strategy,
            "execution.precision": precision,
        }
    )
    resolved = resolve_torch_configs(
        overrides,
        dataset=_validated_fixture(tmp_path),
        require_all_explicit=True,
    )

    trainer = L.Trainer(
        accelerator=_trainer_accelerator(resolved.execution_config.accelerator),
        devices=resolved.execution_config.devices,
        strategy=_trainer_strategy(
            resolved.execution_config.strategy,
            resolved.execution_config.devices,
            resolved.execution_config.accelerator,
        ),
        precision=resolved.execution_config.precision,
        max_epochs=0,
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=resolved.execution_config.enable_checkpointing,
        enable_progress_bar=resolved.execution_config.enable_progress_bar,
        deterministic=resolved.execution_config.deterministic,
        enable_model_summary=False,
    )

    assert trainer.precision_plugin.precision == precision
    assert trainer.num_devices == 1
    if strategy == "ddp":
        assert type(trainer.strategy).__name__ == "DDPStrategy"
    else:
        assert type(trainer.strategy).__name__ == "SingleDeviceStrategy"


def test_runtime_compatibility_fields_are_effective_before_snapshot() -> None:
    resolved = resolve_torch_configs(_ci_overrides())

    assert resolved.training_config.framework == "Lightning"
    assert resolved.training_config.orchestrator == "Lightning"
    assert resolved.snapshot["training"]["orchestrator"] == "Lightning"
    resolved.training_config.n_devices = resolved.execution_config.devices
    resolved.training_config.strategy = resolved.execution_config.strategy
    resolved.training_config.device = resolved.execution_config.accelerator
    resolved.training_config.num_workers = resolved.execution_config.num_workers
    resolved.training_config.orchestrator = "Lightning"
    resolved.validate_integrity()


@pytest.mark.parametrize(
    "path",
    [
        "dataset.kind",
        "datagen.objects_per_probe",
        "model.intensity_scale",
        "model.intensity_scale_trainable",
        "model.max_position_jitter",
        "model.hybrid_encoder_conv_hidden_channels",
        "training.training_directories",
        "training.output_dir",
        "training.device",
        "training.num_workers",
        "execution.learning_rate",
        "execution.gradient_clip_val",
        "execution.middle_trim",
        "execution.hybrid_resnet_blocks",
        "inference.experiment_number",
        "model.gridsize",
        "unknown.value",
    ],
)
def test_rejects_non_owned_derived_inert_and_unknown_paths(path: str) -> None:
    with pytest.raises(ConfigResolutionError, match=path.replace(".", r"\.")):
        resolve_torch_configs({path: 1})


def test_typo_error_suggests_close_allowlisted_path() -> None:
    with pytest.raises(
        ConfigResolutionError, match=r"data\.probe_scal.*data\.probe_scale"
    ):
        resolve_torch_configs({"data.probe_scal": 4.0})


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("data.C", True, "exact integer"),
        ("data.C", 4.0, "exact integer"),
        ("data.probe_scale", True, "number"),
        ("data.probe_scale", "4", "number"),
        ("data.probe_normalize", 1, "boolean"),
        ("data.probe_normalize", "false", "boolean"),
        ("data.grid_size", [2], "2 items"),
        ("data.grid_size", [2, True], r"data\.grid_size\[1\]"),
        ("model.mode", "unsupervised", "expected one of"),
        ("training.gradient_clip_val", "none", "number or null"),
    ],
)
def test_type_coercion_rejects_stringly_and_bool_numeric_values(
    path: str,
    value: object,
    message: str,
) -> None:
    overrides = _ci_overrides()
    overrides[path] = value
    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


def test_toml_arrays_coerce_to_declared_tuples_and_int_to_float() -> None:
    overrides = _ci_overrides()
    overrides.update(
        {
            "data.grid_size": [1, 4],
            "data.x_bounds": [0, 1],
            "data.probe_scale": 5,
        }
    )

    resolved = resolve_torch_configs(overrides)

    assert resolved.data_config.grid_size == (1, 4)
    assert resolved.data_config.x_bounds == (0.0, 1.0)
    assert resolved.data_config.probe_scale == 5.0


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.grid_size", np.asarray([2, 2])),
        ("data.grid_size", object()),
    ],
)
def test_non_json_native_override_values_fail_before_sentinel_comparison(
    path: str,
    value: object,
) -> None:
    overrides = _ci_overrides()
    overrides[path] = value

    with pytest.raises(ConfigResolutionError, match=rf"{path}.*JSON-native"):
        resolve_torch_configs(overrides)


def test_numeric_overflow_is_normalized_to_path_specific_resolution_error() -> None:
    overrides = _ci_overrides()
    overrides["data.probe_scale"] = 10**10000

    with pytest.raises(ConfigResolutionError, match=r"data\.probe_scale.*finite"):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("value", [0.0, -1.0, float("inf"), float("nan")])
def test_gradient_clip_requires_positive_finite_or_disabled(value: float) -> None:
    overrides = _ci_overrides()
    overrides["training.gradient_clip_val"] = value

    with pytest.raises(
        ConfigResolutionError,
        match=r"training\.gradient_clip_val.*(positive|finite)",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"data.N": 0}, "data.N must be positive"),
        ({"data.C": 3}, "data.C must equal data.grid_size"),
        ({"model.C_model": 3}, "model.C_model.*data.C"),
        ({"model.C_forward": 3}, "model.C_forward.*data.C"),
        ({"training.epochs": 0}, "training.epochs must be positive"),
        ({"training.learning_rate": 0.0}, "training.learning_rate must be positive"),
        ({"training.lr_warmup_epochs": 4}, "lr_warmup_epochs.*epochs"),
        ({"execution.devices": 0}, "execution.devices"),
        ({"execution.num_workers": -1}, "execution.num_workers"),
    ],
)
def test_rejects_geometry_and_numeric_invariant_violations(
    updates: dict[str, object],
    message: str,
) -> None:
    overrides = _ci_overrides()
    overrides.update(updates)
    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("architecture", ["hybrid_resnet", "cnn", "fno"])
def test_same_mapping_shape_resolves_multiple_architecture_families(
    architecture: str,
) -> None:
    resolved = resolve_torch_configs(_ci_overrides(architecture=architecture))

    assert resolved.model_config.architecture == architecture
    assert resolved.ci_scaling_active is True


def test_datagen_is_exact_fixed_default_and_existing_tuple_order_is_exact() -> None:
    resolved = resolve_torch_configs(_ci_overrides())

    assert asdict(resolved.datagen_config) == asdict(DatagenConfig())
    assert resolved.existing_config == (
        resolved.data_config,
        resolved.model_config,
        resolved.training_config,
        resolved.inference_config,
        resolved.datagen_config,
    )


def _legacy_overrides(*, loss_mode: str = "poisson") -> dict[str, Any]:
    poisson = loss_mode == "poisson"
    return {
        "data.scale_contract_version": "legacy_v1",
        "data.measurement_domain": "normalized_amplitude",
        "model.mode": "Unsupervised",
        "model.architecture": "hybrid_resnet",
        "model.generator_output_mode": "real_imag",
        "model.physics_forward_mode": "amplitude",
        "model.loss_function": "Poisson" if poisson else "MAE",
        "training.torch_loss_mode": loss_mode,
        "inference.patch_weighting": "probe",
        "inference.varpro_scaling": False,
        "execution.accelerator": "cpu",
        "execution.devices": 1,
        "execution.strategy": "auto",
    }


def test_ci_activation_requires_all_four_effective_predicates() -> None:
    assert resolve_torch_configs(_ci_overrides()).ci_scaling_active is True

    amplitude = _ci_overrides()
    amplitude["model.physics_forward_mode"] = "amplitude"
    amplitude.pop("model.rect_s1s2_trainable")
    assert resolve_torch_configs(amplitude).ci_scaling_active is False

    legacy = resolve_torch_configs(_legacy_overrides())
    assert legacy.ci_scaling_active is False
    assert legacy.profile.version == "legacy_v1"


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {
                "model.mode": "Supervised",
                "model.loss_function": "MAE",
                "training.torch_loss_mode": "mae",
            },
            "rectangular_scaled.*Unsupervised.*Poisson",
        ),
        (
            {
                "model.loss_function": "MAE",
                "training.torch_loss_mode": "mae",
            },
            "rectangular_scaled.*Poisson",
        ),
        ({"inference.varpro_scaling": False}, "inference.varpro_scaling"),
        ({"inference.patch_weighting": "uniform"}, "inference.patch_weighting"),
        ({"model.rect_s1s2_trainable": False}, "model.rect_s1s2_trainable"),
        ({"model.generator_output_mode": "amp_phase"}, "model.generator_output_mode"),
    ],
)
def test_ci_rejects_inactive_loss_and_inference_contradictions(
    updates: dict[str, object],
    message: str,
) -> None:
    overrides = _ci_overrides()
    overrides.update(updates)
    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


def test_cnn_ci_requires_cnn_output_field_not_generator_output_field() -> None:
    overrides = _ci_overrides(architecture="cnn")
    overrides["model.cnn_output_mode"] = "amp_phase"

    with pytest.raises(ConfigResolutionError, match="model.cnn_output_mode"):
        resolve_torch_configs(overrides)


def test_supervised_cnn_rejects_ignored_real_imag_output_control() -> None:
    overrides = _ci_overrides(architecture="cnn")
    overrides.update(
        {
            "model.mode": "Supervised",
            "model.physics_forward_mode": "amplitude",
            "model.loss_function": "MAE",
            "training.torch_loss_mode": "mae",
            "inference.varpro_scaling": False,
        }
    )
    overrides.pop("model.rect_s1s2_trainable")

    with pytest.raises(
        ConfigResolutionError,
        match=r"model.cnn_output_mode.*non-applicable",
    ):
        resolve_torch_configs(overrides)

    overrides.pop("model.cnn_output_mode")
    overrides["model.amp_activation"] = "sigmoid"
    resolved = resolve_torch_configs(overrides)
    model = Ptycho_Supervised(
        resolved.model_config,
        resolved.data_config,
        resolved.training_config,
    )
    assert model.generator_output == "amp_phase"


def test_amplitude_mode_with_default_ci_metadata_is_inactive() -> None:
    resolved = resolve_torch_configs(
        {
            "model.physics_forward_mode": "amplitude",
            "execution.accelerator": "cpu",
        }
    )

    assert resolved.profile.version == "ci_intensity_v2"
    assert resolved.profile.measurement_domain == "count_intensity"
    assert resolved.ci_scaling_active is False


@pytest.mark.parametrize(
    ("path", "value"),
    [("data.normalize", "Group"), ("data.data_scaling", "Max")],
)
def test_ci_rejects_legacy_normalization_controls(path: str, value: str) -> None:
    overrides = _ci_overrides()
    overrides[path] = value

    with pytest.raises(ConfigResolutionError, match=rf"{path}.*non-applicable"):
        resolve_torch_configs(overrides)


def test_explicit_legacy_profile_keeps_effective_normalization_controls() -> None:
    overrides = _legacy_overrides()
    overrides.update({"data.normalize": "Group", "data.data_scaling": "Max"})

    resolved = resolve_torch_configs(overrides)

    assert resolved.data_config.normalize == "Group"
    assert resolved.data_config.data_scaling == "Max"


def test_explicit_legacy_profile_accepts_auditable_disabled_rectangular_scales() -> None:
    overrides = _legacy_overrides()
    overrides["model.rect_s1s2_trainable"] = False

    resolved = resolve_torch_configs(overrides)

    assert resolved.model_config.rect_s1s2_trainable is False


def test_amplitude_physics_gain_override_propagates_nondefault_to_snapshot() -> None:
    overrides = _legacy_overrides()
    overrides["model.amplitude_physics_gain"] = 4.0

    resolved = resolve_torch_configs(overrides)

    assert resolved.model_config.amplitude_physics_gain == 4.0
    assert resolved.snapshot["model"]["amplitude_physics_gain"] == 4.0


def test_ci_rejects_nonunit_amplitude_physics_gain_override() -> None:
    overrides = _ci_overrides()
    overrides["model.amplitude_physics_gain"] = 4.0

    with pytest.raises(
        ConfigResolutionError,
        match=r"amplitude_physics_gain must be 1\.0.*rectangular",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("loss_mode", ["poisson", "mae"])
def test_explicit_legacy_pair_uses_amplitude_without_varpro(loss_mode: str) -> None:
    resolved = resolve_torch_configs(_legacy_overrides(loss_mode=loss_mode))

    assert resolved.model_config.physics_forward_mode == "amplitude"
    assert resolved.inference_config.varpro_scaling is False
    assert resolved.training_config.nll is (loss_mode == "poisson")
    assert resolved.ci_scaling_active is False


@pytest.mark.parametrize(
    "updates",
    [
        {"data.measurement_domain": "count_intensity"},
        {"model.rect_s1s2_trainable": True},
        {"inference.varpro_scaling": True},
        {"model.physics_forward_mode": "rectangular_scaled"},
    ],
)
def test_legacy_rejects_mixed_or_ci_only_behavior(updates: dict[str, object]) -> None:
    overrides = _legacy_overrides()
    overrides.update(updates)
    with pytest.raises(ConfigResolutionError):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize(
    "updates",
    [
        {"model.loss_function": "MAE"},
        {"training.torch_loss_mode": "mae"},
    ],
)
def test_primary_loss_nll_and_model_loss_must_agree(
    updates: dict[str, object],
) -> None:
    overrides = _ci_overrides()
    overrides.update(updates)
    with pytest.raises(ConfigResolutionError, match="must agree"):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize(
    ("loss_mode", "expected_nll"),
    [("poisson", True), ("mae", False)],
)
def test_training_nll_is_derived_from_primary_loss(
    loss_mode: str,
    expected_nll: bool,
) -> None:
    resolved = resolve_torch_configs(_legacy_overrides(loss_mode=loss_mode))

    assert resolved.training_config.nll is expected_nll
    assert resolved.snapshot["training"]["nll"] is expected_nll


def test_json_numeric_probe_masks_resolve_but_tensor_values_do_not() -> None:
    overrides = _ci_overrides()
    mask = [[1 if row == column else 0 for column in range(64)] for row in range(64)]
    overrides["model.probe_mask"] = mask

    resolved = resolve_torch_configs(overrides)

    assert resolved.model_config.probe_mask == mask
    tensor_overrides = _ci_overrides()
    tensor_overrides["model.probe_mask"] = True
    tensor_overrides["model.probe_mask_tensor"] = mask
    tensor_resolved = resolve_torch_configs(tensor_overrides)
    assert tensor_resolved.model_config.probe_mask_tensor == mask
    with pytest.raises(ConfigResolutionError, match="rejects tensor values"):
        resolve_torch_configs({"model.probe_mask": torch.ones(2, 2)})


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("model.probe_mask", 1.0),
        ("model.probe_mask_tensor", True),
        ("model.probe_mask", [[1, 0], [1]]),
        ("model.probe_mask", [[1, float("nan")]]),
    ],
)
def test_probe_masks_reject_unsupported_json_shapes(field: str, value: object) -> None:
    with pytest.raises(ConfigResolutionError, match=field.replace(".", r"\.")):
        resolve_torch_configs({field: value})


def test_cli_epochs_and_output_root_are_last_and_seed_is_not_a_config_path(
    tmp_path: Path,
) -> None:
    resolved = resolve_torch_configs(
        _ci_overrides(), epochs=9, output_root=tmp_path / "run"
    )

    assert resolved.training_config.epochs == 9
    assert resolved.training_config.output_dir == str(tmp_path / "run")
    with pytest.raises(ConfigResolutionError, match="training.output_dir"):
        resolve_torch_configs({"training.output_dir": "manifest-owned"})
    with pytest.raises(ConfigResolutionError, match="seed"):
        resolve_torch_configs({"training.seed": 7})


def test_resolver_never_calls_legacy_factories_or_mutating_updaters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(*_args: object, **_kwargs: object) -> None:
        pytest.fail("legacy config mutation/factory path was called")

    monkeypatch.setattr(config_params, "update_existing_config", forbidden)
    monkeypatch.setattr(legacy_config_factory, "resolve_profile_overrides", forbidden)
    monkeypatch.setattr(legacy_config_factory, "create_training_payload", forbidden)

    resolve_torch_configs(_ci_overrides())


def test_snapshot_is_complete_compact_sorted_and_roundtrips() -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    snapshot = resolved.snapshot

    for namespace, config in (
        ("data", resolved.data_config),
        ("model", resolved.model_config),
        ("training", resolved.training_config),
        ("inference", resolved.inference_config),
        ("datagen", resolved.datagen_config),
        ("execution", resolved.execution_config),
    ):
        assert set(snapshot[namespace]) == {field.name for field in fields(config)}
    assert json.loads(resolved.canonical_json) == snapshot
    assert (
        json.dumps(snapshot, allow_nan=False, separators=(",", ":"), sort_keys=True)
        == resolved.canonical_json
    )
    assert " " not in resolved.canonical_json
    assert (
        snapshot["execution"]["learning_rate"] == snapshot["training"]["learning_rate"]
    )
    assert snapshot["training"]["num_workers"] == snapshot["execution"]["num_workers"]


def test_snapshot_copy_isolated_and_later_config_mutation_is_detected() -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    snapshot = resolved.snapshot
    snapshot["data"]["C"] = 99

    assert resolved.snapshot["data"]["C"] == 4
    resolved.data_config.C = 99
    with pytest.raises(ConfigResolutionError, match="mutated"):
        resolved.validate_integrity()
    with pytest.raises(ConfigResolutionError, match="mutated"):
        resolved.existing_config


@pytest.mark.parametrize(
    ("config_name", "field_name", "mutated_value"),
    [
        ("data_config", "C", 99),
        ("model_config", "C_model", 99),
        ("training_config", "epochs", 99),
        ("inference_config", "batch_size", 99),
        ("datagen_config", "objects_per_probe", 99),
        ("execution_config", "num_workers", 99),
    ],
)
@pytest.mark.parametrize(
    "accessor",
    ["snapshot", "to_jsonable", "canonical_json", "to_json", "existing_config"],
)
def test_every_public_serialization_path_rejects_each_live_config_mutation(
    config_name: str,
    field_name: str,
    mutated_value: object,
    accessor: str,
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    setattr(getattr(resolved, config_name), field_name, mutated_value)

    with pytest.raises(ConfigResolutionError, match="mutated"):
        value = getattr(resolved, accessor)
        if callable(value):
            value()


@pytest.mark.parametrize(
    "accessor",
    ["snapshot", "to_jsonable", "canonical_json", "to_json", "existing_config"],
)
def test_every_public_serialization_path_normalizes_recursive_mutation(
    accessor: str,
) -> None:
    resolved = resolve_torch_configs(_ci_overrides())
    recursive_mask: list[object] = []
    recursive_mask.append(recursive_mask)
    resolved.model_config.probe_mask = recursive_mask

    with pytest.raises(
        ConfigResolutionError,
        match=r"model\.probe_mask.*recursive.*JSON",
    ):
        value = getattr(resolved, accessor)
        if callable(value):
            value()


def test_claim_grade_mode_rejects_implicit_allowlisted_defaults(
    tmp_path: Path,
) -> None:
    overrides = _claim_grade_overrides("fno")
    overrides.pop("data.C")

    with pytest.raises(ConfigResolutionError, match=r"missing:.*data\.C"):
        resolve_torch_configs(
            overrides,
            dataset=_validated_fixture(tmp_path),
            require_all_explicit=True,
        )


def _validated_fixture(
    root: Path,
    *,
    kind: str = "synthetic",
    truth: str = "object_truth",
    domain: str = "count_intensity",
    detector_size: int = 64,
    standalone: bool = False,
):
    probe = np.ones((detector_size, detector_size), dtype=np.complex64)
    descriptor = _bundle(
        root,
        kind=kind,
        truth=truth,
        domain=domain,
        probe_array=probe,
    )
    descriptor["detector_shape"] = [detector_size, detector_size]
    measurement = np.full(
        (25, detector_size, detector_size),
        250_000 if domain == "count_intensity" else 0.5,
        dtype=np.uint32 if domain == "count_intensity" else np.float32,
    )
    for split in ("train", "test"):
        path = root / descriptor[split]
        with np.load(path, allow_pickle=False) as archive:
            payload = {key: archive[key] for key in archive.files}
        payload[descriptor["measurement_key"]] = measurement.copy()
        np.savez(path, **payload)
        descriptor[f"{split}_sha256"] = _fixture_file_sha256(path)
    if domain == "count_intensity":
        descriptor["dose"] = {
            "train": _fixture_dose(measurement),
            "test": _fixture_dose(measurement),
        }
    _refresh_provenance(root, descriptor)
    dataset_id = descriptor.pop("_id")
    if standalone:
        path = root / "dataset.toml"
        dose = descriptor["dose"]
        path.write_text(
            f'''[schema]
version = 1

[dataset]
id = "{dataset_id}"
kind = "{descriptor["kind"]}"
format = "npz_mmap"
scale_contract_version = "{descriptor["scale_contract_version"]}"
measurement_domain = "{descriptor["measurement_domain"]}"
truth = "{descriptor["truth"]}"
truth_location = "{descriptor["truth_location"]}"
truth_key = "{descriptor["truth_key"]}"
measurement_key = "{descriptor["measurement_key"]}"
probe_key = "{descriptor["probe_key"]}"
x_key = "{descriptor["x_key"]}"
y_key = "{descriptor["y_key"]}"
coords_convention = "xy_pixels"
detector_shape = [{detector_size}, {detector_size}]
grouping_max_C = {descriptor["grouping_max_C"]}
probe_modes = {descriptor["probe_modes"]}
train = "{descriptor["train"]}"
test = "{descriptor["test"]}"
reference = "{descriptor["reference"]}"
provenance = "{descriptor["provenance"]}"
train_sha256 = "{descriptor["train_sha256"]}"
test_sha256 = "{descriptor["test_sha256"]}"
reference_sha256 = "{descriptor["reference_sha256"]}"
provenance_sha256 = "{descriptor["provenance_sha256"]}"

[dataset.probe]
source = "{descriptor["probe"]["source"]}"
calibration = "{descriptor["probe"]["calibration"]}"
gauge = "{descriptor["probe"]["gauge"]}"
mask_policy = "{descriptor["probe"]["mask_policy"]}"
sha256 = "{descriptor["probe"]["sha256"]}"

[dataset.dose.train]
counts_mean = {dose["train"]["counts_mean"]}
photons_per_image_min = {dose["train"]["photons_per_image_min"]}
photons_per_image_mean = {dose["train"]["photons_per_image_mean"]}
max_observed_count = {dose["train"]["max_observed_count"]}
dtype_max = {dose["train"]["dtype_max"]}
saturation_fraction = {dose["train"]["saturation_fraction"]}

[dataset.dose.test]
counts_mean = {dose["test"]["counts_mean"]}
photons_per_image_min = {dose["test"]["photons_per_image_min"]}
photons_per_image_mean = {dose["test"]["photons_per_image_mean"]}
max_observed_count = {dose["test"]["max_observed_count"]}
dtype_max = {dose["test"]["dtype_max"]}
saturation_fraction = {dose["test"]["saturation_fraction"]}
''',
            encoding="utf-8",
        )
        return load_standalone_dataset(path)
    return load_checked_dataset(dataset_id, descriptor, repo_root=root)


def test_resolver_validates_synthetic_dataset_requirements_via_task4(
    tmp_path: Path,
) -> None:
    validated = _validated_fixture(tmp_path)
    overrides = _ci_overrides()
    overrides["dataset.id"] = validated.descriptor.id

    resolved = resolve_torch_configs(
        overrides,
        dataset=validated,
        required_capabilities=("has_object_truth", "supports_count_metrics"),
    )

    assert resolved.dataset_id == validated.descriptor.id
    assert resolved.snapshot["dataset_id"] == validated.descriptor.id
    assert resolved.data_config.neighbor_function == "Nearest"
    assert resolved.data_config.K == 6
    assert resolved.snapshot["data"]["neighbor_function"] == "Nearest"
    assert resolved.snapshot["data"]["K"] == 6


def test_required_dataset_capabilities_require_a_sealed_dataset() -> None:
    with pytest.raises(
        ConfigResolutionError,
        match=r"required_capabilities.*sealed.*dataset",
    ):
        resolve_torch_configs(
            _ci_overrides(),
            required_capabilities=("has_object_truth",),
        )


def test_strict_explicit_resolution_may_defer_sealed_dataset_validation() -> None:
    resolved = resolve_torch_configs(
        _claim_grade_overrides("fno"),
        require_all_explicit=True,
    )

    assert resolved.dataset_id is None


def test_generic_resolution_may_omit_dataset_without_capabilities() -> None:
    resolved = resolve_torch_configs(_ci_overrides(), required_capabilities=())

    assert resolved.dataset_id is None


@pytest.mark.parametrize(
    "required_capabilities",
    [
        "has_object_truth",
        b"has_object_truth",
        7,
        ("has_object_truth", 7),
        ("",),
        ("   ",),
    ],
)
def test_required_capabilities_keyword_rejects_invalid_boundaries(
    required_capabilities: object,
) -> None:
    with pytest.raises(
        ConfigResolutionError,
        match=r"required_capabilities.*iterable.*nonempty strings",
    ):
        resolve_torch_configs(
            _ci_overrides(),
            required_capabilities=required_capabilities,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("dataset_id", ["", "   ", 7, b"dataset"])
def test_dataset_id_keyword_rejects_invalid_boundaries(dataset_id: object) -> None:
    with pytest.raises(
        ConfigResolutionError,
        match=r"dataset_id.*nonempty.*string",
    ):
        resolve_torch_configs(
            _ci_overrides(),
            dataset_id=dataset_id,  # type: ignore[arg-type]
        )


def test_dataset_id_keyword_must_match_selected_sealed_dataset(tmp_path: Path) -> None:
    validated = _validated_fixture(tmp_path)

    with pytest.raises(
        ConfigResolutionError,
        match=rf"dataset\.id='other'.*{validated.descriptor.id}",
    ):
        resolve_torch_configs(
            _ci_overrides(),
            dataset=validated,
            dataset_id="other",
        )


@pytest.mark.parametrize(
    ("kind", "truth", "use_bundle"),
    [
        ("synthetic", "object_truth", True),
        ("experimental", "reference_reconstruction", False),
    ],
)
@pytest.mark.parametrize("detector_size", [64, 128])
def test_selected_dataset_detector_geometry_must_match_data_n(
    tmp_path: Path,
    kind: str,
    truth: str,
    use_bundle: bool,
    detector_size: int,
) -> None:
    validated = _validated_fixture(
        tmp_path,
        kind=kind,
        truth=truth,
        detector_size=detector_size,
        standalone=kind == "experimental",
    )
    overrides = _ci_overrides()
    overrides["dataset.id"] = validated.descriptor.id
    selected = validated.bundle if use_bundle else validated

    if detector_size == 64:
        resolved = resolve_torch_configs(overrides, dataset=selected)
        assert resolved.dataset_id == validated.descriptor.id
    else:
        with pytest.raises(
            ConfigResolutionError,
            match=(
                rf"{validated.descriptor.id}.*detector_shape=\(128, 128\)"
                r".*data.N=64"
            ),
        ):
            resolve_torch_configs(overrides, dataset=selected)


def test_n_subsample_changes_canonical_group_count() -> None:
    coordinates = np.arange(4, dtype=np.float64)
    valid_indices = np.arange(4)
    group_counts = []
    for n_subsample in (1, 3):
        overrides = _ci_overrides(architecture="fno")
        overrides.update(
            {
                "data.C": 1,
                "data.grid_size": [1, 1],
                "data.n_subsample": n_subsample,
                "model.C_model": 1,
                "model.C_forward": 1,
            }
        )
        resolved = resolve_torch_configs(overrides)
        indices, _ = group_coords(
            coordinates,
            coordinates,
            coordinates,
            coordinates,
            None,
            valid_indices,
            resolved.data_config,
            C=1,
        )
        group_counts.append(len(indices))

    assert group_counts == [4, 12]


def test_resolver_uses_sealed_bundle_and_rejects_profile_mismatch(
    tmp_path: Path,
) -> None:
    validated = _validated_fixture(tmp_path)
    overrides = _legacy_overrides()
    overrides["dataset.id"] = validated.descriptor.id

    with pytest.raises(
        ConfigResolutionError, match="dataset compatibility.*scale_contract"
    ):
        resolve_torch_configs(overrides, dataset=validated.bundle)


@pytest.mark.parametrize(
    ("truth", "required", "passes"),
    [
        ("reference_reconstruction", "has_reference", True),
        ("reference_reconstruction", "has_object_truth", False),
        ("none", "has_reference", False),
    ],
)
def test_experimental_reference_capabilities_remain_distinct(
    tmp_path: Path,
    truth: str,
    required: str,
    passes: bool,
) -> None:
    validated = _validated_fixture(tmp_path, kind="experimental", truth=truth)
    overrides = _ci_overrides()
    overrides["dataset.id"] = validated.descriptor.id

    if passes:
        resolved = resolve_torch_configs(
            overrides, dataset=validated, required_capabilities=(required,)
        )
        assert resolved.dataset_id == validated.descriptor.id
    else:
        with pytest.raises(ConfigResolutionError, match=required):
            resolve_torch_configs(
                overrides, dataset=validated, required_capabilities=(required,)
            )


def test_legacy_dataset_rejects_required_count_capability(tmp_path: Path) -> None:
    validated = _validated_fixture(tmp_path, domain="normalized_amplitude")

    with pytest.raises(ConfigResolutionError, match="supports_count_metrics"):
        resolve_torch_configs(
            _legacy_overrides(),
            dataset=validated,
            required_capabilities=("supports_count_metrics",),
        )


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"data.grid_size": [-1, -4]}, "data.grid_size.*positive"),
        (
            {"execution.num_workers": 0, "execution.persistent_workers": True},
            "persistent_workers.*num_workers",
        ),
        (
            {"execution.num_workers": 0, "execution.prefetch_factor": 2},
            "prefetch_factor.*num_workers",
        ),
        ({"execution.strategy": "mystery"}, "execution.strategy"),
    ],
)
def test_runtime_boundary_values_cannot_be_normalized_or_deferred(
    updates: dict[str, object],
    message: str,
) -> None:
    overrides = _ci_overrides()
    overrides.update(updates)

    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


def test_constant_scheduler_allows_lr_below_unused_plateau_minimum() -> None:
    overrides = _ci_overrides()
    overrides.pop("training.lr_warmup_epochs")
    overrides.update(
        {
            "training.scheduler": "Default",
            "training.learning_rate": 1e-5,
        }
    )

    resolved = resolve_torch_configs(overrides)

    assert resolved.training_config.learning_rate == 1e-5


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"data.N": 32}, "data.N.*64, 128, or 256"),
        ({"model.probe_mask": [[1, 1], [1, 1]]}, "probe_mask.*shape.*64, 64"),
        (
            {"model.probe_mask_tensor": [[-1] * 64 for _ in range(64)]},
            "probe_mask_tensor.*nonnegative",
        ),
        ({"model.amp_activation": "relu"}, "model.amp_activation"),
    ],
)
def test_model_controls_fail_before_deferred_runtime_construction(
    updates: dict[str, object],
    message: str,
) -> None:
    overrides = _ci_overrides()
    overrides.update(updates)

    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        (
            {"training.epochs_fine_tune": 1, "training.fine_tune_gamma": 1.1},
            "training.fine_tune_gamma.*\(0, 1\]",
        ),
        (
            {"training.gradient_clip_algorithm": "mystery"},
            "training.gradient_clip_algorithm",
        ),
        ({"execution.checkpoint_monitor_metric": ""}, "checkpoint_monitor_metric"),
    ],
)
def test_broad_string_and_range_annotations_do_not_permit_inert_values(
    updates: dict[str, object],
    message: str,
) -> None:
    architecture = "cnn" if "training.epochs_fine_tune" in updates else "hybrid_resnet"
    overrides = _ci_overrides(architecture=architecture)
    overrides.update(updates)

    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize(
    ("channels", "grid_size"),
    [(2, [1, 2]), (4, [2, 2])],
)
def test_cnn_rejects_multi_channel_non_reassembled_object_before_construction(
    channels: int,
    grid_size: list[int],
) -> None:
    overrides = _ci_overrides(architecture="cnn")
    overrides.update(
        {
            "data.C": channels,
            "data.grid_size": grid_size,
            "model.C_model": channels,
            "model.C_forward": channels,
            "model.object_big": False,
        }
    )

    with pytest.raises(
        ConfigResolutionError,
        match=r"architecture='cnn'.*object_big=false.*C=1",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize(
    ("channels", "grid_size", "object_big"),
    [(1, [1, 1], False), (4, [2, 2], True)],
)
def test_cnn_valid_channel_object_boundaries_complete_full_forward(
    channels: int,
    grid_size: list[int],
    object_big: bool,
) -> None:
    overrides = _ci_overrides(architecture="cnn")
    overrides.update(
        {
            "data.C": channels,
            "data.grid_size": grid_size,
            "model.C_model": channels,
            "model.C_forward": channels,
            "model.object_big": object_big,
        }
    )
    resolved = resolve_torch_configs(overrides)

    expected = (1, channels, 64, 64)
    assert _canonical_forward_shapes(resolved) == (expected, expected, expected)


@pytest.mark.parametrize("channels", [0, 2, 3, 5])
def test_cnn_decoder_output_channels_must_be_one_or_c_model(channels: int) -> None:
    overrides = _ci_overrides(architecture="cnn")
    overrides["model.decoder_last_amp_channels"] = channels

    with pytest.raises(
        ConfigResolutionError,
        match=r"model.decoder_last_amp_channels.*1 or model.C_model",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("fraction", [0.0, 0.01, 0.500001, 1.0])
def test_cnn_decoder_outer_fraction_rejects_clamped_or_ineffective_values(
    fraction: float,
) -> None:
    overrides = _ci_overrides(architecture="cnn")
    overrides["model.n_filters_scale"] = 2
    overrides["model.decoder_last_c_outer_fraction"] = fraction

    with pytest.raises(
        ConfigResolutionError, match=r"decoder_last_c_outer_fraction.*0.015625.*0.5"
    ):
        resolve_torch_configs(overrides)


def test_cnn_decoder_outer_fraction_changes_constructed_channel_split() -> None:
    outer_channels = []
    for fraction in (0.125, 0.25):
        overrides = _ci_overrides(architecture="cnn")
        overrides["model.decoder_last_c_outer_fraction"] = fraction
        resolved = resolve_torch_configs(overrides)
        model = Autoencoder(resolved.model_config, resolved.data_config)
        outer_channels.append(model.decoder_amp.amp.c_outer)

    assert outer_channels == [8, 16]


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.K", 5),
        ("data.K_quadrant", 30),
        ("data.neighbor_function", "Min_dist"),
        ("data.neighbor_function", "4_quadrant"),
        ("data.min_neighbor_distance", 0.5),
        ("data.max_neighbor_distance", 8.0),
        ("data.scan_pattern", "Rectangular"),
    ],
)
def test_alternate_grouping_controls_are_unsupported(path: str, value: object) -> None:
    with pytest.raises(ConfigResolutionError, match=rf"{path}.*unsupported"):
        resolve_torch_configs({path: value})


@pytest.mark.parametrize("middle_trim", [2, 64])
def test_middle_trim_accepts_even_positive_boundaries(middle_trim: int) -> None:
    overrides = _ci_overrides()
    overrides["inference.middle_trim"] = middle_trim

    resolved = resolve_torch_configs(overrides)

    assert resolved.inference_config.middle_trim == middle_trim


@pytest.mark.parametrize(
    ("middle_trim", "message"),
    [(0, "positive"), (3, "even"), (66, r"must not exceed data.N")],
)
def test_middle_trim_rejects_invalid_geometry(
    middle_trim: int,
    message: str,
) -> None:
    overrides = _ci_overrides()
    overrides["inference.middle_trim"] = middle_trim

    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("architecture", ["cnn", "hybrid_resnet", "fno"])
def test_resolved_representative_configs_smoke_construct_canonical_model(
    architecture: str,
) -> None:
    resolved = resolve_torch_configs(_ci_overrides(architecture=architecture))

    assert _canonical_forward_shapes(resolved) == (
        (1, 4, 64, 64),
        (1, 4, 64, 64),
        (1, 4, 64, 64),
    )


def _canonical_forward_shapes(
    resolved: ResolvedTorchConfigs,
) -> tuple[tuple[int, ...], ...]:
    channels = resolved.data_config.C
    patch_size = resolved.data_config.N

    model = PtychoPINN(
        resolved.model_config,
        resolved.data_config,
        resolved.training_config,
    ).eval()
    diffraction = torch.rand(1, channels, patch_size, patch_size)
    positions = torch.zeros(1, channels, 1, 2)
    probe = torch.ones(
        1,
        channels,
        1,
        patch_size,
        patch_size,
        dtype=torch.complex64,
    )
    scale = torch.ones(1, 1, 1, 1)
    with torch.no_grad():
        output = model(diffraction, positions, probe, scale, scale)
    return tuple(tuple(item.shape) for item in output)


@pytest.mark.parametrize(
    "architecture",
    [
        "ffno",
        "hybrid",
        "stable_hybrid",
        "fno_vanilla",
        "neuralop_uno",
        "hybrid_resnet_ffno_ptychoblock_encoder",
        "hybrid_resnet_ptychoblock_ffno_encoder",
        "spectral_resnet_bottleneck_net",
        "spectral_resnet_bottleneck_linear_decoder",
        "hybrid_resnet_ffno_bottleneck",
        "hybrid_resnet_convnext_bottleneck",
    ],
)
def test_architectures_outside_closed_capability_table_are_rejected(
    architecture: str,
) -> None:
    with pytest.raises(ConfigResolutionError, match=r"model.architecture.*unsupported"):
        resolve_torch_configs(_ci_overrides(architecture=architecture))


@pytest.mark.parametrize("fno_blocks", [1, 2])
def test_hybrid_resnet_rejects_too_few_fno_blocks(fno_blocks: int) -> None:
    overrides = _ci_overrides(architecture="hybrid_resnet")
    overrides["model.fno_blocks"] = fno_blocks

    with pytest.raises(ConfigResolutionError, match=r"model.fno_blocks.*at least 3"):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("fno_blocks", [3, 5])
def test_hybrid_resnet_fno_block_depth_perturbs_constructed_forward(
    fno_blocks: int,
) -> None:
    overrides = _ci_overrides(architecture="hybrid_resnet")
    overrides["model.fno_blocks"] = fno_blocks
    resolved = resolve_torch_configs(overrides)
    model = PtychoPINN(
        resolved.model_config,
        resolved.data_config,
        resolved.training_config,
    ).eval()

    assert len(model.generator.encoder_blocks) == fno_blocks
    assert _canonical_forward_shapes(resolved) == (
        (1, 4, 64, 64),
        (1, 4, 64, 64),
        (1, 4, 64, 64),
    )


@pytest.mark.parametrize("fno_blocks", [1, 3])
def test_fno_block_depth_perturbs_constructed_forward(fno_blocks: int) -> None:
    overrides = _ci_overrides(architecture="fno")
    overrides["model.fno_blocks"] = fno_blocks
    resolved = resolve_torch_configs(overrides)
    model = PtychoPINN(
        resolved.model_config,
        resolved.data_config,
        resolved.training_config,
    ).eval()

    assert len(model.generator.fno_blocks) == fno_blocks
    assert _canonical_forward_shapes(resolved) == (
        (1, 4, 64, 64),
        (1, 4, 64, 64),
        (1, 4, 64, 64),
    )


@pytest.mark.parametrize("architecture", ["fno", "hybrid_resnet"])
@pytest.mark.parametrize("value", [True, 3.0])
def test_fno_blocks_requires_an_exact_integer(
    architecture: str,
    value: object,
) -> None:
    overrides = _ci_overrides(architecture=architecture)
    overrides["model.fno_blocks"] = value

    with pytest.raises(ConfigResolutionError, match=r"model.fno_blocks.*exact integer"):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("architecture", ["fno", "hybrid_resnet"])
@pytest.mark.parametrize("epochs_fine_tune", [0, 1])
def test_non_cnn_fine_tuning_is_resolver_owned_and_not_sweepable(
    architecture: str,
    epochs_fine_tune: int,
) -> None:
    overrides = _ci_overrides(architecture=architecture)
    overrides["training.epochs_fine_tune"] = epochs_fine_tune
    if epochs_fine_tune:
        overrides["training.fine_tune_gamma"] = 0.5

    with pytest.raises(
        ConfigResolutionError,
        match=r"training.epochs_fine_tune.*non-applicable",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize("architecture", ["fno", "hybrid_resnet"])
def test_non_cnn_fine_tune_gamma_is_non_applicable(architecture: str) -> None:
    overrides = _ci_overrides(architecture=architecture)
    overrides["training.fine_tune_gamma"] = 0.5

    with pytest.raises(
        ConfigResolutionError,
        match=r"training.fine_tune_gamma.*non-applicable",
    ):
        resolve_torch_configs(overrides)


def test_cnn_fine_tune_gamma_is_non_applicable_when_transition_is_disabled() -> None:
    overrides = _ci_overrides(architecture="cnn")
    overrides.update(
        {
            "training.epochs_fine_tune": 0,
            "training.fine_tune_gamma": 0.5,
        }
    )

    with pytest.raises(
        ConfigResolutionError,
        match=r"training.fine_tune_gamma.*non-applicable",
    ):
        resolve_torch_configs(overrides)


def test_cnn_claim_fine_tune_transition_requires_explicit_gamma(
    tmp_path: Path,
) -> None:
    overrides = _claim_grade_overrides("cnn")
    overrides["training.epochs_fine_tune"] = 2

    with pytest.raises(ConfigResolutionError, match=r"missing:.*fine_tune_gamma"):
        resolve_torch_configs(
            overrides,
            dataset=_validated_fixture(tmp_path),
            require_all_explicit=True,
        )


def test_cnn_fine_tune_transition_freezes_encoder_and_scales_learning_rate() -> None:
    overrides = _ci_overrides(architecture="cnn")
    overrides.update(
        {
            "training.epochs_fine_tune": 2,
            "training.fine_tune_gamma": 0.5,
        }
    )
    resolved = resolve_torch_configs(overrides)
    model = PtychoPINN_Lightning(
        resolved.model_config,
        resolved.data_config,
        resolved.training_config,
        resolved.inference_config,
    )
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = type(
        "TrainerBoundary",
        (),
        {
            "current_epoch": resolved.training_config.epochs - 1,
            "optimizers": [optimizer],
        },
    )()
    callback = EncoderFreezeCallback(
        freeze_at_epoch=resolved.training_config.epochs,
        lr_gamma=resolved.training_config.fine_tune_gamma,
    )

    assert all(
        parameter.requires_grad
        for parameter in model.model.autoencoder.encoder.parameters()
    )
    callback.on_train_epoch_start(trainer, model)
    assert all(
        parameter.requires_grad
        for parameter in model.model.autoencoder.encoder.parameters()
    )
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)

    trainer.current_epoch = resolved.training_config.epochs
    callback.on_train_epoch_start(trainer, model)

    assert not any(
        parameter.requires_grad
        for parameter in model.model.autoencoder.encoder.parameters()
    )
    assert model._fine_tuning_mode is True
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.05)


@pytest.mark.parametrize("architecture", ["fno", "hybrid_resnet"])
def test_learned_input_channels_is_locked_to_proven_single_channel(
    architecture: str,
) -> None:
    overrides = _ci_overrides(architecture=architecture)
    overrides["model.learned_input_channels"] = 2

    with pytest.raises(
        ConfigResolutionError,
        match=r"model.learned_input_channels.*1",
    ):
        resolve_torch_configs(overrides)


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        ("model.resnet_width", 2, r"resnet_width.*divisible by 4"),
        ("model.resnet_width", 6, r"resnet_width.*divisible by 4"),
        ("model.max_hidden_channels", 1, r"max_hidden_channels.*at least 4"),
        ("model.max_hidden_channels", 3, r"max_hidden_channels.*at least 4"),
    ],
)
def test_hybrid_widths_fail_before_builder_runtime(
    path: str,
    value: int,
    message: str,
) -> None:
    overrides = _ci_overrides(architecture="hybrid_resnet")
    overrides[path] = value

    with pytest.raises(ConfigResolutionError, match=message):
        resolve_torch_configs(overrides)


def test_disabled_hybrid_skips_reject_inert_style() -> None:
    overrides = _ci_overrides(architecture="hybrid_resnet")
    overrides.update(
        {
            "model.hybrid_skip_connections": False,
            "model.hybrid_skip_style": "add",
        }
    )

    with pytest.raises(
        ConfigResolutionError, match=r"hybrid_skip_style.*non-applicable"
    ):
        resolve_torch_configs(overrides)


def test_hybrid_structural_fields_have_model_owner_not_execution_owner() -> None:
    overrides = _ci_overrides(architecture="hybrid_resnet")
    overrides.update(
        {
            "model.hybrid_encoder_fusion_mode": "branch_gated_layerscale",
            "model.hybrid_encoder_layerscale_init": 0.05,
            "model.hybrid_encoder_branch_gate_init": 0.2,
            "model.hybrid_encoder_branch_select": "conv_only",
        }
    )

    resolved = resolve_torch_configs(overrides)

    assert resolved.model_config.hybrid_encoder_fusion_mode == "branch_gated_layerscale"
    assert resolved.model_config.hybrid_encoder_layerscale_init == 0.05
    assert resolved.model_config.hybrid_encoder_branch_gate_init == 0.2
    assert resolved.model_config.hybrid_encoder_branch_select == "conv_only"
    assert resolved.execution_config.hybrid_encoder_fusion_mode == "baseline"
    assert resolved.execution_config.hybrid_encoder_branch_select == "both"


@pytest.mark.parametrize("style", ["add", "concat", "gated_add"])
def test_enabled_hybrid_skip_styles_construct_and_forward(
    tmp_path: Path,
    style: str,
) -> None:
    overrides = _claim_grade_overrides("hybrid_resnet")
    overrides.update(
        {
            "model.hybrid_skip_connections": True,
            "model.hybrid_skip_style": style,
        }
    )

    resolved = resolve_torch_configs(
        overrides,
        dataset=_validated_fixture(tmp_path),
        require_all_explicit=True,
    )

    assert resolved.model_config.hybrid_skip_style == style
    assert _canonical_forward_shapes(resolved) == (
        (1, 4, 64, 64),
        (1, 4, 64, 64),
        (1, 4, 64, 64),
    )


def test_enabled_hybrid_skips_require_explicit_style_in_claim_grade(
    tmp_path: Path,
) -> None:
    overrides = _claim_grade_overrides("hybrid_resnet")
    overrides["model.hybrid_skip_connections"] = True

    with pytest.raises(ConfigResolutionError, match=r"missing:.*hybrid_skip_style"):
        resolve_torch_configs(
            overrides,
            dataset=_validated_fixture(tmp_path),
            require_all_explicit=True,
        )


def _claim_grade_overrides(architecture: str) -> dict[str, Any]:
    overrides: dict[str, Any] = {
        "data.scale_contract_version": "ci_intensity_v2",
        "data.measurement_domain": "count_intensity",
        "data.N": 64,
        "data.C": 4,
        "data.n_subsample": 1,
        "data.grid_size": [2, 2],
        "data.probe_scale": 4.0,
        "data.probe_normalize": True,
        "data.x_bounds": [0.1, 0.9],
        "data.y_bounds": [0.1, 0.9],
        "model.mode": "Unsupervised",
        "model.architecture": architecture,
        "model.C_model": 4,
        "model.object_big": True,
        "model.probe_big": False,
        "model.offset": 6,
        "model.C_forward": 4,
        "model.training_patch_weighting": "probe",
        "model.physics_forward_mode": "rectangular_scaled",
        "model.rect_s1s2_trainable": True,
        "model.loss_function": "Poisson",
        "model.probe_mask": False,
        "model.amp_loss": "disabled",
        "model.phase_loss": "disabled",
        "training.learning_rate": 2e-4,
        "training.epochs": 10,
        "training.batch_size": 16,
        "training.scheduler": "Default",
        "training.accum_steps": 1,
        "training.gradient_clip_val": "disabled",
        "training.optimizer": "adam",
        "training.weight_decay": 0.0,
        "training.adam_beta1": 0.9,
        "training.adam_beta2": 0.999,
        "training.log_grad_norm": False,
        "training.torch_loss_mode": "poisson",
        "training.experiment_name": "claim-grade",
        "training.notes": "",
        "training.model_name": "PtychoPINNv2",
        "inference.middle_trim": 32,
        "inference.batch_size": 32,
        "inference.patch_weighting": "probe",
        "inference.varpro_scaling": True,
        "execution.accelerator": "cpu",
        "execution.devices": 1,
        "execution.strategy": "auto",
        "execution.deterministic": True,
        "execution.precision": "32-true",
        "execution.num_workers": 0,
        "execution.pin_memory": False,
        "execution.enable_progress_bar": False,
        "execution.enable_checkpointing": True,
        "execution.checkpoint_save_top_k": 1,
        "execution.checkpoint_monitor_metric": "val_loss",
        "execution.checkpoint_mode": "min",
        "execution.early_stop_patience": 100,
    }
    if architecture == "cnn":
        overrides.update(
            {
                "model.cnn_output_mode": "real_imag",
                "model.use_shared_decoder": False,
                "model.n_filters_scale": 2,
                "model.batch_norm": False,
                "model.cbam_encoder": True,
                "model.cbam_bottleneck": False,
                "model.decoder_last_c_outer_fraction": 0.125,
                "model.decoder_last_amp_channels": 1,
                "training.epochs_fine_tune": 0,
            }
        )
    else:
        overrides.update(
            {
                "model.generator_output_mode": "real_imag",
                "model.fno_modes": 12,
                "model.fno_width": 32,
                "model.fno_input_transform": "none",
                "model.learned_input_channels": 1,
            }
        )
        if architecture == "fno":
            overrides.update(
                {
                    "model.fno_blocks": 4,
                    "model.fno_cnn_blocks": 2,
                }
            )
        else:
            overrides.update(
                {
                    "model.fno_blocks": 4,
                    "model.max_hidden_channels": "auto",
                    "model.resnet_width": "auto",
                    "model.hybrid_skip_connections": False,
                    "model.hybrid_downsample_steps": 2,
                    "model.hybrid_downsample_op": "stride_conv",
                    "model.hybrid_encoder_conv_hidden_scale": 1.0,
                    "model.hybrid_encoder_spectral_hidden_scale": 1.0,
                    "model.hybrid_resnet_blocks": 6,
                }
            )
    return overrides


@pytest.mark.parametrize("architecture", ["cnn", "hybrid_resnet", "fno"])
def test_complete_claim_grade_architecture_fixtures_resolve(
    tmp_path: Path,
    architecture: str,
) -> None:
    resolved = resolve_torch_configs(
        _claim_grade_overrides(architecture),
        dataset=_validated_fixture(tmp_path),
        require_all_explicit=True,
    )

    assert resolved.model_config.architecture == architecture
    assert resolved.training_config.nll is True
    if architecture != "cnn":
        assert resolved.training_config.epochs_fine_tune == 0
        assert resolved.training_config.fine_tune_gamma == 0.1
    assert (
        resolved.execution_config.learning_rate
        == resolved.training_config.learning_rate
    )
    assert _canonical_forward_shapes(resolved) == (
        (1, 4, 64, 64),
        (1, 4, 64, 64),
        (1, 4, 64, 64),
    )
    lightning_model = PtychoPINN_Lightning(
        resolved.model_config,
        resolved.data_config,
        resolved.training_config,
        resolved.inference_config,
    )
    assert resolved.execution_config.checkpoint_monitor_metric == "val_loss"
    assert (
        _resolve_checkpoint_monitor(resolved.execution_config, lightning_model)
        == lightning_model.val_loss_name
    )
    checkpoint = ModelCheckpoint(
        dirpath=str(tmp_path),
        monitor=lightning_model.val_loss_name,
        mode=resolved.execution_config.checkpoint_mode,
        save_top_k=resolved.execution_config.checkpoint_save_top_k,
    )
    early_stopping = EarlyStopping(
        monitor=lightning_model.val_loss_name,
        mode=resolved.execution_config.checkpoint_mode,
        patience=resolved.execution_config.early_stop_patience,
        strict=True,
    )
    assert checkpoint.monitor == lightning_model.val_loss_name
    assert early_stopping.monitor == lightning_model.val_loss_name


@pytest.mark.parametrize("architecture", ["cnn", "hybrid_resnet", "fno"])
def test_claim_grade_rejects_each_missing_applicable_field(
    tmp_path: Path,
    architecture: str,
) -> None:
    complete = _claim_grade_overrides(architecture)
    dataset = _validated_fixture(tmp_path)
    for missing_path in sorted(complete):
        incomplete = dict(complete)
        incomplete.pop(missing_path)
        with pytest.raises(ConfigResolutionError, match=rf"missing:.*{missing_path}"):
            resolve_torch_configs(
                incomplete,
                dataset=dataset,
                require_all_explicit=True,
            )


@pytest.mark.parametrize(
    ("architecture", "path", "value"),
    [
        ("hybrid_resnet", "model.cnn_output_mode", "real_imag"),
        ("cnn", "model.generator_output_mode", "real_imag"),
        ("cnn", "model.fno_modes", 12),
        ("fno", "model.decoder_last_c_outer_fraction", 0.125),
        ("fno", "model.hybrid_resnet_blocks", 6),
        ("fno", "model.probe_mask_sigma", 1.0),
        ("fno", "training.plateau_factor", 0.5),
        ("fno", "execution.prefetch_factor", "auto"),
    ],
)
def test_non_applicable_controls_are_rejected(
    architecture: str,
    path: str,
    value: object,
) -> None:
    overrides = _claim_grade_overrides(architecture)
    overrides[path] = value

    with pytest.raises(ConfigResolutionError, match=rf"{path}.*non-applicable"):
        resolve_torch_configs(overrides)


def test_named_toml_sentinels_resolve_only_for_applicable_optionals(
    tmp_path: Path,
) -> None:
    hybrid = resolve_torch_configs(_claim_grade_overrides("hybrid_resnet"))
    assert hybrid.model_config.max_hidden_channels is None
    assert hybrid.model_config.resnet_width is None
    assert hybrid.training_config.gradient_clip_val is None
    assert hybrid.model_config.amp_loss is None
    assert hybrid.model_config.phase_loss is None

    workers = _claim_grade_overrides("fno")
    workers.update(
        {
            "execution.num_workers": 2,
            "execution.persistent_workers": True,
            "execution.prefetch_factor": "auto",
        }
    )
    resolved_workers = resolve_torch_configs(
        workers,
        dataset=_validated_fixture(tmp_path),
        require_all_explicit=True,
    )
    assert resolved_workers.execution_config.prefetch_factor is None

    probe = _claim_grade_overrides("fno")
    probe.update(
        {
            "model.probe_mask": True,
            "model.probe_mask_sigma": 1.0,
            "model.probe_mask_diameter": "auto",
            "model.probe_mask_tensor": "auto",
        }
    )
    resolved_probe = resolve_torch_configs(
        probe,
        dataset=_validated_fixture(tmp_path),
        require_all_explicit=True,
    )
    assert resolved_probe.model_config.probe_mask_diameter is None
    assert resolved_probe.model_config.probe_mask_tensor is None

    with pytest.raises(ConfigResolutionError, match="data.probe_scale.*number"):
        resolve_torch_configs({"data.probe_scale": "auto"})
