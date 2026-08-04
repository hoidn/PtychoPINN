"""Public CLI contracts for the generic synthetic pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import ast

import pytest


def _plain(value):
    if isinstance(value, dict) or hasattr(value, "items"):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def test_pyproject_installs_the_canonical_synthetic_command():
    root = Path(__file__).resolve().parents[2]
    payload = (root / "pyproject.toml").read_text(encoding="utf-8")
    assert 'ptycho_synthetic = "scripts.simulation.synthetic_pipeline:main"' in payload
    assert '"PyYAML"' in payload
    assert "\"tomli; python_version < '3.11'\"" in payload


def test_public_cli_preflight_imports_have_a_python310_tomllib_fallback():
    root = Path(__file__).resolve().parents[2]
    for relative in (
        "scripts/simulation/synthetic_pipeline.py",
        "ptycho/config/config.py",
    ):
        tree = ast.parse((root / relative).read_text(encoding="utf-8"))
        imports = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        assert "tomllib" in imports, relative
        assert "tomli" in imports, relative


def test_no_stage_selection_resolves_the_complete_default_workflow(monkeypatch):
    from scripts.simulation import synthetic_pipeline as cli
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    captured = {}
    monkeypatch.setattr(
        cli,
        "_run_pipeline",
        lambda request: captured.setdefault("request", request),
    )

    assert cli.main([]) == 0

    request = captured["request"]
    resolved = resolve_synthetic_workflow(
        profile=request.profile,
        file_values=request.file_values,
        cli_values=request.cli_values,
    )
    assert resolved.workflow.stages == (
        "simulate",
        "train",
        "reconstruct",
        "evaluate",
    )
    assert request.raw_argv == ()


def test_config_file_values_survive_when_flags_are_omitted_and_cli_wins(tmp_path):
    from scripts.simulation import synthetic_pipeline as cli
    from ptycho.workflows.synthetic_config import resolve_synthetic_workflow

    config_path = tmp_path / "workflow.json"
    config_path.write_text(
        json.dumps(
            {
                "profile": "synthetic-lines",
                "simulation": {"N": 64, "seed": 19},
                "training": {"epochs": 7, "batch_size": 5},
                "inference": {"groups_per_center": 3},
                "workflow": {"output_root": str(tmp_path / "from-file")},
            }
        ),
        encoding="utf-8",
    )
    argv = ["--config", str(config_path), "--epochs", "9"]
    args = cli.parse_arguments(argv)

    request = cli.build_pipeline_request(args, raw_argv=tuple(argv))

    file_values = _plain(request.file_values)
    cli_values = _plain(request.cli_values)
    assert file_values["simulation"] == {"N": 64, "seed": 19}
    assert file_values["training"] == {"epochs": 7, "batch_size": 5}
    assert file_values["inference"] == {"groups_per_center": 3}
    assert cli_values == {"training": {"epochs": 9}}
    assert request.profile == "synthetic-lines"
    resolved = resolve_synthetic_workflow(
        profile=request.profile,
        file_values=request.file_values,
        cli_values=request.cli_values,
    )
    assert resolved.simulation.train.N == 64
    assert resolved.training.epochs == 9
    assert resolved.training.batch_size == 5
    assert resolved.inference.groups_per_center == 3


def test_explicit_cli_surface_maps_to_owned_namespaces(tmp_path):
    from scripts.simulation import synthetic_pipeline as cli

    argv = [
        "--profile",
        "synthetic-lines",
        "--stages",
        "simulate,train",
        "--output-root",
        str(tmp_path / "run"),
        "--architecture",
        "cnn",
        "--N",
        "128",
        "--gridsize",
        "2",
        "--object-kind",
        "lines",
        "--object-size",
        "392",
        "--seed",
        "3",
        "--scan-buffer",
        "64",
        "--scan-offset",
        "4",
        "--photons-per-pattern",
        "1e9",
        "--beamstop-diameter",
        "11",
        "--probe-source",
        "custom",
        "--probe-path",
        str(tmp_path / "probe.npz"),
        "--probe-transform",
        "smooth:0.5|pad_extrapolate_boundary_matched:128",
        "--simulation-probe-mask-diameter",
        "72",
        "--model-probe-mask",
        "--model-probe-mask-diameter",
        "68",
        "--model-probe-mask-sigma",
        "1.5",
        "--train-patterns",
        "4096",
        "--test-patterns",
        "1024",
        "--train-raw-selection",
        "4096",
        "--training-groups",
        "4096",
        "--validation-groups",
        "1024",
        "--neighbor-count",
        "4",
        "--neighbor-pool-size",
        "4",
        "--epochs",
        "5",
        "--batch-size",
        "16",
        "--optimizer",
        "adamw",
        "--learning-rate",
        "0.0002",
        "--scheduler",
        "ReduceLROnPlateau",
        "--gradient-clip-val",
        "0.25",
        "--gradient-clip-algorithm",
        "norm",
        "--groups-per-center",
        "2",
        "--no-varpro",
        "--accelerator",
        "cuda",
        "--devices",
        "1",
        "--strategy",
        "auto",
        "--precision",
        "32-true",
        "--workers",
        "0",
        "--logger",
        "csv",
        "--deterministic",
        "--checkpoint-save-top-k",
        "1",
    ]
    request = cli.build_pipeline_request(
        cli.parse_arguments(argv),
        raw_argv=tuple(argv),
    )
    patch = _plain(request.cli_values)

    assert patch["simulation"] == {
        "N": 128,
        "gridsize": 2,
        "seed": 3,
        "train_patterns": 4096,
        "test_patterns": 1024,
        "object": {"kind": "lines", "image_size": [392, 392]},
        "probe": {
            "source": "custom",
            "source_path": str(tmp_path / "probe.npz"),
            "transform_pipeline": "smooth:0.5|pad_extrapolate_boundary_matched:128",
            "mask_diameter": 72.0,
        },
        "scan": {"buffer": 64, "offset": 4},
        "detector": {
            "photons_per_pattern": 1e9,
            "beamstop_diameter": 11.0,
        },
    }
    assert patch["model"]["architecture"] == "cnn"
    assert patch["model"]["probe_mask"] is True
    assert patch["model"]["probe_mask_diameter"] == 68.0
    assert patch["model"]["probe_mask_sigma"] == 1.5
    assert patch["training"] == {
        "train_raw_selection": 4096,
        "training_groups": 4096,
        "validation_groups": 1024,
        "neighbor_count": 4,
        "neighbor_pool_size": 4,
        "epochs": 5,
        "batch_size": 16,
        "optimizer": "adamw",
        "learning_rate": 0.0002,
        "scheduler": "ReduceLROnPlateau",
        "gradient_clip_val": 0.25,
        "gradient_clip_algorithm": "norm",
    }
    assert patch["inference"] == {
        "groups_per_center": 2,
        "varpro_scaling": False,
    }
    assert patch["workflow"] == {
        "stages": ["simulate", "train"],
        "output_root": str(tmp_path / "run"),
        "accelerator": "cuda",
        "devices": 1,
        "strategy": "auto",
        "precision": "32-true",
        "num_workers": 0,
        "logger_backend": "csv",
        "deterministic": True,
        "checkpoint_save_top_k": 1,
    }


@pytest.mark.parametrize(
    "stages",
    [
        "",
        "train,simulate",
        "simulate,simulate",
        "simulate,unknown",
        "simulate,",
        ",simulate",
        "simulate,,train",
    ],
)
def test_stage_selection_is_validated_by_argparse(stages):
    from scripts.simulation import synthetic_pipeline as cli

    with pytest.raises(SystemExit) as error:
        cli.parse_arguments(["--stages", stages])
    assert error.value.code == 2


def test_partial_count_intensity_switches_are_not_public_flags():
    from scripts.simulation import synthetic_pipeline as cli

    with pytest.raises(SystemExit) as error:
        cli.parse_arguments(["--measurement-domain", "count_intensity"])
    assert error.value.code == 2


def test_structured_config_rejects_unknown_suffix_and_nonobject(tmp_path):
    from scripts.simulation import synthetic_pipeline as cli

    text_path = tmp_path / "workflow.txt"
    text_path.write_text("{}", encoding="utf-8")
    args = cli.parse_arguments(["--config", str(text_path)])
    with pytest.raises(ValueError, match="JSON.*TOML.*YAML"):
        cli.build_pipeline_request(args)

    json_path = tmp_path / "workflow.json"
    json_path.write_text("[]", encoding="utf-8")
    args = cli.parse_arguments(["--config", str(json_path)])
    with pytest.raises(ValueError, match="object/mapping"):
        cli.build_pipeline_request(args)


def test_yaml_config_is_supported_and_explicit_profile_wins(tmp_path):
    from scripts.simulation import synthetic_pipeline as cli

    config_path = tmp_path / "workflow.yaml"
    config_path.write_text(
        "profile: stale-profile\ntraining:\n  epochs: 6\n",
        encoding="utf-8",
    )
    args = cli.parse_arguments(
        [
            "--config",
            str(config_path),
            "--profile",
            "synthetic-lines",
        ]
    )

    request = cli.build_pipeline_request(args)

    assert request.profile == "synthetic-lines"
    assert _plain(request.file_values) == {"training": {"epochs": 6}}


def test_toml_config_is_supported(tmp_path):
    from scripts.simulation import synthetic_pipeline as cli

    config_path = tmp_path / "workflow.toml"
    config_path.write_text(
        'profile = "synthetic-lines"\n[training]\nepochs = 8\n',
        encoding="utf-8",
    )

    request = cli.build_pipeline_request(
        cli.parse_arguments(["--config", str(config_path)])
    )

    assert request.profile == "synthetic-lines"
    assert _plain(request.file_values) == {"training": {"epochs": 8}}


def test_config_profile_must_be_a_nonempty_string_when_present(tmp_path):
    from scripts.simulation import synthetic_pipeline as cli

    config_path = tmp_path / "workflow.json"
    config_path.write_text('{"profile": null}', encoding="utf-8")

    with pytest.raises(ValueError, match="profile must be a nonempty string"):
        cli.build_pipeline_request(cli.parse_arguments(["--config", str(config_path)]))


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--object-kind", "dead_leaves"], r"object\.kind.*lines"),
        (["--object-size", "256"], r"object\.image_size.*392"),
        (["--scan-offset", "9"], r"scan\.offset.*exactly 4"),
        (["--beamstop-diameter", "8"], r"beamstop_diameter.*unsupported"),
    ],
)
def test_flat_acquisition_v1_restrictions_fail_before_executor(
    tmp_path,
    extra,
    message,
):
    from scripts.simulation import synthetic_pipeline as cli
    from ptycho.workflows.synthetic_pipeline import run_synthetic_pipeline

    argv = [
        "--stages",
        "simulate",
        "--output-root",
        str(tmp_path),
        *extra,
    ]
    request = cli.build_pipeline_request(
        cli.parse_arguments(argv),
        raw_argv=tuple(argv),
    )
    calls = []

    with pytest.raises(ValueError, match=message):
        run_synthetic_pipeline(
            request,
            simulation_executor=lambda stage_request: calls.append(stage_request),
        )

    assert calls == []


def test_execution_cross_field_restrictions_fail_before_executor(tmp_path):
    from scripts.simulation import synthetic_pipeline as cli
    from ptycho.workflows.synthetic_pipeline import run_synthetic_pipeline

    argv = [
        "--stages",
        "simulate",
        "--output-root",
        str(tmp_path),
        "--workers",
        "0",
        "--persistent-workers",
    ]
    request = cli.build_pipeline_request(
        cli.parse_arguments(argv),
        raw_argv=tuple(argv),
    )
    calls = []

    with pytest.raises(
        ValueError,
        match="persistent_workers=True requires num_workers > 0",
    ):
        run_synthetic_pipeline(
            request,
            simulation_executor=lambda stage_request: calls.append(stage_request),
        )

    assert calls == []
