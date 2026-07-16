"""Focused CLI contracts for canonical simulation configuration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from ptycho.config import load_simulation_config
from ptycho.config import (
    DetectorSimulationConfig,
    ProbeSimulationConfig,
    ScanSimulationConfig,
    SimulationConfig,
    SyntheticObjectConfig,
)
from ptycho.config.config import ModelConfig, TrainingConfig


def _write_simulation_toml(
    path: Path,
    *,
    N: int = 128,
    kind: str = "lines",
    source_path: str = "probe.npz",
    scan_kind: str = "grid",
) -> Path:
    path.write_text(
        f"""
[simulation]
N = {N}
seed = 3

[simulation.probe]
source = "custom"
source_path = "{source_path}"
transform_pipeline = "smooth:0.5|pad_extrapolate_boundary_matched:{N}"

[simulation.object]
kind = "{kind}"
image_size = [256, 256]
objects_per_probe = 4
diffractions_per_object = 700
set_phi = true

[simulation.scan]
kind = "{scan_kind}"
grid_size = [2, 2]
offset = 4
outer_offset_train = 8
outer_offset_test = 20
train_groups = 3
test_groups = 2
buffer = 12

[simulation.detector]
photons_per_pattern = 1e8
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return path


def test_load_simulation_config_accepts_wrapped_toml_and_rejects_unknown_keys(
    tmp_path: Path,
):
    config = load_simulation_config(_write_simulation_toml(tmp_path / "sim.toml"))
    assert config.N == 128
    assert config.scan.grid_size == (2, 2)

    invalid = tmp_path / "invalid.toml"
    invalid.write_text("[simulation]\nN = 64\nepochs = 10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="simulation.epochs"):
        load_simulation_config(invalid)


def test_grid_lines_cli_file_values_and_explicit_cli_precedence(tmp_path: Path):
    from scripts.studies import grid_lines_workflow as cli

    config_path = _write_simulation_toml(tmp_path / "sim.toml")
    from_file = cli.build_config(
        cli.parse_args(
            [
                "--simulation-config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "file"),
            ]
        )
    )
    assert from_file.N == 128
    assert from_file.gridsize == 2
    assert from_file.nphotons == 1e8
    assert from_file.probe_transform_pipeline.endswith(":128")

    overridden = cli.build_config(
        cli.parse_args(
            [
                "--simulation-config",
                str(config_path),
                "--output-dir",
                str(tmp_path / "override"),
                "--N",
                "64",
                "--gridsize",
                "1",
                "--nphotons",
                "200000000",
            ]
        )
    )
    assert overridden.N == 64
    assert overridden.gridsize == 1
    assert overridden.nphotons == 2e8
    assert overridden.probe_transform_pipeline.endswith(":64")


def test_grid_lines_cli_without_file_retains_historical_defaults(tmp_path: Path):
    from scripts.studies import grid_lines_workflow as cli

    cfg = cli.build_config(
        cli.parse_args(
            [
                "--N",
                "64",
                "--gridsize",
                "1",
                "--output-dir",
                str(tmp_path),
            ]
        )
    )
    assert cfg.nimgs_train == 2
    assert cfg.nimgs_test == 2
    assert cfg.nphotons == 1e9
    assert cfg.probe_scale_mode == "pad_preserve"
    assert cfg.probe_smoothing_sigma == 0.5


def test_generic_simulation_cli_resolves_file_then_explicit_legacy_overrides(
    tmp_path: Path,
):
    from scripts.simulation import simulate_and_save as cli

    input_path = tmp_path / "input.npz"
    config_path = _write_simulation_toml(
        tmp_path / "sim.toml",
        N=64,
        source_path=str(input_path),
        scan_kind="nongrid",
    )
    args = cli.parse_arguments(
        [
            "--simulation-config",
            str(config_path),
            "--input-file",
            str(input_path),
            "--output-file",
            str(tmp_path / "out.npz"),
            "--n-images",
            "900",
            "--n-photons",
            "200000000",
            "--seed",
            "9",
        ]
    )
    simulation = cli.resolve_simulation_config(
        args,
        object_shape=(256, 256),
        probe_shape=(32, 32),
    )
    assert simulation.object.diffractions_per_object == 900
    assert simulation.detector.photons_per_pattern == 2e8
    assert simulation.seed == 9


def test_generic_simulation_cli_prepares_a_smaller_input_probe_from_the_recipe(
    tmp_path: Path,
):
    from scripts.simulation import simulate_and_save as cli

    input_path = tmp_path / "input.npz"
    config_path = _write_simulation_toml(
        tmp_path / "sim.toml",
        N=128,
        source_path=str(input_path),
        scan_kind="nongrid",
    )
    args = cli.parse_arguments(
        [
            "--simulation-config",
            str(config_path),
            "--input-file",
            str(input_path),
            "--output-file",
            str(tmp_path / "out.npz"),
        ]
    )
    simulation = cli.resolve_simulation_config(
        args,
        object_shape=(256, 256),
        probe_shape=(64, 64),
    )
    yy, xx = np.indices((64, 64))
    source = (
        (1.0 + 0.01 * yy)
        * np.exp(1j * (0.002 * (xx**2 + yy**2) + 0.05 * np.sin(xx)))
    ).astype(np.complex64)
    np.savez(input_path, probeGuess=source)

    prepared = cli.prepare_probe_for_simulation(source, simulation)

    assert prepared.shape == (128, 128)
    assert np.array_equal(
        prepared[32:96, 32:96],
        cli.apply_probe_transform_pipeline(
            source,
            cli.parse_probe_transform_pipeline("smooth:0.5"),
        ),
    )


def test_generic_simulation_uses_the_configured_probe_source_not_input_payload(
    tmp_path: Path,
):
    from scripts.simulation import simulate_and_save as cli

    configured_source = np.ones((4, 4), dtype=np.complex64)
    configured_path = tmp_path / "configured.npz"
    np.savez(configured_path, probeGuess=configured_source)
    unrelated_input_probe = np.full((4, 4), 7 + 2j, dtype=np.complex64)
    simulation = SimulationConfig(
        N=8,
        probe=ProbeSimulationConfig(
            source="custom",
            source_path=configured_path,
            transform_pipeline="pad_preserve:8",
        ),
        object=SyntheticObjectConfig(kind="lines", image_size=(32, 32)),
        scan=ScanSimulationConfig(kind="nongrid", buffer=4),
    )

    prepared = cli.prepare_probe_for_simulation(unrelated_input_probe, simulation)

    assert np.array_equal(prepared[2:6, 2:6], configured_source)
    assert not np.any(prepared == unrelated_input_probe[0, 0])


def test_generic_simulation_rejects_grid_recipe_instead_of_ignoring_geometry(
    tmp_path: Path,
):
    from scripts.simulation import simulate_and_save as cli

    input_path = tmp_path / "input.npz"
    config_path = _write_simulation_toml(
        tmp_path / "grid.toml", N=64, source_path=str(input_path)
    )
    args = cli.parse_arguments(
        [
            "--simulation-config",
            str(config_path),
            "--input-file",
            str(input_path),
            "--output-file",
            str(tmp_path / "out.npz"),
        ]
    )

    with pytest.raises(ValueError, match=r"simulation\.scan\.kind.*nongrid"):
        cli.resolve_simulation_config(
            args,
            object_shape=(256, 256),
            probe_shape=(32, 32),
        )


def test_generic_simulation_persists_complete_probe_identity(
    monkeypatch, tmp_path: Path
):
    from ptycho import nongrid_simulation
    from scripts.simulation import simulate_and_save as cli

    yy, xx = np.indices((4, 4))
    raw_probe = np.exp(1j * (0.02 * (xx**2 + yy**2) + 0.1 * xx)).astype(
        np.complex64
    )
    input_path = tmp_path / "input.npz"
    np.savez(
        input_path,
        objectGuess=np.ones((32, 32), dtype=np.complex64),
        probeGuess=raw_probe,
    )
    simulation = SimulationConfig(
        N=8,
        probe=ProbeSimulationConfig(
            source="custom",
            source_path=input_path,
            transform_pipeline="pad_extrapolate_boundary_matched:8",
        ),
        object=SyntheticObjectConfig(
            kind="lines", image_size=(32, 32), diffractions_per_object=1
        ),
        scan=ScanSimulationConfig(kind="nongrid", buffer=4),
        detector=DetectorSimulationConfig(photons_per_pattern=1e8),
        seed=3,
    )
    training = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False),
        n_images=1,
        nphotons=1e8,
    )
    captured: dict[str, object] = {}
    fake_raw = SimpleNamespace(
        xcoords=np.array([4]),
        ycoords=np.array([4]),
        xcoords_start=np.array([0]),
        ycoords_start=np.array([0]),
        diff3d=np.ones((1, 8, 8), dtype=np.float32),
        probeGuess=np.ones((8, 8), dtype=np.complex64),
        scan_index=np.array([0]),
    )
    monkeypatch.setattr(
        nongrid_simulation,
        "generate_simulated_data",
        lambda **kwargs: (
            fake_raw,
            np.ones((1, 8, 8), dtype=np.complex64),
        ),
    )
    monkeypatch.setattr(
        "ptycho.metadata.MetadataManager.save_with_metadata",
        lambda path, payload, metadata: captured.update(
            path=path, payload=payload, metadata=metadata
        ),
    )

    cli.simulate_and_save(
        training,
        simulation,
        input_path,
        tmp_path / "out.npz",
        None,
    )

    additional = captured["metadata"]["additional_parameters"]
    assert len(additional["simulation_config_sha256"]) == 64
    assert len(additional["dataset_recipe_sha256"]) == 64
    lineage = additional["probe_lineage"]
    assert lineage["source_path"] == str(input_path)
    assert len(lineage["source_file_sha256"]) == 64
    assert len(lineage["raw_probe_sha256"]) == 64
    assert len(lineage["transformed_probe_sha256"]) == 64
    assert lineage["normalized_transform_pipeline"] == (
        "pad_extrapolate_boundary_matched:8"
    )
    assert lineage["boundary_method"] == "harmonic_dirichlet_c0"
    assert lineage["laplacian_residual"] <= lineage["solver_tolerance"]


def test_generic_simulation_preflights_existing_output_before_generation(
    monkeypatch, tmp_path: Path
):
    from ptycho import nongrid_simulation
    from ptycho.metadata import MetadataManager
    from scripts.simulation import simulate_and_save as cli

    input_path = tmp_path / "input.npz"
    np.savez(
        input_path,
        objectGuess=np.ones((32, 32), dtype=np.complex64),
        probeGuess=np.ones((4, 4), dtype=np.complex64),
    )
    output_path = tmp_path / "out.npz"
    MetadataManager.save_with_metadata(
        str(output_path),
        {"old": np.array([1])},
        {
            "additional_parameters": {
                "simulation_config_sha256": "0" * 64,
                "dataset_recipe_sha256": "1" * 64,
            }
        },
    )
    simulation = SimulationConfig(
        N=8,
        probe=ProbeSimulationConfig(
            source="custom",
            source_path=input_path,
            transform_pipeline="pad_preserve:8",
        ),
        object=SyntheticObjectConfig(
            kind="lines", image_size=(32, 32), diffractions_per_object=1
        ),
        scan=ScanSimulationConfig(kind="nongrid", buffer=4),
    )
    training = TrainingConfig(
        model=ModelConfig(N=8, gridsize=1, object_big=False), n_images=1
    )
    monkeypatch.setattr(
        nongrid_simulation,
        "generate_simulated_data",
        lambda **kwargs: pytest.fail("must reject before generation"),
    )

    with pytest.raises(ValueError, match="Use a distinct output identity"):
        cli.simulate_and_save(
            training,
            simulation,
            input_path,
            output_path,
            None,
        )


@pytest.mark.parametrize(
    "extra_args, message",
    [
        (
            ["--probe-npz", "probe.npz", "--probe-source", "ideal_disk"],
            "probe-npz.*probe-source",
        ),
        (
            [
                "--probe-transform-pipeline",
                "pad_preserve:128",
                "--probe-scale-mode",
                "pad_preserve",
            ],
            "probe-transform-pipeline.*probe-scale-mode",
        ),
        (
            [
                "--probe-transform-pipeline",
                "pad_preserve:128",
                "--probe-smoothing-sigma",
                "0.5",
            ],
            "probe-transform-pipeline.*probe-smoothing-sigma",
        ),
    ],
)
def test_grid_lines_cli_rejects_conflicting_probe_aliases(
    tmp_path: Path, extra_args: list[str], message: str
):
    from scripts.studies import grid_lines_workflow as cli

    config_path = _write_simulation_toml(tmp_path / "sim.toml")
    args = cli.parse_args(
        [
            "--simulation-config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            *extra_args,
        ]
    )

    with pytest.raises(ValueError, match=message):
        cli.build_config(args)


def test_synthetic_lines_runner_rejects_non_lines_recipe(tmp_path: Path):
    from scripts.simulation import run_with_synthetic_lines as cli

    config_path = _write_simulation_toml(
        tmp_path / "dead_leaves.toml", N=64, kind="dead_leaves"
    )
    args, extra = cli.parse_arguments(
        [
            "--simulation-config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
        ]
    )
    assert extra == []
    with pytest.raises(ValueError, match="object.kind='lines'"):
        cli.resolve_synthetic_simulation(args)


def test_synthetic_lines_wrapper_hands_raw_probe_to_the_generic_owner(
    monkeypatch, tmp_path: Path
):
    from scripts.simulation import run_with_synthetic_lines as cli

    raw_probe = np.ones((8, 8), dtype=np.complex64)
    source = tmp_path / "source_probe.npz"
    np.savez(source, probeGuess=raw_probe)
    simulation = SimulationConfig(
        N=16,
        probe=ProbeSimulationConfig(
            source="custom",
            source_path=source,
            transform_pipeline="pad_extrapolate_boundary_matched:16",
        ),
        object=SyntheticObjectConfig(kind="lines", image_size=(32, 32)),
        scan=ScanSimulationConfig(kind="nongrid", buffer=4),
    )
    monkeypatch.setattr(
        "ptycho.diffsim.sim_object_image",
        lambda size: np.ones((size, size, 1), dtype=np.complex64),
    )

    generated = cli.generate_and_save_synthetic_input(tmp_path, simulation)

    with np.load(generated, allow_pickle=False) as archive:
        handed_off = archive["probeGuess"]
    assert handed_off.shape == (8, 8)
    assert np.array_equal(handed_off, raw_probe)
