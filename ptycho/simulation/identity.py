"""Stable simulation-recipe and generated-probe identity helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ptycho.config import (
    SimulationConfig,
    simulation_config_sha256,
    simulation_config_to_dict,
)


def canonical_sha256(value: object) -> str:
    """Hash a JSON-native value with one deterministic encoding."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest()


def build_simulation_probe_lineage(
    simulation: SimulationConfig,
    *,
    raw_probe: np.ndarray,
    normalized_pipeline: str,
    transformed_probe: np.ndarray,
    transform_metadata: Mapping[str, Any],
) -> dict[str, object]:
    """Bind a resolved simulation recipe to source and transformed probe bytes."""

    simulation_payload = simulation_config_to_dict(simulation)
    simulation_digest = simulation_config_sha256(simulation)
    source_path = simulation.probe.source_path
    source_file_sha256 = (
        file_sha256(source_path)
        if source_path is not None and source_path.is_file()
        else None
    )
    raw_probe_digest = array_sha256(raw_probe)
    transformed_probe_digest = array_sha256(transformed_probe)
    solver_identity = {
        key: transform_metadata.get(key)
        for key in ("boundary_method", "solver", "solver_tolerance")
    }
    recipe_payload = {
        "simulation_config_sha256": simulation_digest,
        "source_file_sha256": source_file_sha256,
        "raw_probe_sha256": raw_probe_digest,
        "normalized_transform_pipeline": normalized_pipeline,
        "transformed_probe_sha256": transformed_probe_digest,
        **solver_identity,
    }
    return {
        "simulation_config": simulation_payload,
        "simulation_config_sha256": simulation_digest,
        "dataset_recipe_sha256": canonical_sha256(recipe_payload),
        "probe_lineage": {
            "source_kind": simulation.probe.source,
            "source_path": str(source_path) if source_path is not None else None,
            "source_file_sha256": source_file_sha256,
            "raw_probe_sha256": raw_probe_digest,
            "normalized_transform_pipeline": normalized_pipeline,
            "transformed_probe_sha256": transformed_probe_digest,
            **dict(transform_metadata),
        },
    }


def reject_mismatched_output_identity(
    path: str | Path,
    *,
    expected_simulation_digest: str,
    expected_recipe_digest: str,
) -> None:
    """Fail before overwriting an existing dataset with another recipe."""

    output = Path(path)
    if not output.exists():
        return
    from ptycho.metadata import MetadataManager

    _, metadata = MetadataManager.load_with_metadata(str(output))
    additional = {} if metadata is None else metadata.get("additional_parameters", {})
    existing_simulation = additional.get("simulation_config_sha256")
    existing_recipe = additional.get("dataset_recipe_sha256")
    if (
        existing_simulation != expected_simulation_digest
        or existing_recipe != expected_recipe_digest
    ):
        raise ValueError(
            f"existing dataset {output} has simulation_config_sha256="
            f"{existing_simulation!r}, dataset_recipe_sha256={existing_recipe!r}; "
            f"requested {expected_simulation_digest!r}/{expected_recipe_digest!r}. "
            "Use a distinct output identity."
        )
