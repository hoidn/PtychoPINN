"""Validation cluster extracted from ``ptycho_torch.inference``.

Holds the three workflow/identity validators used by the reconstruction seams
(``validate_bundle_matches_workflow``, ``_validate_flat_npz``,
``_validate_authentic_channels``) plus the private comparison helpers they
share. The reconstruction seams stay in ``ptycho_torch.inference`` and import
back the symbols they need; the validation code never imports
``ptycho_torch.inference`` (keeps the dependency direction one-way).
"""

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
from ptycho.acquisition import decode_acquisition


def _values_agree(left: Any, right: Any) -> bool:
    import torch

    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        try:
            left_tensor = torch.as_tensor(left).detach().cpu()
            right_tensor = torch.as_tensor(right).detach().cpu()
        except (TypeError, ValueError):
            return False
        return (
            left_tensor.shape == right_tensor.shape
            and left_tensor.dtype == right_tensor.dtype
            and bool(torch.equal(left_tensor, right_tensor))
        )
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        try:
            return bool(np.array_equal(np.asarray(left), np.asarray(right)))
        except (TypeError, ValueError):
            return False
    if isinstance(left, Path) or isinstance(right, Path):
        return Path(left) == Path(right)
    return left == right


def _require_record_fields_agree(
    namespace: str,
    expected: Any,
    actual: Any,
    field_names: tuple[str, ...],
    *,
    skipped: frozenset[str] = frozenset(),
) -> None:
    for name in field_names:
        if name in skipped:
            continue
        expected_value = getattr(expected, name)
        actual_value = getattr(actual, name)
        if not _values_agree(expected_value, actual_value):
            raise ValueError(
                f"{namespace}.{name} mismatch: expected {expected_value!r}, "
                f"loaded {actual_value!r}"
            )


def _workflow_field_names(expected: Any, config: Any) -> tuple[str, ...]:
    """Field names present on both a resolved workflow section and a loaded config."""
    return tuple(
        item.name for item in fields(config) if hasattr(expected, item.name)
    )


def validate_bundle_matches_workflow(
    model: Any,
    loader_params: Mapping[str, Any],
    expected_workflow: Any,
    *,
    reconstruction_method: str,
    runtime_inference_config: Any,
    groups_per_center: int,
) -> None:
    """Reject a strictly-loaded bundle whose identity drifts from the caller's
    resolved workflow.

    This is a workflow-conformance gate, not decode-time redundancy: it compares
    the loaded bundle against the caller-resolved synthetic workflow, which
    decode can never see.
    """
    if expected_workflow is None:
        return
    if not (
        is_dataclass(expected_workflow)
        and all(
            hasattr(expected_workflow, name)
            for name in ("data", "model", "training", "inference")
        )
    ):
        raise TypeError(
            "expected_workflow must be a resolved synthetic workflow dataclass"
        )

    data_config = model.data_config
    model_config = model.model_config
    model_fields = tuple(model._model_spec.to_payload()["model_config"])
    gain_record = loader_params.get("amplitude_physics_gain_record")

    _require_record_fields_agree(
        "resolved_workflow.data",
        expected_workflow.data,
        data_config,
        tuple(item.name for item in fields(data_config)),
    )

    expected_provenance = getattr(
        expected_workflow.model,
        "amplitude_physics_gain_provenance",
        None,
    )
    pending_gain = expected_provenance == "pending_training_split_derivation"
    _require_record_fields_agree(
        "resolved_workflow.model",
        expected_workflow.model,
        model_config,
        model_fields,
        skipped=(
            frozenset({"amplitude_physics_gain"})
            if pending_gain
            else frozenset()
        ),
    )
    if pending_gain:
        if gain_record is None or gain_record.provenance != "derived":
            raise ValueError(
                "resolved pending amplitude gain requires the bundle's derived gain sidecar"
            )
    elif gain_record is not None and expected_provenance is not None:
        expected_sidecar_provenance = {
            "explicit": "override",
            "scale_contract_fixed": "scale_contract_fixed",
        }.get(expected_provenance)
        if (
            expected_sidecar_provenance is not None
            and gain_record.provenance != expected_sidecar_provenance
        ):
            raise ValueError(
                "amplitude gain provenance disagrees with resolved_workflow.model"
            )

    _require_record_fields_agree(
        "resolved_workflow.training",
        expected_workflow.training,
        model.training_config,
        _workflow_field_names(expected_workflow.training, model.training_config),
    )
    if not hasattr(expected_workflow.training, "training_groups"):
        raise TypeError(
            "expected_workflow.training must expose training_groups"
        )
    expected_training_groups = expected_workflow.training.training_groups
    if not _values_agree(
        expected_training_groups,
        model.training_config.training_groups,
    ):
        raise ValueError(
            "resolved_workflow.training.training_groups mismatch with "
            "loaded training_config.training_groups: expected "
            f"{expected_training_groups!r}, loaded "
            f"{model.training_config.training_groups!r}"
        )

    _require_record_fields_agree(
        "resolved_workflow.inference",
        expected_workflow.inference,
        model.inference_config,
        _workflow_field_names(expected_workflow.inference, model.inference_config),
    )
    expected_method = getattr(
        expected_workflow.inference,
        "reconstruction_method",
        "barycentric",
    )
    if expected_method != reconstruction_method:
        raise ValueError(
            "expected_workflow.inference.reconstruction_method must be "
            f"{reconstruction_method!r} for reconstruct_npz_{reconstruction_method}"
        )

    expected_groups = getattr(
        expected_workflow.inference,
        "groups_per_center",
        None,
    )
    if expected_groups != groups_per_center:
        raise ValueError(
            "runtime groups_per_center disagrees with "
            "expected_workflow.inference.groups_per_center: "
            f"{groups_per_center!r} != {expected_groups!r}"
        )
    _require_record_fields_agree(
        "runtime inference",
        expected_workflow.inference,
        runtime_inference_config,
        _workflow_field_names(expected_workflow.inference, runtime_inference_config),
    )


def _validate_flat_npz(
    test_data_path: Path,
    data_config: Any,
    *,
    dataset_manifest_path: Optional[Path] = None,
    expected_workflow: Any = None,
    artifact_identity_path: Optional[Path] = None,
) -> None:
    """Validate one held-out acquisition and optional flat-v1 identity."""
    strict_flat_v1 = dataset_manifest_path is not None
    required_dtypes = {
        "xcoords": np.dtype(np.float64),
        "ycoords": np.dtype(np.float64),
        "probeGuess": np.dtype(np.complex64),
    }
    optional_dtypes = {
        "objectGuess": np.dtype(np.complex64),
        "Y": np.dtype(np.complex64),
        "probe_simulated": np.dtype(np.complex64),
        "xcoords_start": np.dtype(np.float64),
        "ycoords_start": np.dtype(np.float64),
        "scan_index": np.dtype(np.int64),
        "object_index": np.dtype(np.int64),
        "object_amplitude_scale": np.dtype(np.float64),
    }
    with np.load(test_data_path, allow_pickle=False) as archive:
        archive_names = set(archive.files)
        if strict_flat_v1 and "diff3d" not in archive_names:
            raise ValueError(
                f"{test_data_path}: flat-v1 NPZ is missing required key 'diff3d'"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}

    record = decode_acquisition(
        test_data_path,
        coordinate_policy="strict" if strict_flat_v1 else "trailing",
    )
    diffraction_name = "diff3d" if "diff3d" in arrays else "diffraction"
    expected_dtypes = {
        diffraction_name: np.dtype(np.float32),
        **required_dtypes,
        **optional_dtypes,
    }
    for name, expected_dtype in expected_dtypes.items():
        if (
            strict_flat_v1
            and name in arrays
            and arrays[name].dtype != expected_dtype
        ):
            raise ValueError(
                f"{test_data_path}: {name} dtype must be {expected_dtype.name}, "
                f"got {arrays[name].dtype.name}"
            )
        if name in arrays and not np.isfinite(arrays[name]).all():
            raise ValueError(f"{test_data_path}: {name} contains nonfinite values")

    diffraction = record.diff3d
    N = int(data_config.N)
    if strict_flat_v1 and arrays["diff3d"].shape != diffraction.shape:
        raise ValueError(
            f"{test_data_path}: diff3d must have shape (M, N, N) "
            f"with loaded N={N}, got {arrays['diff3d'].shape}"
        )
    if diffraction.shape[1:] != (N, N):
        raise ValueError(
            f"{test_data_path}: {diffraction_name} must have shape (M, N, N) "
            f"with loaded N={N}, got {diffraction.shape}"
        )
    if np.any(diffraction < 0):
        raise ValueError(f"{diffraction_name} measurements must be nonnegative")

    if dataset_manifest_path is None:
        return
    manifest_path = Path(dataset_manifest_path)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"dataset manifest does not exist: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot decode dataset manifest {manifest_path}") from error
    manifest_schema = manifest.get("schema_version")
    if manifest_schema not in {
        "flat-acquisition-manifest-v1",
        "flat-acquisition-manifest-v2",
        "flat-acquisition-manifest-v3",
        "flat-acquisition-manifest-v4",
    }:
        raise ValueError(
            "dataset manifest schema_version must be "
            "flat-acquisition-manifest-v1, flat-acquisition-manifest-v2, or "
            "flat-acquisition-manifest-v3, or flat-acquisition-manifest-v4"
        )
    if manifest.get("storage_layout") != "flat_acquisition_v1":
        raise ValueError("dataset manifest storage_layout is not flat_acquisition_v1")
    splits = manifest.get("splits")
    split_record = splits.get("test") if isinstance(splits, Mapping) else None
    if not isinstance(split_record, Mapping):
        raise ValueError("dataset manifest is missing splits.test")
    if manifest_schema == "flat-acquisition-manifest-v4":
        object_record = manifest.get("object")
        if not isinstance(object_record, Mapping) or object_record.get(
            "recipe"
        ) != "frozen-object-bank-v1":
            raise ValueError(
                "flat-acquisition-manifest-v4 requires frozen-object-bank-v1"
            )
    if expected_workflow is not None:
        if not hasattr(expected_workflow, "simulation"):
            raise TypeError(
                "expected_workflow must expose simulation identity when a "
                "dataset manifest is supplied"
            )
        from ptycho.config import (
            simulation_config_sha256,
            simulation_config_to_dict,
        )
        from ptycho.simulation.flat_acquisition import (
            _source_seed_lineage,
            derive_seed_lineage,
        )
        from ptycho.workflows.synthetic_config import (
            synthetic_simulation_compatibility_identity,
            synthetic_workflow_to_dict,
        )

        expected_semantic = synthetic_workflow_to_dict(expected_workflow)
        if manifest.get("profile") != expected_workflow.profile:
            raise ValueError("dataset manifest profile disagrees with resolved workflow")
        if manifest.get("recipe_version") != expected_workflow.recipe_version:
            raise ValueError(
                "dataset manifest recipe_version disagrees with resolved workflow"
            )
        if synthetic_simulation_compatibility_identity(
            manifest.get("simulation")
        ) != synthetic_simulation_compatibility_identity(
            expected_semantic["simulation"]
        ):
            raise ValueError(
                "dataset manifest simulation disagrees with resolved workflow"
            )
        expected_seed_lineage = derive_seed_lineage(
            expected_workflow.simulation.train.seed
        )
        frozen_source = (
            expected_workflow.simulation.object_recipe
            == "frozen-object-bank-v1"
        )
        if (manifest_schema == "flat-acquisition-manifest-v4") != frozen_source:
            raise ValueError(
                "flat-acquisition-manifest-v4 and frozen-object-bank-v1 must "
                "be used together"
            )
        if frozen_source:
            expected_seed_lineage = _source_seed_lineage(
                expected_seed_lineage
            )
        if manifest.get("seed_lineage") != expected_seed_lineage:
            raise ValueError(
                "dataset manifest seed_lineage disagrees with resolved workflow"
            )
        expected_test_simulation = expected_workflow.simulation.test
        if split_record.get("simulation_config") != simulation_config_to_dict(
            expected_test_simulation
        ):
            raise ValueError(
                "dataset manifest splits.test simulation_config disagrees "
                "with resolved workflow"
            )
        if split_record.get(
            "simulation_config_sha256"
        ) != simulation_config_sha256(expected_test_simulation):
            raise ValueError(
                "dataset manifest splits.test simulation_config_sha256 mismatch"
            )
        if split_record.get("seed_lineage") != expected_seed_lineage:
            raise ValueError(
                "dataset manifest splits.test seed_lineage disagrees with "
                "resolved workflow"
            )
        for field_name, lineage_name in (
            ("coordinate_seed", "test_coordinates"),
            ("detector_seed", "test_noise"),
        ):
            if split_record.get(field_name) != expected_seed_lineage[lineage_name]:
                raise ValueError(
                    f"dataset manifest splits.test {field_name} disagrees "
                    "with resolved workflow"
                )
    artifact_path = manifest_path.parent / str(split_record.get("artifact_path", ""))
    identity_path = (
        test_data_path
        if artifact_identity_path is None
        else Path(artifact_identity_path)
    )
    if artifact_path.resolve() != identity_path.resolve():
        raise ValueError(
            "dataset manifest splits.test artifact_path does not identify held-out NPZ"
        )
    from ptycho.simulation.flat_acquisition import validate_split_manifest_record

    validate_split_manifest_record(
        test_data_path,
        split_record,
        split="test",
        split_recipe_sha256=split_record.get("split_recipe_sha256"),
    )
    normalization_record = split_record.get("object_amplitude_normalization")
    if "object_amplitude_scale" in arrays:
        if not isinstance(normalization_record, Mapping):
            raise ValueError(
                "dataset manifest splits.test object amplitude normalization is missing"
            )
        scale = np.asarray(arrays["object_amplitude_scale"])
        if scale.shape != () or not _values_agree(
            float(scale), normalization_record.get("scale")
        ):
            raise ValueError(
                "dataset manifest splits.test object amplitude scale mismatch"
            )
    elif normalization_record is not None:
        raise ValueError(
            "dataset manifest splits.test object amplitude normalization has no NPZ scale"
        )
    measurement = split_record.get("measurement_identity")
    if not isinstance(measurement, Mapping):
        raise ValueError(
            "dataset manifest splits.test measurement_identity must be a mapping"
        )
    if measurement.get("scale_contract_version") != data_config.scale_contract_version:
        raise ValueError("dataset manifest scale_contract_version disagrees with bundle")
    if measurement.get("measurement_domain") != data_config.measurement_domain:
        raise ValueError("dataset manifest measurement_domain disagrees with bundle")
    photons = measurement.get("photons_per_pattern")
    if not _values_agree(photons, float(data_config.nphotons)):
        raise ValueError(
            "dataset manifest photons_per_pattern disagrees with bundle "
            f"DataConfig.nphotons: {photons!r} != {data_config.nphotons!r}"
        )


def _validate_authentic_channels(
    dataset: Any,
    data_config: Any,
) -> tuple[set[int], int, np.ndarray, np.ndarray]:
    """Reject any grouped representation that collapsed the C4 scan identity."""
    import torch

    mmap = getattr(dataset, "mmap_ptycho", None)
    if mmap is None:
        raise ValueError("PtychoDataset did not expose mmap_ptycho")
    try:
        images = torch.as_tensor(mmap["images"])
        indices = torch.as_tensor(mmap["nn_indices"])
        coords = torch.as_tensor(mmap["coords_global"])
    except (KeyError, TypeError) as error:
        raise ValueError(
            "mmap dataset is missing images/nn_indices/coords_global channel identity"
        ) from error
    C = data_config.gridsize * data_config.gridsize
    if (
        images.ndim < 2
        or images.shape[1] != C
        or indices.ndim != 2
        or indices.shape[1] != C
        or coords.ndim != 4
        or coords.shape[1:] != (C, 1, 2)
        or indices.shape[0] == 0
        or indices.shape[0] != coords.shape[0]
        or indices.shape[0] != images.shape[0]
    ):
        raise ValueError(
            f"C4 channel identity was collapsed: expected aligned (groups, C={C}) mmap tensors"
        )
    index_rows = indices.detach().cpu().numpy()
    if not np.issubdtype(index_rows.dtype, np.integer):
        raise ValueError("C4 channel scan identities must be integer indices")
    if np.any(index_rows < 0):
        raise ValueError("C4 channel scan identities must be nonnegative")
    grouping_enabled = getattr(dataset, "group_coords_enabled", None)
    grouped = bool(grouping_enabled()) if callable(grouping_enabled) else C > 1
    if not grouped:
        # Ungrouped layouts persist LOCAL row indices in nn_indices while the
        # reassembly identity evidence reports SOURCE scan ids
        # (dataloader writes index_range vs valid_indices; reassembly remaps
        # via filtered[used]). Map to the source space so both sides of the
        # coverage gate speak the same identity language.
        valid_per_file = getattr(dataset, "valid_indices_per_file", None)
        if not valid_per_file or len(valid_per_file) != 1:
            raise ValueError(
                "ungrouped scan-identity mapping requires exactly one source file"
            )
        filtered = np.asarray(valid_per_file[0], dtype=np.int64).reshape(-1)
        if index_rows.size and index_rows.max() >= filtered.size:
            raise ValueError("Ungrouped scan ids are outside the filtered split")
        index_rows = filtered[index_rows]
    coord_rows = coords.detach().cpu().numpy().reshape(indices.shape[0], C, 2)
    if C > 1:
        for index_row, coord_row in zip(index_rows, coord_rows, strict=True):
            if len(set(int(item) for item in index_row)) != C:
                raise ValueError("C4 channel scan identities must be distinct within every group")
            if len({tuple(float(item) for item in coord) for coord in coord_row}) != C:
                raise ValueError("C4 channel global coordinates must be distinct within every group")
    channel_indices = np.array(index_rows, dtype=np.int64, copy=True)
    channel_coordinates = np.array(coord_rows, dtype=np.float64, copy=True)
    channel_indices.setflags(write=False)
    channel_coordinates.setflags(write=False)
    flat_indices = channel_indices.reshape(-1).tolist()
    return (
        set(int(item) for item in flat_indices),
        len(flat_indices),
        channel_indices,
        channel_coordinates,
    )
