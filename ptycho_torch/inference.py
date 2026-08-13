"""
PyTorch Inference Module for Ptychography Reconstruction

The canonical CLI loads versioned model bundles, performs reconstruction, and
generates amplitude/phase PNGs.

Usage Examples:

  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data datasets/Run1084_recon3_postPC_shrunk_3.npz \\
      --output_dir inference_outputs \\
      --n_images 32 \\
      --device cpu

References:
  - Phase E2 plan: plans/active/INTEGRATE-PYTORCH-001/phase_e2_implementation.md §E2.C2
  - Test contract: tests/torch/test_integration_workflow_torch.py
  - Red phase evidence: plans/active/INTEGRATE-PYTORCH-001/reports/2025-10-17T213500Z/red_phase.md §2.3
"""

#Generic
import os
import argparse
import copy
import gc
import json
import shutil
import sys
import tempfile
from dataclasses import dataclass, fields, is_dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING

import numpy as np
from ptycho.acquisition import decode_acquisition
from ptycho.config.legacy_state import scoped_legacy_params
from ptycho.reconstruction_policy import OutputSpec, resolve_cli_reconstruction_policy
from ptycho_torch.reconstruction_ports import present_reconstruction_canvas

#ML libraries
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import torch
    from ptycho_torch.config_params import DataConfig
    from ptycho_torch.reassembly_diagnostics import ReassemblyDiagnostics

def _training_normalization_scale(diffraction: "torch.Tensor") -> "torch.Tensor":
    """
    Match RawData.normalize_data() normalization used in training.

    RawData.normalize_data() computes:
        scale = sqrt(((N/2)^2) / mean(sum(diffraction^2)))

    Returns a (B, 1, 1, 1) tensor for broadcast with (B, C, H, W).
    """
    import torch

    if diffraction.ndim == 4:
        diff = diffraction.squeeze(1)
    else:
        diff = diffraction

    mean_sum = torch.mean(torch.sum(diff ** 2, dim=(-2, -1)))
    if mean_sum.item() <= 0:
        # Fall back to unity scaling when diffraction is all zeros (test fixtures / degenerate inputs).
        return torch.ones((diffraction.shape[0], 1, 1, 1), device=diffraction.device, dtype=diffraction.dtype)

    n = float(diff.shape[-1])
    scale = torch.sqrt(torch.tensor((n / 2.0) ** 2, device=diffraction.device, dtype=diffraction.dtype) / mean_sum)
    return scale.view(1, 1, 1, 1).expand(diffraction.shape[0], 1, 1, 1)


def save_individual_reconstructions(obj_amp, obj_phase, output_dir):
    """
    Save individual amplitude and phase reconstructions as separate PNG files.

    This function generates the specific output artifacts expected by the PyTorch
    integration test workflow (Phase E2.C2).

    Args:
        obj_amp: Reconstructed amplitude array (numpy array)
        obj_phase: Reconstructed phase array (numpy array)
        output_dir: Directory to save output images (str or Path)

    Outputs:
        - <output_dir>/reconstructed_amplitude.png
        - <output_dir>/reconstructed_phase.png
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create amplitude figure
    fig_amp, ax_amp = plt.subplots(figsize=(6, 6))
    try:
        im_amp = ax_amp.imshow(obj_amp, cmap='gray')
        plt.colorbar(im_amp, ax=ax_amp)
        ax_amp.set_title('Reconstructed Amplitude')
        ax_amp.axis('off')

        amp_path = output_dir / "reconstructed_amplitude.png"
        plt.savefig(amp_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig_amp)
    print(f"Saved amplitude reconstruction to: {amp_path}")

    # Create phase figure
    fig_phase, ax_phase = plt.subplots(figsize=(6, 6))
    try:
        im_phase = ax_phase.imshow(obj_phase, cmap='gray')
        plt.colorbar(im_phase, ax=ax_phase)
        ax_phase.set_title('Reconstructed Phase')
        ax_phase.axis('off')

        phase_path = output_dir / "reconstructed_phase.png"
        plt.savefig(phase_path, dpi=150, bbox_inches='tight')
    finally:
        plt.close(fig_phase)
    print(f"Saved phase reconstruction to: {phase_path}")


def _resolve_reassembly_route(patch_weighting, varpro_scaling):
    """
    Decide which stitching path honors the requested inference knobs
    (Conformance D4, 2026-07-14 CI paper-conformance audit Theme 2.1).

    Args:
        patch_weighting: 'uniform' (legacy binary/uniform stitching) or 'probe'
            (|P|^2-weighted barycentric assembly).
        varpro_scaling: Whether the VarPro (s1, s2) least-squares intensity
            refit is requested.

    Returns:
        'uniform' when neither knob deviates from the legacy CLI behavior
        (keeps that path bit-identical), else 'barycentric'.

    Raises:
        ValueError: patch_weighting is not one of 'uniform' / 'probe'.
    """
    return resolve_cli_reconstruction_policy(
        patch_weighting,
        varpro_scaling,
    ).compatibility_route


def _describe_requested_knobs(patch_weighting, varpro_scaling):
    """Human-readable list of the non-default stitching/scaling knobs."""
    requested = []
    if patch_weighting != 'uniform':
        requested.append(f"patch_weighting={patch_weighting!r}")
    if varpro_scaling:
        requested.append("varpro_scaling=True")
    return ", ".join(requested)


def _require_ci_varpro_scaling(model, inference_config):
    """Fail closed when active CI inference would omit physical scaling."""
    from ptycho_torch.scaling_contract import CI_SCALE_CONTRACT, COUNT_INTENSITY

    model_config = getattr(model, "model_config", None)
    data_config = getattr(model, "data_config", None)
    active_ci = (
        getattr(model_config, "physics_forward_mode", "amplitude")
        == "rectangular_scaled"
        and getattr(data_config, "scale_contract_version", None)
        == CI_SCALE_CONTRACT
        and getattr(data_config, "measurement_domain", None) == COUNT_INTENSITY
    )
    if active_ci and not bool(getattr(inference_config, "varpro_scaling", False)):
        raise ValueError(
            "Active CI inference requires --varpro-scaling so the reported "
            "reconstruction is in the physical count-consistent scale."
        )


def _snapshot_array(value: Any, *, name: str, complex_required: bool = False) -> np.ndarray:
    """Detach one reconstruction artifact from CUDA and mmap ownership."""
    import torch

    if isinstance(value, torch.Tensor):
        array = value.detach().to("cpu").contiguous().numpy()
    else:
        array = np.asarray(value)
    array = np.array(array, copy=True)
    if array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-2 canvas, got shape {array.shape}")
    if complex_required and not np.iscomplexobj(array):
        raise ValueError(f"{name} must be complex")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class BarycentricReconstructionResult:
    """Self-contained output of strict-load mmap barycentric reconstruction."""

    complex_canvas: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    prescale_canvas: np.ndarray
    effective_data_config: "DataConfig"
    canvas_weights: np.ndarray
    canvas_anchor: Mapping[str, Any]
    channel_indices: np.ndarray
    reassembly: "ReassemblyDiagnostics"


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


def _validate_loaded_reconstruction_identity(
    model: Any,
    loader_params: Mapping[str, Any],
    expected_workflow: Any = None,
) -> None:
    """Join the strict loader's dual-written identity before any mmap work."""
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )
    from ptycho_torch.model_spec import ModelSpec
    from ptycho_torch.scaling_contract import (
        resolve_scale_contract,
        validate_scale_contract,
    )

    required = (
        ("data_config", DataConfig),
        ("model_config", ModelConfig),
        ("training_config", TrainingConfig),
        ("inference_config", InferenceConfig),
    )
    for name, expected_type in required:
        value = getattr(model, name, None)
        if not isinstance(value, expected_type):
            raise ValueError(
                f"strict bundle model is missing required {name} ({expected_type.__name__})"
            )

    model_spec = getattr(model, "_model_spec", None)
    if not isinstance(model_spec, ModelSpec):
        raise ValueError("strict bundle model is missing required persisted ModelSpec")
    spec_config = model_spec.to_model_config()
    # ModelSpec deliberately excludes the legacy derived ``object_big`` alias;
    # its payload keys are the authoritative dual-written structural surface.
    model_fields = tuple(model_spec.to_payload()["model_config"])
    _require_record_fields_agree(
        "ModelSpec/model_config", spec_config, model.model_config, model_fields
    )

    data_config = model.data_config
    model_config = model.model_config
    resolve_scale_contract(
        data_config.scale_contract_version,
        data_config.measurement_domain,
    )
    validate_scale_contract(
        data_config,
        model_config,
        model.training_config,
    )
    if model_config.C_model != data_config.C:
        raise ValueError(
            f"model_config.C_model={model_config.C_model} conflicts with data_config.C={data_config.C}"
        )
    if model_config.C_forward != data_config.C:
        raise ValueError(
            f"model_config.C_forward={model_config.C_forward} conflicts with data_config.C={data_config.C}"
        )
    if (
        model_config.object_layout == "grouped_patches"
        and int(data_config.grid_size[0]) * int(data_config.grid_size[1])
        != data_config.C
    ):
        raise ValueError(
            "grouped model channel count must equal data_config.grid_size product"
        )

    gain_record = loader_params.get("amplitude_physics_gain_record")
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

    expected_data_fields = tuple(item.name for item in fields(data_config))
    _require_record_fields_agree(
        "resolved_workflow.data",
        expected_workflow.data,
        data_config,
        expected_data_fields,
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

    training_names = tuple(
        item.name
        for item in fields(model.training_config)
        if hasattr(expected_workflow.training, item.name)
    )
    _require_record_fields_agree(
        "resolved_workflow.training",
        expected_workflow.training,
        model.training_config,
        training_names,
    )
    if not hasattr(expected_workflow.training, "training_groups"):
        raise TypeError(
            "expected_workflow.training must expose training_groups"
        )
    expected_training_groups = expected_workflow.training.training_groups
    if not _values_agree(
        expected_training_groups,
        model.training_config.n_groups,
    ):
        raise ValueError(
            "resolved_workflow.training.training_groups mismatch with "
            "loaded training_config.n_groups: expected "
            f"{expected_training_groups!r}, loaded "
            f"{model.training_config.n_groups!r}"
        )
    inference_names = tuple(
        item.name
        for item in fields(model.inference_config)
        if hasattr(expected_workflow.inference, item.name)
    )
    _require_record_fields_agree(
        "resolved_workflow.inference",
        expected_workflow.inference,
        model.inference_config,
        inference_names,
    )
    if getattr(
        expected_workflow.inference,
        "reconstruction_method",
        "barycentric",
    ) != "barycentric":
        raise ValueError(
            "expected_workflow.inference.reconstruction_method must be "
            "'barycentric' for reconstruct_npz_barycentric"
        )


def _validate_expected_runtime_reconstruction(
    expected_workflow: Any,
    runtime_inference_config: Any,
    *,
    groups_per_center: int,
) -> None:
    """Reject runtime reconstruction knobs that drift from resolved identity."""
    if expected_workflow is None:
        return
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
    names = tuple(
        item.name
        for item in fields(runtime_inference_config)
        if hasattr(expected_workflow.inference, item.name)
    )
    _require_record_fields_agree(
        "runtime inference",
        expected_workflow.inference,
        runtime_inference_config,
        names,
    )


def _validate_flat_npz(
    test_data_path: Path,
    data_config: Any,
    *,
    dataset_manifest_path: Optional[Path] = None,
    expected_workflow: Any = None,
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
    if manifest.get("schema_version") != "flat-acquisition-manifest-v1":
        raise ValueError("dataset manifest schema_version is not flat-acquisition-manifest-v1")
    if manifest.get("storage_layout") != "flat_acquisition_v1":
        raise ValueError("dataset manifest storage_layout is not flat_acquisition_v1")
    splits = manifest.get("splits")
    split_record = splits.get("test") if isinstance(splits, Mapping) else None
    if not isinstance(split_record, Mapping):
        raise ValueError("dataset manifest is missing splits.test")
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
        from ptycho.simulation.flat_acquisition import derive_seed_lineage
        from ptycho.workflows.synthetic_config import synthetic_workflow_to_dict

        expected_semantic = synthetic_workflow_to_dict(expected_workflow)
        if manifest.get("profile") != expected_workflow.profile:
            raise ValueError("dataset manifest profile disagrees with resolved workflow")
        if manifest.get("recipe_version") != expected_workflow.recipe_version:
            raise ValueError(
                "dataset manifest recipe_version disagrees with resolved workflow"
            )
        if manifest.get("simulation") != expected_semantic["simulation"]:
            raise ValueError(
                "dataset manifest simulation disagrees with resolved workflow"
            )
        expected_seed_lineage = derive_seed_lineage(
            expected_workflow.simulation.train.seed
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
    if artifact_path.resolve() != test_data_path.resolve():
        raise ValueError(
            "dataset manifest splits.test artifact_path does not identify held-out NPZ"
        )
    from ptycho.simulation.identity import (
        array_sha256,
        canonical_sha256,
        file_sha256,
    )

    recorded_npz_sha256 = split_record.get("npz_sha256")
    if recorded_npz_sha256 != file_sha256(test_data_path):
        raise ValueError("dataset manifest splits.test npz_sha256 mismatch")
    recorded_array_hashes = split_record.get("array_sha256")
    if not isinstance(recorded_array_hashes, Mapping):
        raise ValueError(
            "dataset manifest splits.test array_sha256 must be a mapping"
        )
    computed_array_hashes = {
        name: array_sha256(value) for name, value in arrays.items()
    }
    computed_shapes = {
        name: list(value.shape) for name, value in arrays.items()
    }
    computed_dtypes = {
        name: value.dtype.name for name, value in arrays.items()
    }
    if dict(recorded_array_hashes) != computed_array_hashes:
        raise ValueError("dataset manifest splits.test array_sha256 mismatch")
    expected_dataset_identity = {
        "split_recipe_sha256": split_record.get("split_recipe_sha256"),
        "array_sha256": computed_array_hashes,
        "shapes": computed_shapes,
        "dtypes": computed_dtypes,
    }
    if split_record.get("dataset_identity") != expected_dataset_identity:
        raise ValueError("dataset manifest splits.test dataset_identity mismatch")
    if split_record.get("dataset_sha256") != canonical_sha256(
        expected_dataset_identity
    ):
        raise ValueError("dataset manifest splits.test dataset_sha256 mismatch")
    shapes = split_record.get("shapes")
    if not isinstance(shapes, Mapping):
        raise ValueError("dataset manifest splits.test shapes must be a mapping")
    if dict(shapes) != computed_shapes:
        raise ValueError("dataset manifest splits.test shapes mismatch")
    dtypes = split_record.get("dtypes")
    if not isinstance(dtypes, Mapping):
        raise ValueError("dataset manifest splits.test dtypes must be a mapping")
    if dict(dtypes) != computed_dtypes:
        raise ValueError("dataset manifest splits.test dtypes mismatch")
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
) -> tuple[set[int], int, np.ndarray]:
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
    C = int(data_config.C)
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
    if C > 1:
        coord_rows = coords.detach().cpu().numpy().reshape(indices.shape[0], C, 2)
        for index_row, coord_row in zip(index_rows, coord_rows, strict=True):
            if len(set(int(item) for item in index_row)) != C:
                raise ValueError("C4 channel scan identities must be distinct within every group")
            if len({tuple(float(item) for item in coord) for coord in coord_row}) != C:
                raise ValueError("C4 channel global coordinates must be distinct within every group")
    channel_indices = np.array(index_rows, dtype=np.int64, copy=True)
    channel_indices.setflags(write=False)
    flat_indices = channel_indices.reshape(-1).tolist()
    return (
        set(int(item) for item in flat_indices),
        len(flat_indices),
        channel_indices,
    )


def _reconstruct_loaded_npz_barycentric(
    model: Any,
    test_data_path: Path,
    *,
    run_root: Path,
    groups_per_center: int,
    inference_config: Any,
    device: str,
    num_workers: int,
    inference_batch_size: Optional[int],
    precision: str,
    quiet: bool,
) -> BarycentricReconstructionResult:
    from ptycho_torch.dataloader import PtychoDataset
    from ptycho_torch.reassembly import reconstruct_image_barycentric
    from ptycho_torch.reassembly_diagnostics import ReassemblyDiagnostics

    runtime_data_config = replace(
        model.data_config,
        n_subsample=groups_per_center,
    )
    runtime_training_config = replace(
        model.training_config,
        device=str(device),
        num_workers=num_workers,
    )
    runtime_inference_config = inference_config
    if inference_batch_size is not None:
        runtime_inference_config = replace(
            runtime_inference_config,
            batch_size=inference_batch_size,
        )
    _require_ci_varpro_scaling(model, runtime_inference_config)
    requested = _describe_requested_knobs(
        runtime_inference_config.patch_weighting,
        runtime_inference_config.varpro_scaling,
    )
    if not quiet:
        print(f"Reassembly route: barycentric ({requested})")

    dataset = None
    reconstruction_dataset = None
    with tempfile.TemporaryDirectory(
        prefix="barycentric-workspace-",
        dir=run_root,
    ) as workspace_name:
        workspace = Path(workspace_name)
        staged_dir = workspace / "staged"
        staged_dir.mkdir()
        shutil.copy2(test_data_path, staged_dir / test_data_path.name)
        try:
            dataset = PtychoDataset(
                str(staged_dir),
                model.model_config,
                runtime_data_config,
                training_config=runtime_training_config,
                data_dir=str(workspace / "mmap" / "memmap"),
                remake_map=True,
                require_complete_group_coverage=(
                    runtime_data_config.neighbor_function == "Nearest"
                    and groups_per_center == 1
                ),
            )
            (
                expected_scan_ids,
                expected_patch_count,
                channel_indices,
            ) = _validate_authentic_channels(dataset, runtime_data_config)
            canvas, reconstruction_dataset, diagnostics, prescale_canvas = (
                reconstruct_image_barycentric(
                    model,
                    dataset,
                    runtime_training_config,
                    runtime_data_config,
                    model.model_config,
                    runtime_inference_config,
                    gpu_ids=None,
                    verbose=not quiet,
                    structured_diagnostics=True,
                    precision=precision,
                )
            )
            if not isinstance(diagnostics, ReassemblyDiagnostics):
                raise TypeError(
                    "reconstruct_image_barycentric did not return structured "
                    "ReassemblyDiagnostics"
                )
            if not expected_scan_ids.issubset(set(diagnostics.used_scan_ids)):
                missing = sorted(expected_scan_ids - set(diagnostics.used_scan_ids))
                raise ValueError(
                    "barycentric reassembly did not use every C4 channel scan "
                    f"id: {missing}"
                )
            if (
                diagnostics.total_patches != expected_patch_count
                or diagnostics.accepted_patches != expected_patch_count
            ):
                raise ValueError(
                    "barycentric reassembly must accept every authentic C4 "
                    f"channel patch ({diagnostics.accepted_patches}/"
                    f"{diagnostics.total_patches}, expected {expected_patch_count})"
                )
            policy = resolve_cli_reconstruction_policy(
                runtime_inference_config.patch_weighting,
                runtime_inference_config.varpro_scaling,
            )
            amplitude, phase = present_reconstruction_canvas(canvas, policy.output)
            canvas_snapshot = _snapshot_array(
                canvas, name="complex_canvas", complex_required=True
            )
            prescale_snapshot = _snapshot_array(
                prescale_canvas, name="prescale_canvas", complex_required=True
            )
            amplitude_snapshot = _snapshot_array(amplitude, name="amplitude")
            phase_snapshot = _snapshot_array(phase, name="phase")
            weight_snapshot = _snapshot_array(
                diagnostics.canvas_weights,
                name="canvas weights",
            )
            if weight_snapshot.shape != canvas_snapshot.shape:
                raise ValueError("canvas weights shape must match the complex canvas")
            if not bool(np.any(weight_snapshot > 0)):
                raise ValueError(
                    "canvas weights must contain nonempty positive support"
                )
            anchor_snapshot = copy.deepcopy(dict(diagnostics.canvas_anchor))
            if "scan_com" not in anchor_snapshot:
                raise ValueError(
                    "structured reassembly diagnostics are missing scan_com anchor"
                )
            result = BarycentricReconstructionResult(
                complex_canvas=canvas_snapshot,
                amplitude=amplitude_snapshot,
                phase=phase_snapshot,
                prescale_canvas=prescale_snapshot,
                effective_data_config=runtime_data_config,
                canvas_weights=weight_snapshot,
                canvas_anchor=anchor_snapshot,
                channel_indices=channel_indices,
                reassembly=diagnostics,
            )
        finally:
            # TensorDict mmap handles must be unreachable before the owned
            # TemporaryDirectory attempts cleanup, including error paths.
            del reconstruction_dataset
            dataset = None
            gc.collect()

    if not quiet:
        print(f"Reconstruction shape: {result.amplitude.shape}")
        print(
            f"Amplitude range: [{result.amplitude.min():.4f}, "
            f"{result.amplitude.max():.4f}]"
        )
        print(
            f"Phase range: [{result.phase.min():.4f}, {result.phase.max():.4f}]"
        )
    return result


def reconstruct_npz_barycentric(
    bundle_path: os.PathLike[str] | str,
    test_npz_path: os.PathLike[str] | str,
    *,
    run_root: os.PathLike[str] | str,
    groups_per_center: int = 1,
    expected_workflow: Any = None,
    dataset_manifest_path: Optional[os.PathLike[str] | str] = None,
    scale_contract_version: Optional[str] = None,
    measurement_domain: Optional[str] = None,
    inference_config: Any = None,
    device: str = "cpu",
    num_workers: int = 0,
    inference_batch_size: Optional[int] = None,
    precision: str = "32-true",
    quiet: bool = False,
) -> BarycentricReconstructionResult:
    """Strictly reload one bundle and reconstruct one flat NPZ through mmap."""
    from ptycho.config.legacy_state import isolated_archived_params_scope
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.workflows.components import load_inference_bundle_torch

    if isinstance(groups_per_center, bool) or not isinstance(groups_per_center, int):
        raise TypeError("groups_per_center must be a positive integer")
    if groups_per_center <= 0:
        raise ValueError("groups_per_center must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be a nonnegative integer")
    if num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")
    if inference_batch_size is not None and (
        isinstance(inference_batch_size, bool)
        or not isinstance(inference_batch_size, int)
        or inference_batch_size <= 0
    ):
        raise ValueError("inference_batch_size must be a positive integer or None")

    bundle_input = Path(bundle_path)
    bundle_dir = (
        bundle_input.parent
        if bundle_input.name == "wts.h5.zip"
        else bundle_input
    )
    archive_path = bundle_dir / "wts.h5.zip"
    if not archive_path.is_file() or archive_path.stat().st_size <= 0:
        raise FileNotFoundError(
            f"strict reconstruction requires a nonempty {archive_path}"
        )
    test_data_path = Path(test_npz_path)
    if not test_data_path.is_file() or test_data_path.suffix.lower() != ".npz":
        raise FileNotFoundError(f"held-out NPZ does not exist: {test_data_path}")
    output_root = Path(run_root)
    output_root.mkdir(parents=True, exist_ok=True)
    if not output_root.is_dir():
        raise NotADirectoryError(f"run_root is not a directory: {output_root}")

    with isolated_archived_params_scope():
        models, loader_params = load_inference_bundle_torch(
            bundle_dir,
            model_name="diffraction_to_obj",
            scale_contract_version=scale_contract_version,
            measurement_domain=measurement_domain,
        )
    if "diffraction_to_obj" not in models:
        raise ValueError("strict bundle did not contain diffraction_to_obj")
    model = models["diffraction_to_obj"]
    _validate_loaded_reconstruction_identity(
        model,
        loader_params,
        expected_workflow=expected_workflow,
    )
    _validate_flat_npz(
        test_data_path,
        model.data_config,
        dataset_manifest_path=(
            Path(dataset_manifest_path)
            if dataset_manifest_path is not None
            else None
        ),
        expected_workflow=expected_workflow,
    )
    if inference_config is not None and not isinstance(
        inference_config, InferenceConfig
    ):
        raise TypeError("inference_config must be a Torch InferenceConfig")
    runtime_inference_config = model.inference_config
    if inference_config is not None:
        # Treat the caller record as an inference-knob carrier only. Geometry,
        # trimming, and other serialized identity continue to start from the
        # strictly loaded configuration.
        runtime_inference_config = replace(
            runtime_inference_config,
            patch_weighting=inference_config.patch_weighting,
            varpro_scaling=inference_config.varpro_scaling,
            log_patch_stats=inference_config.log_patch_stats,
            patch_stats_limit=inference_config.patch_stats_limit,
        )
    if inference_batch_size is not None:
        runtime_inference_config = replace(
            runtime_inference_config,
            batch_size=inference_batch_size,
        )
    _validate_expected_runtime_reconstruction(
        expected_workflow,
        runtime_inference_config,
        groups_per_center=groups_per_center,
    )
    return _reconstruct_loaded_npz_barycentric(
        model,
        test_data_path,
        run_root=output_root,
        groups_per_center=groups_per_center,
        inference_config=runtime_inference_config,
        device=device,
        num_workers=num_workers,
        inference_batch_size=None,
        precision=precision,
        quiet=quiet,
    )


def _run_barycentric_inference_and_reconstruct(
    model,
    test_data_path,
    pt_inference_config,
    execution_config,
    device,
    output_dir,
    quiet=False,
):
    """Compatibility adapter for already-loaded callers; public CLIs load once."""
    result = _reconstruct_loaded_npz_barycentric(
        model,
        Path(test_data_path),
        run_root=Path(output_dir),
        groups_per_center=1,
        inference_config=pt_inference_config,
        device=device,
        num_workers=int(getattr(execution_config, "num_workers", 0) or 0),
        inference_batch_size=getattr(
            execution_config, "inference_batch_size", None
        ),
        precision=getattr(execution_config, "precision", "32-true"),
        quiet=quiet,
    )
    return result.amplitude, result.phase


def _run_inference_and_reconstruct(model, raw_data, config, execution_config, device, quiet=False, intensity_scale=None):
    """
    Extract inference logic into testable helper function (Phase D.C C3).

    Args:
        model: Loaded Lightning module (should be in eval mode)
        raw_data: RawData instance with test data
        config: TFInferenceConfig with n_groups, etc.
        execution_config: PyTorchExecutionConfig with device, batch size, etc.
        device: Torch device string ('cpu', 'cuda', 'mps')
        quiet: Suppress progress output (default: False)

    Returns:
        Tuple of (amplitude, phase) numpy arrays

    Notes:
        - Wraps existing simplified inference logic (lines 563-641)
        - Enforces DTYPE-001 (float32 for diffraction, complex64 for probe)
        - Averages across batch for single reconstruction
        - DEVICE-MISMATCH-001: Ensures model is on the correct device
    """
    import torch
    from ptycho_torch.scaling_contract import (
        CI_SCALE_CONTRACT,
        ci_scaling_active,
        resolve_scale_contract,
    )

    model_config = getattr(model, "model_config", None)
    data_config = getattr(model, "data_config", None)
    if model_config is not None and ci_scaling_active(model_config):
        profile = resolve_scale_contract(
            getattr(data_config, "scale_contract_version", None),
            getattr(data_config, "measurement_domain", None),
        )
        if profile.version == CI_SCALE_CONTRACT:
            raise RuntimeError(
                "ci_intensity_v2 inference requires the canonical "
                "reconstruct_image_barycentric physical-probe VarPro path. The "
                "simplified uniform-stitching path cannot produce CI-scaled output. "
                "Re-run with --patch-weighting probe --varpro-scaling to route "
                "through it."
            )

    # DEVICE-MISMATCH-001 fix: Ensure model is on the requested device and in eval mode
    model.to(device)
    model.eval()

    # DTYPE ENFORCEMENT (Phase D1d): Cast to float32 per DATA-001
    diffraction = torch.from_numpy(raw_data.diff3d).to(device, dtype=torch.float32)
    probe = torch.from_numpy(raw_data.probeGuess).to(device, dtype=torch.complex64)

    from ptycho import debug_parity
    debug_parity.log_array_stats("torch.diffraction_raw", raw_data.diff3d)
    debug_parity.log_array_stats("torch.probe_raw", raw_data.probeGuess)

    # Limit to n_groups
    diffraction = diffraction[:config.n_groups]

    # Add channel dimension if needed: (n, H, W) -> (n, 1, H, W)
    if diffraction.ndim == 3:
        diffraction = diffraction.unsqueeze(1)

    # Match expected channel count for grouped inputs (gridsize>1)
    expected_channels = None
    if hasattr(model, 'data_config') and hasattr(model.data_config, 'C'):
        expected_channels = int(model.data_config.C)
    elif hasattr(model, 'model_config') and hasattr(model.model_config, 'C_model'):
        expected_channels = int(model.model_config.C_model)
    elif hasattr(config, 'model') and hasattr(config.model, 'gridsize'):
        expected_channels = int(config.model.gridsize) ** 2

    if expected_channels and diffraction.shape[1] == 1 and expected_channels > 1:
        diffraction = diffraction.repeat(1, expected_channels, 1, 1)

    # Ensure probe is complex64
    if not torch.is_complex(probe):
        probe = probe.to(torch.complex64)

    # Add batch dimension to probe if needed
    if probe.ndim == 2:
        probe = probe.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, H, W)

    # Prepare positions (API requires it), real offsets computed for reassembly below
    batch_size = diffraction.shape[0]
    N = diffraction.shape[-1]
    positions = torch.zeros((batch_size, 1, 1, 2), device=device)

    # Prepare scaling factors (match training normalization)
    from ptycho_torch import helper as hh
    from ptycho_torch.config_params import DataConfig as PTDataConfig

    data_cfg_norm = PTDataConfig(N=int(N), grid_size=(1, 1))
    rms_scale = _training_normalization_scale(diffraction)
    rms_scale = rms_scale.to(device=device, dtype=torch.float32)

    if intensity_scale is not None:
        physics_scale = torch.full((batch_size, 1, 1, 1), float(intensity_scale), device=device, dtype=torch.float32)
    else:
        physics_scale = hh.get_physics_scaling_factor(diffraction.squeeze(1), data_cfg_norm)
        if not isinstance(physics_scale, torch.Tensor):
            physics_scale = torch.from_numpy(physics_scale)
        physics_scale = physics_scale.to(device=device, dtype=torch.float32)
        if physics_scale.ndim == 1:
            physics_scale = physics_scale.view(-1, 1, 1, 1)

    physics_weight = 1.0 if getattr(model, 'torch_loss_mode', 'poisson') == 'poisson' else 0.0
    input_scale_factor = rms_scale
    output_scale_factor = (1.0 - physics_weight) * rms_scale + physics_weight * physics_scale  # noqa: F841

    if not quiet:
        print(f"Running inference on {batch_size} images...")

    # Forward pass through model to get per-patch complex predictions
    with torch.no_grad():
        patch_complex = model.forward_predict(
            diffraction,
            positions,
            probe,
            input_scale_factor
        )

    # Compute pixel offsets relative to center-of-mass (B, 1, 1, 2)
    x = torch.from_numpy(raw_data.xcoords[:batch_size]).to(device=device, dtype=torch.float32)
    y = torch.from_numpy(raw_data.ycoords[:batch_size]).to(device=device, dtype=torch.float32)
    dx = x - torch.mean(x)
    dy = y - torch.mean(y)
    offsets = torch.stack([dx, dy], dim=-1).view(batch_size, 1, 1, 2)
    if offsets.shape[1] == 1 and patch_complex.ndim == 4 and patch_complex.shape[1] > 1:
        offsets = offsets.repeat(1, patch_complex.shape[1], 1, 1)
    debug_parity.log_offsets_stats("torch.offsets_global", offsets)

    if os.getenv("PTYCHO_TORCH_STITCH_DEBUG") == "1":
        from ptycho_torch.debug import summarize_offsets
        print(summarize_offsets("offsets_before_reassembly", offsets))

    # Position-aware reassembly using torch helper to produce stitched canvas
    from ptycho_torch.config_params import DataConfig, ModelConfig
    from ptycho_torch import helper as hh

    # Minimal configs required for padding and translation
    N = patch_complex.shape[-1]
    data_cfg = DataConfig(N=int(N), grid_size=(1, 1))
    model_cfg = ModelConfig()
    # Collapse batch into channel dimension so reassembly uses all patches
    patch_complex_reassemble = patch_complex.reshape(1, -1, N, N)
    offsets_reassemble = offsets.reshape(1, -1, 1, 2)
    # Ensure channel consistency for reassembly (C_forward must match predicted channels)
    model_cfg.C_forward = int(patch_complex_reassemble.shape[1])

    crop_size = getattr(config, "stitch_crop_size", 20)
    if crop_size > N:
        crop_size = int(N)
    imgs_merged, _, _ = hh.reassemble_patches_position_real(
        patch_complex_reassemble, offsets_reassemble, data_cfg, model_cfg, crop_size=crop_size
    )
    debug_parity.log_array_stats("torch.reassembly_output", imgs_merged)

    canvas = imgs_merged[0]  # (M, M)
    result_amp, result_phase = present_reconstruction_canvas(canvas, OutputSpec())

    if not quiet:
        print(f"Reconstruction shape: {result_amp.shape}")
        print(f"Amplitude range: [{result_amp.min():.4f}, {result_amp.max():.4f}]")
        print(f"Phase range: [{result_phase.min():.4f}, {result_phase.max():.4f}]")

    return result_amp, result_phase


@scoped_legacy_params
def cli_main():
    """
    CLI entrypoint for PyTorch Lightning checkpoint inference (ADR-003 Phase D.C thin wrapper).

    Thin wrapper that delegates to shared helpers (ptycho_torch.cli.shared) for validation,
    execution config construction, and device resolution. Inference orchestration extracted
    to _run_inference_and_reconstruct() helper for testability.

    Usage:
        python -m ptycho_torch.inference \\
            --model_path <training_output_dir> \\
            --test_data <npz_file> \\
            --output_dir <inference_output_dir> \\
            --n_images 32 \\
            --accelerator cpu \\
            [--quiet]

    Expected Output Artifacts:
        - <output_dir>/reconstructed_amplitude.png
        - <output_dir>/reconstructed_phase.png

    References:
        - Blueprint: plans/active/ADR-003-BACKEND-API/reports/2025-10-20T114500Z/phase_d_cli_wrappers_inference/inference_refactor.md
        - Test contract: tests/torch/test_cli_inference_torch.py
        - Shared helpers: ptycho_torch/cli/shared.py
    """
    raw_argv = tuple(sys.argv[1:])
    parser = argparse.ArgumentParser(
        description="PyTorch Lightning checkpoint inference for ptychography reconstruction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run inference on trained model
  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data datasets/Run1084_recon3_postPC_shrunk_3.npz \\
      --output_dir inference_outputs \\
      --n_images 32 \\
      --device cpu

  # Run with quiet output
  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data test.npz \\
      --output_dir outputs \\
      --n_images 64 \\
      --device cuda \\
      --quiet
        """
    )

    parser.add_argument(
        '--model_path',
        type=str,
        required=True,
        help='Path to training output directory containing Lightning checkpoint (expects checkpoints/last.ckpt or wts.pt)'
    )
    parser.add_argument(
        '--test_data',
        type=str,
        required=True,
        help='Path to test data NPZ file (must conform to specs/data_contracts.md)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help='Directory to save reconstruction outputs (amplitude/phase PNGs)'
    )
    parser.add_argument(
        '--n_images',
        type=int,
        default=None,
        help=(
            'Number of images to use for reconstruction on the uniform stitching '
            'path (default: 32). Incompatible with --patch-weighting probe / '
            '--varpro-scaling, which always reconstruct from the full scan.'
        )
    )
    parser.add_argument(
        '--patch-weighting',
        choices=['uniform', 'probe'],
        default='uniform',
        dest='patch_weighting',
        help=(
            "Stitching weight for patch reassembly: 'uniform' keeps the legacy "
            "CLI path unchanged (default); 'probe' applies |P|^2-weighted "
            "barycentric assembly via reconstruct_image_barycentric."
        )
    )
    parser.add_argument(
        '--varpro-scaling',
        action='store_true',
        dest='varpro_scaling',
        help=(
            'Apply the VarPro (s1, s2) least-squares intensity refit during '
            'reconstruction (routes stitching through '
            'reconstruct_image_barycentric).'
        )
    )
    parser.add_argument(
        '--groups-per-center',
        type=int,
        default=1,
        dest='groups_per_center',
        help=(
            'Fresh coordinate groups drawn per eligible center on the mmap '
            'barycentric path (default: 1). This is runtime-only and does not '
            'overwrite the checkpoint DataConfig.'
        ),
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['cpu', 'cuda'],
        default='cpu',
        help='Device to run inference on (cpu or cuda, default: cpu)'
    )
    parser.add_argument(
        '--quiet',
        action='store_true',
        help='Suppress progress output'
    )
    parser.add_argument(
        '--log-patch-stats',
        action='store_true',
        help='Log per-patch statistics during inference (default: disabled)'
    )
    parser.add_argument(
        '--patch-stats-limit',
        type=int,
        default=None,
        help='Maximum number of batches to log for patch stats (default: no limit)'
    )
    parser.add_argument(
        '--probe-mask',
        dest='probe_mask',
        action='store_true',
        default=False,
        help='Enable Torch probe masking during inference normalization/forward pass (default: disabled).'
    )
    parser.add_argument(
        '--no-probe-mask',
        dest='probe_mask',
        action='store_false',
        help='Disable Torch probe masking during inference.'
    )
    parser.add_argument(
        '--probe-mask-sigma',
        type=float,
        default=1.0,
        dest='probe_mask_sigma',
        help='Gaussian sigma (pixels) for probe-mask edge smoothing (default: 1.0 smooth edge).'
    )
    parser.add_argument(
        '--probe-mask-diameter',
        type=float,
        default=None,
        dest='probe_mask_diameter',
        help='Probe-mask disk diameter in pixels (default: N/2).'
    )
    parser.add_argument(
        '--scale-contract-version',
        choices=['ci_intensity_v2', 'legacy_v1'],
        default=None,
        help='Scaling profile override; must be paired with --measurement-domain.',
    )
    parser.add_argument(
        '--measurement-domain',
        choices=['count_intensity', 'normalized_amplitude'],
        default=None,
        help='Measurement-domain override; must be paired with --scale-contract-version.',
    )

    # Execution config flags (Phase C4.C5 - ADR-003)
    parser.add_argument(
        '--accelerator',
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'gpu', 'cuda', 'tpu', 'mps'],
        help=(
            'Hardware accelerator: auto (auto-detect, default), cpu (CPU-only), '
            'gpu (NVIDIA GPU), cuda (alias for gpu), tpu (Google TPU), mps (Apple Silicon).'
        )
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=0,
        dest='num_workers',
        help=(
            'Number of DataLoader worker processes (default: 0 = synchronous). '
            'Typical values: 2-8 for multi-core systems.'
        )
    )
    parser.add_argument(
        '--inference-batch-size',
        type=int,
        default=None,
        dest='inference_batch_size',
        help=(
            'Batch size for inference DataLoader (default: None = use training batch_size). '
            'Larger values increase throughput. Typical: 16-64 for GPU, 4-8 for CPU.'
        )
    )

    args = parser.parse_args()

    # --- Conformance D4: resolve the stitching/scaling route up front ---
    # 'uniform' keeps the legacy reassemble_patches_position_real path
    # bit-identical; any non-default knob routes through
    # reconstruct_image_barycentric. Requested-but-unsatisfiable combinations
    # fail fast here instead of being silently discarded.
    reassembly_route = _resolve_reassembly_route(
        args.patch_weighting, args.varpro_scaling
    )
    if reassembly_route == 'barycentric':
        requested_knobs = _describe_requested_knobs(
            args.patch_weighting, args.varpro_scaling
        )
        if args.n_images is not None:
            raise ValueError(
                f"--n_images cannot be honored together with {requested_knobs}: "
                "the barycentric reconstruction path always uses the full scan. "
                "Drop --n_images or the stitching/scaling flag(s)."
            )
        if args.probe_mask:
            raise ValueError(
                f"--probe-mask cannot be honored together with {requested_knobs}: "
                "the barycentric path uses the checkpoint's own probe "
                "configuration. Drop one of the flags."
            )
    n_images = args.n_images if args.n_images is not None else 32

    # --- Phase D.C C3: Validate paths using shared helper ---
    from ptycho_torch.cli.shared import validate_paths
    try:
        validate_paths(
            train_file=None,  # Inference mode: no training file
            test_file=Path(args.test_data),
            output_dir=Path(args.output_dir),
        )
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    # Preserve raw-option suppliedness until the factory resolves runtime.
    from ptycho_torch.cli.shared import build_execution_request_from_args
    try:
        execution_request = build_execution_request_from_args(
            args,
            mode='inference',
            explicit_options=raw_argv,
            lane='native-inference',
        )
    except ValueError as e:
        print(f"ERROR: Invalid execution config: {e}")
        sys.exit(1)

    # Fail-fast: Check Lightning availability
    try:
        import lightning  # noqa: F401
        import torch
    except ImportError as e:
        raise RuntimeError(
            "PyTorch Lightning backend requires 'lightning' and 'torch' packages. "
            "Install via: pip install -e .[torch]\n"
            f"Import error: {e}"
        )

    # Phase C4.C6+C4.C7: Delegate to factory for CONFIG-001 compliance (ADR-003)
    # Replaces manual checkpoint loading and config construction with centralized
    # factory pattern. The factory handles:
    # 1. Path validation and checkpoint discovery
    # 2. CONFIG-001 bridging (update_legacy_dict before any IO)
    # 3. Config translation (PyTorch → TensorFlow canonical dataclasses)
    # 4. Execution config merging with override precedence

    from ptycho_torch.config_factory import create_inference_payload
    from ptycho.raw_data import RawData

    # Convert paths to Path objects
    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data)
    output_dir = Path(args.output_dir)

    # Build overrides dict for factory
    overrides = {
        'n_groups': n_images,  # Map CLI arg to config field
        'probe_mask': args.probe_mask,
        'probe_mask_sigma': args.probe_mask_sigma,
        'probe_mask_diameter': args.probe_mask_diameter,
        'log_patch_stats': args.log_patch_stats,
        'patch_stats_limit': args.patch_stats_limit,
        # Conformance D4: thread the stitching/scaling knobs so the resolved
        # pt_inference_config matches the routing decision above.
        'patch_weighting': args.patch_weighting,
        'varpro_scaling': args.varpro_scaling,
    }
    if args.scale_contract_version is not None:
        overrides['scale_contract_version'] = args.scale_contract_version
    if args.measurement_domain is not None:
        overrides['measurement_domain'] = args.measurement_domain

    # Call factory to construct all configs and populate params.cfg
    try:
        payload = create_inference_payload(
            model_path=model_path,
            test_data_file=test_data_path,
            output_dir=output_dir,
            overrides=overrides,
            execution_config=execution_request,
        )

        # Extract configs from payload (factory already populated params.cfg)
        tf_inference_config = payload.tf_inference_config
        execution_config = payload.execution_config

        if not args.quiet:
            print("Loaded configuration from model checkpoint")
            print(f"Test data: {test_data_path}")
            print(f"Output directory: {output_dir}")
            print(f"N groups: {tf_inference_config.n_groups}")
            print(f"Execution config: accelerator={execution_config.accelerator}, "
                  f"num_workers={execution_config.num_workers}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to create inference payload.\n"
            f"Error: {e}\n"
            "Ensure model_path contains wts.h5.zip and test_data conforms to DATA-001."
        )

    if reassembly_route == 'barycentric':
        try:
            import torch

            device_map = {
                'cpu': 'cpu',
                'gpu': 'cuda',
                'cuda': 'cuda',
                'mps': 'mps',
                'auto': 'cuda' if torch.cuda.is_available() else 'cpu',
            }
            device = device_map.get(execution_config.accelerator, 'cpu')
            precision = getattr(execution_config, 'precision', None)
            if precision not in {'32-true', '16-mixed', 'bf16-mixed'}:
                precision = '32-true'
            result = reconstruct_npz_barycentric(
                model_path,
                test_data_path,
                run_root=output_dir,
                groups_per_center=args.groups_per_center,
                scale_contract_version=args.scale_contract_version,
                measurement_domain=args.measurement_domain,
                inference_config=payload.pt_inference_config,
                device=device,
                num_workers=int(execution_config.num_workers or 0),
                inference_batch_size=execution_config.inference_batch_size,
                precision=precision,
                quiet=args.quiet,
            )
            amplitude, phase = result.amplitude, result.phase
            save_individual_reconstructions(amplitude, phase, output_dir)
            if payload.pt_inference_config.log_patch_stats:
                from ptycho_torch.patch_stats_instrumentation import PatchStatsLogger

                amp_tensor = torch.as_tensor(amplitude).unsqueeze(0).unsqueeze(0)
                logger = PatchStatsLogger(
                    output_dir=output_dir / "analysis",
                    enabled=True,
                    limit=payload.pt_inference_config.patch_stats_limit,
                )
                logger.log_batch(amp_tensor, phase="inference", batch_idx=0)
                logger.finalize()
            if not args.quiet:
                print("\nInference completed successfully!")
                print(f"Output artifacts saved to: {output_dir}")
            return 0
        except Exception as e:
            print(f"ERROR: Inference failed with exception: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            return 1

    # Load checkpoint via spec-compliant bundle loader (Phase C4.C6/C4.C7 - ADR-003)
    # Replaces manual checkpoint search with factory-validated wts.h5.zip loading
    try:
        import torch
        from ptycho_torch.workflows.components import load_inference_bundle_torch

        # load_inference_bundle_torch expects bundle_dir containing wts.h5.zip
        # It handles CONFIG-001 (restores params.cfg from archive) and returns
        # (models_dict, params_dict) matching TensorFlow baseline API
        models_dict, params_dict = load_inference_bundle_torch(
            bundle_dir=model_path,
            model_name='diffraction_to_obj',
            scale_contract_version=args.scale_contract_version,
            measurement_domain=args.measurement_domain,
        )

        # Extract Lightning module from models dict
        model = models_dict['diffraction_to_obj']
        model.eval()

        # Resolve device from execution config
        device_map = {
            'cpu': 'cpu',
            'gpu': 'cuda',
            'cuda': 'cuda',
            'mps': 'mps',
            'auto': 'cuda' if torch.cuda.is_available() else 'cpu',
        }
        device = device_map.get(execution_config.accelerator, 'cpu')
        model.to(device)

        if not args.quiet:
            print(f"Loaded model bundle from: {model_path / 'wts.h5.zip'}")
            print(f"Model device: {device}")
            print(f"Restored params.cfg from bundle (N={params_dict.get('N', 'N/A')}, "
                  f"gridsize={params_dict.get('gridsize', 'N/A')})")

    except Exception as e:
        raise RuntimeError(
            f"Failed to load inference bundle from {model_path}.\n"
            f"Error: {e}\n"
            "Ensure model_path contains wts.h5.zip archive (spec-compliant format)."
        )

    _require_ci_varpro_scaling(model, payload.pt_inference_config)

    # Load test data via RawData (factory already validated path)
    # NOTE: params.cfg already populated by factory, so RawData.from_file is safe to call
    try:
        raw_data = RawData.from_file(str(test_data_path))

        if not args.quiet:
            print(f"Loaded test data: {raw_data.diff3d.shape[0]} scan positions")

    except Exception as e:
        raise RuntimeError(
            f"Failed to load test data from {test_data_path}.\n"
            f"Error: {e}\n"
            "Ensure NPZ conforms to specs/data_contracts.md"
        )

    # --- Phase D.C C3: Delegate to inference helper (Conformance D4 routing) ---
    try:
        if reassembly_route == 'barycentric':
            amplitude, phase = _run_barycentric_inference_and_reconstruct(
                model=model,
                test_data_path=test_data_path,
                pt_inference_config=payload.pt_inference_config,
                execution_config=execution_config,
                device=device,
                output_dir=output_dir,
                quiet=args.quiet,
            )
        else:
            amplitude, phase = _run_inference_and_reconstruct(
                model=model,
                raw_data=raw_data,
                config=tf_inference_config,
                execution_config=execution_config,
                device=device,
                quiet=args.quiet,
                intensity_scale=params_dict.get('intensity_scale'),
            )

        # Save individual reconstructions (required by test contract)
        save_individual_reconstructions(amplitude, phase, output_dir)

        if payload.pt_inference_config.log_patch_stats:
            from ptycho_torch.patch_stats_instrumentation import PatchStatsLogger
            import torch

            amp_tensor = torch.as_tensor(amplitude).unsqueeze(0).unsqueeze(0)
            logger = PatchStatsLogger(
                output_dir=output_dir / "analysis",
                enabled=True,
                limit=payload.pt_inference_config.patch_stats_limit,
            )
            logger.log_batch(amp_tensor, phase="inference", batch_idx=0)
            logger.finalize()

        if not args.quiet:
            print("\nInference completed successfully!")
            print(f"Output artifacts saved to: {output_dir}")

        return 0

    except Exception as e:
        print(f"ERROR: Inference failed with exception: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(cli_main())
