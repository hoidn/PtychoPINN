"""
PyTorch Inference Module for Ptychography Reconstruction

The canonical CLI loads versioned model bundles, performs reconstruction, and
generates amplitude/phase PNGs.

Usage Examples:

  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data datasets/Run1084_recon3_postPC_shrunk_3.npz \\
      --output_dir inference_outputs \\
      --device cpu

References:
  - Entry-point contract: docs/specs/spec-ptycho-interfaces.md (python -m ptycho_torch.inference)
  - Test contract: tests/torch/test_integration_workflow_torch.py
  - Red phase evidence: docs/findings.md (see git history for the originating plan) §2.3
"""

#Generic
import os
import argparse
import copy
import gc
import hashlib
import math
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, TYPE_CHECKING

import numpy as np
from ptycho.config.legacy_state import scoped_legacy_params
from ptycho.reconstruction_policy import resolve_cli_reconstruction_policy
from ptycho_torch.reconstruction_ports import present_reconstruction_canvas
from ptycho_torch.inference_validation import (
    _require_record_fields_agree,
    _validate_authentic_channels,
    _validate_flat_npz,
    validate_bundle_matches_workflow,
)

#ML libraries
import matplotlib.pyplot as plt

if TYPE_CHECKING:
    import torch
    from ptycho_torch.config_params import DataConfig
    from ptycho_torch.reassembly_diagnostics import ReassemblyDiagnostics


def save_individual_reconstructions(obj_amp, obj_phase, output_dir):
    """
    Save individual amplitude and phase reconstructions as separate PNG files.

    This function generates the specific output artifacts expected by the PyTorch
    integration test workflow.

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
    im_amp = ax_amp.imshow(obj_amp, cmap='gray')
    plt.colorbar(im_amp, ax=ax_amp)
    ax_amp.set_title('Reconstructed Amplitude')
    ax_amp.axis('off')

    amp_path = output_dir / "reconstructed_amplitude.png"
    plt.savefig(amp_path, dpi=150, bbox_inches='tight')
    plt.close(fig_amp)
    print(f"Saved amplitude reconstruction to: {amp_path}")

    # Create phase figure
    fig_phase, ax_phase = plt.subplots(figsize=(6, 6))
    im_phase = ax_phase.imshow(obj_phase, cmap='gray')
    plt.colorbar(im_phase, ax=ax_phase)
    ax_phase.set_title('Reconstructed Phase')
    ax_phase.axis('off')

    phase_path = output_dir / "reconstructed_phase.png"
    plt.savefig(phase_path, dpi=150, bbox_inches='tight')
    plt.close(fig_phase)
    print(f"Saved phase reconstruction to: {phase_path}")


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
    measurement_gauge_canvas: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    prescale_canvas: np.ndarray
    effective_data_config: "DataConfig"
    canvas_weights: np.ndarray
    canvas_anchor: Mapping[str, Any]
    channel_indices: np.ndarray
    channel_coordinates: np.ndarray
    source_metadata: Mapping[str, np.ndarray]
    reassembly: "ReassemblyDiagnostics"


@dataclass(frozen=True)
class TiledReconstructionResult:
    """Self-contained output of strict-load mmap raster tiling."""

    complex_canvas: np.ndarray
    measurement_gauge_canvas: np.ndarray
    amplitude: np.ndarray
    phase: np.ndarray
    prescale_canvas: np.ndarray
    effective_data_config: "DataConfig"
    canvas_weights: np.ndarray
    canvas_anchor: Mapping[str, Any]
    channel_indices: np.ndarray
    channel_coordinates: np.ndarray
    reassembly: Mapping[str, Any]


def _canonicalize_tiled_patch_order(
    patches: Any,
    scan_ids: Any,
    coordinates: Any,
    *,
    expected_x: Any,
    expected_y: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Order patches by authenticated scan id and verify the declared lattice."""

    patch_array = np.asarray(patches)
    ids = np.asarray(scan_ids)
    coords = np.asarray(coordinates, dtype=np.float64)
    expected_x_array = np.asarray(expected_x, dtype=np.float64)
    expected_y_array = np.asarray(expected_y, dtype=np.float64)
    expected_count = int(expected_x_array.size)
    if (
        expected_x_array.ndim != 1
        or expected_y_array.shape != expected_x_array.shape
        or patch_array.ndim != 3
        or ids.shape != (patch_array.shape[0],)
        or coords.shape != (patch_array.shape[0], 2)
        or patch_array.shape[0] != expected_count
        or not np.issubdtype(ids.dtype, np.integer)
    ):
        raise ValueError(
            "tiled scan ids must form a complete source-row bijection"
        )
    normalized_ids = ids.astype(np.int64, copy=False)
    order = np.argsort(normalized_ids, kind="stable")
    ordered_ids = normalized_ids[order]
    if not np.array_equal(ordered_ids, np.arange(expected_count, dtype=np.int64)):
        raise ValueError(
            "tiled scan ids must form a complete source-row bijection"
        )
    ordered_coords = coords[order]
    if not np.allclose(
        ordered_coords[:, 0], expected_x_array, rtol=0.0, atol=1e-6
    ) or not np.allclose(
        ordered_coords[:, 1], expected_y_array, rtol=0.0, atol=1e-6
    ):
        raise ValueError(
            "tiled scan coordinates do not match the declared fixed-pitch raster"
        )
    ordered = np.ascontiguousarray(patch_array[order])
    ordered_ids = np.ascontiguousarray(ordered_ids)
    ordered.setflags(write=False)
    ordered_ids.setflags(write=False)
    return ordered, ordered_ids


def _validate_loaded_reconstruction_identity(model: Any) -> None:
    """Validate the strict loader's decode/construction boundary before mmap work.

    Channel count is derived from ``data_config.gridsize`` at consumption, and
    scale-contract coherence is enforced by ``decode_artifact_identity`` and
    ``build_ptychopinn_application`` on the load path; this function only
    re-checks the dual-written structural surface that construction derives but
    does not independently verify against the persisted ``ModelSpec``.
    """
    from ptycho_torch.config_params import (
        DataConfig,
        InferenceConfig,
        ModelConfig,
        TrainingConfig,
    )
    from ptycho_torch.model_spec import ModelSpec

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


@dataclass(frozen=True)
class ReconstructionRuntimeParams:
    """Runtime loader parameters for the dataset-in reconstruction kernel.

    All three runtime configs are explicit arguments: the kernel never reads
    them back off the dataset or the global ``params.cfg``. ``data_config`` is
    ``model.data_config`` (the persisted data config, handed straight to
    ``PtychoDataset``); ``training_config`` and ``inference_config`` are the
    runtime (loader) configs. ``source_metadata`` is the NPZ-side
    coordinate/scale metadata the wrapper read during load, threaded through
    to the frozen result.
    """

    data_config: Any
    training_config: Any
    inference_config: Any
    source_metadata: Mapping[str, np.ndarray]
    precision: str = "32-true"
    quiet: bool = False
    enforce_ci_varpro: bool = True
    compute_count_metrics: bool = True



def _stage_and_construct_reconstruction_dataset(
    model: Any,
    test_data_path: Path,
    *,
    workspace: Path,
    groups_per_center: int,
    device: str,
    num_workers: int,
    dataset_manifest_path: Optional[Path],
    expected_workflow: Any,
    rescale_to_nphotons: float | None = None,
) -> tuple[Any, Mapping[str, np.ndarray], Any, Any]:
    """Validate and symlink-stage one flat NPZ into a mmap PtychoDataset.

    The held-out NPZ is linked (not copied) into a staging directory inside
    the caller-provided mmap workspace so ``PtychoDataset``'s directory glob
    finds exactly one scan. Returns
    ``(dataset, source_metadata, runtime_data_config, runtime_training_config)``
    — the runtime configs with the ``groups_per_center`` runtime argument threaded
    to the dataset constructor (no dataclass field round-trip).
    """
    from ptycho_torch.dataloader import PtychoDataset
    runtime_data_config = model.data_config
    runtime_training_config = replace(
        model.training_config,
        device=str(device),
        num_workers=num_workers,
    )
    _validate_flat_npz(
        test_data_path,
        model.data_config,
        dataset_manifest_path=dataset_manifest_path,
        expected_workflow=expected_workflow,
    )
    with np.load(test_data_path, allow_pickle=False) as archive:
        source_metadata = {
            name: np.array(archive[name], copy=True)
            for name in ("xcoords", "ycoords", "object_amplitude_scale")
            if name in archive.files
        }
    staged_dir = workspace / "staged"
    staged_dir.mkdir()
    staged_dir.joinpath(test_data_path.name).symlink_to(test_data_path.resolve())
    ci_target = (
        runtime_data_config.scale_contract_version == "ci_intensity_v2"
        and runtime_data_config.measurement_domain == "count_intensity"
    )
    dataset = PtychoDataset(
        str(staged_dir),
        model.model_config,
        runtime_data_config,
        training_config=runtime_training_config,
        data_dir=str(workspace / "mmap" / "memmap"),
        remake_map=True,
        defer_ci_statistics=ci_target,
        rescale_to_nphotons=rescale_to_nphotons,
        groups_per_center=groups_per_center,
    )
    if ci_target:
        statistics = model.get_ci_statistics()
        if statistics is None:
            raise ValueError("strict CI bundle is missing frozen training statistics")
        dataset.data_dict["ci_statistics"] = statistics
    return dataset, source_metadata, runtime_data_config, runtime_training_config


def reconstruct_from_dataset(
    model: Any,
    dataset: Any,
    *,
    runtime_params: ReconstructionRuntimeParams,
) -> BarycentricReconstructionResult:
    """Reconstruct one already-grouped mmap dataset in place (dataset-in kernel).

    The one documented programmatic reconstruction kernel: a loaded model and
    an already-constructed ``PtychoDataset`` in, frozen amplitude/phase
    snapshots out. No NPZ staging, no ``params.cfg`` access, no global reads;
    the mmap workspace is the caller's (it was supplied when the dataset was
    built). Hop count: CLI -> stage -> kernel.
    """
    from ptycho_torch.reassembly import reconstruct_image_barycentric
    from ptycho_torch.reassembly_diagnostics import ReassemblyDiagnostics

    runtime_data_config = runtime_params.data_config
    training_config = runtime_params.training_config
    inference_config = runtime_params.inference_config
    precision = runtime_params.precision
    quiet = runtime_params.quiet

    if runtime_params.enforce_ci_varpro:
        _require_ci_varpro_scaling(model, inference_config)
    requested = _describe_requested_knobs(
        inference_config.patch_weighting,
        inference_config.varpro_scaling,
    )
    if not quiet:
        print(f"Reassembly route: barycentric ({requested})")

    (
        expected_scan_ids,
        expected_patch_count,
        channel_indices,
        channel_coordinates,
    ) = _validate_authentic_channels(dataset, runtime_data_config)
    reconstruction_dataset = None
    try:
        canvas, reconstruction_dataset, diagnostics, prescale_canvas = (
            reconstruct_image_barycentric(
                model,
                dataset,
                training_config,
                runtime_data_config,
                model.model_config,
                inference_config,
                gpu_ids=None,
                verbose=not quiet,
                structured_diagnostics=True,
                precision=precision,
                compute_count_metrics=runtime_params.compute_count_metrics,
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
            inference_config.patch_weighting,
            inference_config.varpro_scaling,
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
            measurement_gauge_canvas=canvas_snapshot,
            amplitude=amplitude_snapshot,
            phase=phase_snapshot,
            prescale_canvas=prescale_snapshot,
            effective_data_config=runtime_data_config,
            canvas_weights=weight_snapshot,
            canvas_anchor=anchor_snapshot,
            channel_indices=channel_indices,
            channel_coordinates=channel_coordinates,
            source_metadata=dict(runtime_params.source_metadata),
            reassembly=diagnostics,
        )
    finally:
        # TensorDict mmap handles must be unreachable before the caller's
        # workspace teardown, including error paths.
        del reconstruction_dataset
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


def reconstruct(
    model: os.PathLike[str] | str,
    dataset: os.PathLike[str] | str,
    *,
    work_dir: os.PathLike[str] | str | None = None,
    groups_per_center: int = 1,
    expected_workflow: Any = None,
    dataset_manifest_path: Optional[os.PathLike[str] | str] = None,
    scale_contract_version: Optional[str] = None,
    measurement_domain: Optional[str] = None,
    inference_config: Any = None,
    nphotons: float | None = None,
    device: str = "cpu",
    num_workers: int = 0,
    inference_batch_size: Optional[int] = None,
    precision: str = "32-true",
    quiet: bool = False,
) -> BarycentricReconstructionResult:
    """Strictly reload one bundle and reconstruct one flat NPZ through mmap."""
    from ptycho.config.legacy_state import isolated_archived_params_scope
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

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

    bundle_input = Path(model)
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
    test_data_path = Path(dataset)
    if not test_data_path.is_file() or test_data_path.suffix.lower() != ".npz":
        raise FileNotFoundError(f"held-out NPZ does not exist: {test_data_path}")
    workspace_parent = None if work_dir is None else Path(work_dir)
    if workspace_parent is not None:
        workspace_parent.mkdir(parents=True, exist_ok=True)
        if not workspace_parent.is_dir():
            raise NotADirectoryError(f"work_dir is not a directory: {workspace_parent}")

    with isolated_archived_params_scope():
        models, loader_params = load_inference_bundle_torch(
            bundle_dir,
            model_name="diffraction_to_obj",
            scale_contract_version=scale_contract_version,
            measurement_domain=measurement_domain,
        )
    if "diffraction_to_obj" not in models:
        raise ValueError("strict bundle did not contain diffraction_to_obj")
    loaded_model = models["diffraction_to_obj"]
    _validate_loaded_reconstruction_identity(loaded_model)
    if inference_config is not None and not isinstance(
        inference_config, InferenceConfig
    ):
        raise TypeError("inference_config must be a Torch InferenceConfig")
    runtime_inference_config = loaded_model.inference_config
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
    validate_bundle_matches_workflow(
        loaded_model,
        loader_params,
        expected_workflow,
        reconstruction_method="barycentric",
        runtime_inference_config=runtime_inference_config,
        groups_per_center=groups_per_center,
    )
    from ptycho.acquisition import decode_acquisition
    from ptycho_torch.scaling_contract import (
        CI_SCALE_CONTRACT,
        COUNT_INTENSITY,
        LEGACY_SCALE_CONTRACT,
        NORMALIZED_AMPLITUDE,
    )

    target_nphotons = float(loaded_model.data_config.nphotons)
    if nphotons is not None:
        if isinstance(nphotons, (bool, np.bool_)):
            raise TypeError("nphotons must be a positive real scalar")
        try:
            supplied_nphotons = float(nphotons)
        except (TypeError, ValueError, OverflowError) as error:
            raise TypeError("nphotons must be a positive real scalar") from error
        if not math.isfinite(supplied_nphotons) or supplied_nphotons <= 0:
            raise ValueError("nphotons must be positive and finite")
        if supplied_nphotons != target_nphotons:
            raise ValueError(
                "reconstruction nphotons must equal the bundle target: "
                f"{supplied_nphotons!r} != {target_nphotons!r}"
            )

    source = decode_acquisition(test_data_path, coordinate_policy="trailing")
    source_pair = (source.scale_contract_version, source.measurement_domain)
    target_pair = (
        loaded_model.data_config.scale_contract_version,
        loaded_model.data_config.measurement_domain,
    )
    rescale_to_nphotons = None
    if target_pair == (CI_SCALE_CONTRACT, COUNT_INTENSITY):
        if source_pair == (LEGACY_SCALE_CONTRACT, NORMALIZED_AMPLITUDE):
            rescale_to_nphotons = target_nphotons
        elif source_pair == (None, None):
            saved_digest = loader_params.get("rescaled_source_sha256")
            digest_matches = False
            if saved_digest is not None:
                with test_data_path.open("rb") as source_file:
                    digest_matches = (
                        hashlib.file_digest(source_file, "sha256").hexdigest()
                        == saved_digest
                    )
            if digest_matches or nphotons is not None:
                rescale_to_nphotons = target_nphotons
    elif target_pair == (LEGACY_SCALE_CONTRACT, NORMALIZED_AMPLITUDE):
        if source_pair == (CI_SCALE_CONTRACT, COUNT_INTENSITY):
            raise ValueError("count-intensity source is incompatible with a legacy bundle")
    else:
        raise ValueError(f"unsupported loaded measurement target {target_pair!r}")
    manifest_path = (
        Path(dataset_manifest_path)
        if dataset_manifest_path is not None
        else None
    )
    with tempfile.TemporaryDirectory(
        prefix="barycentric-workspace-",
        dir=workspace_parent,
    ) as workspace_name:
        workspace = Path(workspace_name)
        (
            dataset,
            source_metadata,
            runtime_data_config,
            runtime_training_config,
        ) = _stage_and_construct_reconstruction_dataset(
            loaded_model,
            test_data_path,
            workspace=workspace,
            groups_per_center=groups_per_center,
            device=device,
            num_workers=num_workers,
            dataset_manifest_path=manifest_path,
            expected_workflow=expected_workflow,
            rescale_to_nphotons=rescale_to_nphotons,
        )
        result = reconstruct_from_dataset(
            loaded_model,
            dataset,
            runtime_params=ReconstructionRuntimeParams(
                data_config=runtime_data_config,
                training_config=runtime_training_config,
                inference_config=runtime_inference_config,
                source_metadata=source_metadata,
                precision=precision,
                quiet=quiet,
            ),
        )
        del dataset
        gc.collect()
    return result


def reconstruct_from_arrays(
    model: Any,
    arrays: Mapping[str, np.ndarray],
    *,
    runtime_params: ReconstructionRuntimeParams,
    workspace: os.PathLike[str] | str,
    groups_per_center: int = 1,
    device: str = "cpu",
    num_workers: int = 0,
) -> BarycentricReconstructionResult:
    """Reconstruct one flat acquisition staged from in-memory arrays (arrays-in kernel).

    The embedder-facing reconstruction seam (Ptychodus): a loaded model and a
    mapping of in-memory numpy arrays — the flat-acquisition NPZ contents
    (``diff3d``/``diffraction``, ``xcoords``, ``ycoords``, ``probeGuess``, and
    the optional keys) — in, frozen amplitude/phase snapshots out. No source
    NPZ path, no ``params.cfg`` access. Hop count: embedder -> stage (arrays ->
    mmap dataset) -> kernel (``reconstruct_from_dataset``).

    ``runtime_params`` supplies ``inference_config`` and the ``precision``,
    ``quiet``, ``enforce_ci_varpro``, and ``compute_count_metrics`` knobs. Its
    ``data_config``, ``training_config``, and ``source_metadata`` fields are
    derived during staging from the model, the arrays, and the
    ``device``/``num_workers`` staging arguments, and replace any caller values.
    No dataset-manifest or workflow-conformance validation: those identity
    checks belong to the NPZ-path entry (``reconstruct``); the
    arrays-in seam validates the array dtype/shape/coordinate contract only.
    """
    if not isinstance(arrays, Mapping):
        raise TypeError("arrays must be a mapping of numpy array names to arrays")
    if isinstance(groups_per_center, bool) or not isinstance(groups_per_center, int):
        raise TypeError("groups_per_center must be a positive integer")
    if groups_per_center <= 0:
        raise ValueError("groups_per_center must be a positive integer")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise TypeError("num_workers must be a nonnegative integer")
    if num_workers < 0:
        raise ValueError("num_workers must be a nonnegative integer")

    workspace_path = Path(workspace)
    workspace_path.mkdir(parents=True, exist_ok=True)
    arrays_path = workspace_path / "held_out.npz"
    np.savez(
        arrays_path,
        **{name: np.asarray(value) for name, value in arrays.items()},
    )
    (
        dataset,
        source_metadata,
        runtime_data_config,
        runtime_training_config,
    ) = _stage_and_construct_reconstruction_dataset(
        model,
        arrays_path,
        workspace=workspace_path,
        groups_per_center=groups_per_center,
        device=device,
        num_workers=num_workers,
        dataset_manifest_path=None,
        expected_workflow=None,
    )
    kernel_params = replace(
        runtime_params,
        data_config=runtime_data_config,
        training_config=runtime_training_config,
        source_metadata=source_metadata,
    )
    try:
        return reconstruct_from_dataset(
            model,
            dataset,
            runtime_params=kernel_params,
        )
    finally:
        del dataset
        gc.collect()


def reconstruct_npz_tiled(
    bundle_path: os.PathLike[str] | str,
    test_npz_path: os.PathLike[str] | str,
    *,
    run_root: os.PathLike[str] | str,
    groups_per_center: int = 1,
    expected_workflow: Any,
    dataset_manifest_path: Optional[os.PathLike[str] | str] = None,
    scale_contract_version: Optional[str] = None,
    measurement_domain: Optional[str] = None,
    inference_config: Any = None,
    device: str = "cpu",
    num_workers: int = 0,
    inference_batch_size: Optional[int] = None,
    precision: str = "32-true",
    quiet: bool = False,
) -> TiledReconstructionResult:
    """Strictly reload one bundle and assemble a fixed-pitch mmap raster.

    The shared coordinate accumulator is evaluated with a method-derived
    window equal to the fixed raster pitch.  On the validated integer lattice
    every patch lands without interpolation or overlap, so cropping the exact
    unit-weight support is equivalent to the declared raster tiling while
    remaining independent of loader iteration order.
    """

    from ptycho.config.legacy_state import isolated_archived_params_scope
    from ptycho.simulation.flat_acquisition import (
        fixed_pitch_raster_positions,
        ordered_raster_coordinates,
    )
    from ptycho_torch.config_params import InferenceConfig
    from ptycho_torch.reassembly_diagnostics import array_metadata
    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch

    if expected_workflow is None or not hasattr(expected_workflow, "inference"):
        raise TypeError("tiled reconstruction requires expected_workflow identity")
    if expected_workflow.inference.reconstruction_method != "tiled":
        raise ValueError("expected_workflow does not request tiled reconstruction")
    if isinstance(groups_per_center, bool) or not isinstance(groups_per_center, int):
        raise TypeError("groups_per_center must be a positive integer")
    if groups_per_center != 1:
        raise ValueError("tiled reconstruction requires groups_per_center=1")
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
    bundle_dir = bundle_input.parent if bundle_input.name == "wts.h5.zip" else bundle_input
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
    _validate_loaded_reconstruction_identity(model)
    if inference_config is not None and not isinstance(
        inference_config, InferenceConfig
    ):
        raise TypeError("inference_config must be a Torch InferenceConfig")
    runtime_inference_config = model.inference_config
    if inference_config is not None:
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
    validate_bundle_matches_workflow(
        model,
        loader_params,
        expected_workflow,
        reconstruction_method="tiled",
        runtime_inference_config=runtime_inference_config,
        groups_per_center=groups_per_center,
    )

    simulation = expected_workflow.simulation.test
    if simulation.scan.grid_size != (1, 1):
        raise ValueError("tiled reconstruction requires C=1/gridsize=1")
    outer_offset = int(simulation.scan.outer_offset_test)
    if (
        outer_offset <= 0
        or outer_offset % 4
        or outer_offset > 2 * int(simulation.N)
    ):
        raise ValueError(
            "tiled reconstruction requires outer_offset_test divisible by 4 "
            "and no larger than 2*N"
        )
    tile_size = outer_offset // 2
    tiled_inference_config = replace(
        runtime_inference_config,
        middle_trim=tile_size,
        patch_weighting="uniform",
    )
    manifest_path = (
        Path(dataset_manifest_path)
        if dataset_manifest_path is not None
        else None
    )
    with tempfile.TemporaryDirectory(
        prefix="tiled-workspace-",
        dir=output_root,
    ) as workspace_name:
        workspace = Path(workspace_name)
        (
            dataset,
            source_metadata,
            runtime_data_config,
            runtime_training_config,
        ) = _stage_and_construct_reconstruction_dataset(
            model,
            test_data_path,
            workspace=workspace,
            groups_per_center=groups_per_center,
            device=device,
            num_workers=num_workers,
            dataset_manifest_path=manifest_path,
            expected_workflow=expected_workflow,
        )
        base = reconstruct_from_dataset(
            model,
            dataset,
            runtime_params=ReconstructionRuntimeParams(
                data_config=runtime_data_config,
                training_config=runtime_training_config,
                inference_config=tiled_inference_config,
                source_metadata=source_metadata,
                precision=precision,
                quiet=quiet,
                enforce_ci_varpro=True,
                compute_count_metrics=True,
            ),
        )
        del dataset
        gc.collect()
    source_metadata = base.source_metadata
    xcoords = np.asarray(source_metadata["xcoords"], dtype=np.float64)
    ycoords = np.asarray(source_metadata["ycoords"], dtype=np.float64)
    if simulation.object.patch_amplitude_normalization == "mean_patch_max":
        if "object_amplitude_scale" not in source_metadata:
            raise ValueError(
                "tiled reconstruction requires test object_amplitude_scale"
            )
        scale_array = np.asarray(source_metadata["object_amplitude_scale"])
        if scale_array.shape != ():
            raise ValueError("object_amplitude_scale must be scalar")
        object_amplitude_scale = float(scale_array)
    else:
        if "object_amplitude_scale" in source_metadata:
            raise ValueError(
                "object_amplitude_scale requires mean_patch_max identity"
            )
        object_amplitude_scale = 1.0
    if not np.isfinite(object_amplitude_scale) or object_amplitude_scale <= 0.0:
        raise ValueError("object_amplitude_scale must be positive and finite")
    expected_x, expected_y = fixed_pitch_raster_positions(
        n_positions=simulation.object.diffractions_per_object,
        height=simulation.object.image_size[0],
        width=simulation.object.image_size[1],
        patch_size=simulation.N,
        pitch=float(tile_size),
    )
    expected_x, expected_y = ordered_raster_coordinates(
        (expected_x, expected_y),
        frame_order_recipe=getattr(
            expected_workflow.simulation,
            "frame_order_recipe",
            "object-major-v1",
        ),
    )
    if not np.array_equal(xcoords, expected_x) or not np.array_equal(
        ycoords, expected_y
    ):
        raise ValueError("held-out coordinates do not match fixed_pitch_raster")
    flat_ids = np.asarray(base.channel_indices, dtype=np.int64).reshape(-1)
    identity_patches = flat_ids[:, None, None].astype(np.complex64)
    identity_coords = np.asarray(base.channel_coordinates).reshape(-1, 2)
    _canonicalize_tiled_patch_order(
        identity_patches,
        flat_ids,
        identity_coords,
        expected_x=expected_x,
        expected_y=expected_y,
    )

    weights = np.asarray(base.canvas_weights)
    support = weights > 0
    rows = np.flatnonzero(np.any(support, axis=1))
    columns = np.flatnonzero(np.any(support, axis=0))
    if rows.size == 0 or columns.size == 0:
        raise ValueError("tiled reconstruction has empty canvas support")
    row_slice = slice(int(rows[0]), int(rows[-1]) + 1)
    column_slice = slice(int(columns[0]), int(columns[-1]) + 1)
    support_crop = support[row_slice, column_slice]
    weight_crop = np.asarray(weights[row_slice, column_slice])
    side = int(np.sqrt(simulation.object.diffractions_per_object))
    expected_shape = (side * tile_size, side * tile_size)
    if support_crop.shape != expected_shape or not support_crop.all():
        raise ValueError(
            "tiled reconstruction support does not form the expected raster canvas"
        )
    if not np.allclose(weight_crop, 1.0, rtol=0.0, atol=1e-6):
        raise ValueError("tiled reconstruction requires unit nonoverlapping weights")

    normalized_canvas = np.asarray(base.complex_canvas)[row_slice, column_slice]
    normalized_prescale = np.asarray(base.prescale_canvas)[row_slice, column_slice]
    restored_canvas = normalized_canvas * object_amplitude_scale
    restored_prescale = normalized_prescale * object_amplitude_scale
    canvas_snapshot = _snapshot_array(
        restored_canvas,
        name="complex_canvas",
        complex_required=True,
    )
    prescale_snapshot = _snapshot_array(
        restored_prescale,
        name="prescale_canvas",
        complex_required=True,
    )
    weight_snapshot = _snapshot_array(weight_crop, name="canvas weights")
    measurement_gauge_snapshot = _snapshot_array(
        normalized_canvas,
        name="measurement_gauge_canvas",
        complex_required=True,
    )
    amplitude_snapshot = _snapshot_array(np.abs(canvas_snapshot), name="amplitude")
    phase_snapshot = _snapshot_array(np.angle(canvas_snapshot), name="phase")
    scan_com = [float(np.mean(expected_x)), float(np.mean(expected_y))]
    border_size = (int(simulation.N) - tile_size) / 2.0
    truth_origin = [int(np.ceil(border_size)), int(np.ceil(border_size))]
    canvas_anchor = {
        "scan_com": scan_com,
        "canvas_shape": list(expected_shape),
        "canvas_origin_offset": [
            expected_shape[1] // 2 - scan_com[0],
            expected_shape[0] // 2 - scan_com[1],
        ],
        "truth_origin": truth_origin,
        "assembly_method": "tiled_raster_v1",
    }
    reassembly = base.reassembly.to_jsonable()
    reassembly.update(
        {
            "assembly_method": "tiled_raster_v1",
            "canvas_anchor": canvas_anchor,
            "canvas_weights": array_metadata(weight_snapshot),
            "measurement_gauge_canvas": array_metadata(
                measurement_gauge_snapshot
            ),
            "requested_middle_trim": int(runtime_inference_config.middle_trim),
            "effective_tile_size": tile_size,
            "effective_patch_weighting": "uniform",
            "effective_varpro_scaling": bool(
                tiled_inference_config.varpro_scaling
            ),
            "lattice_shape": [side, side],
            "lattice_pitch": [float(tile_size), float(tile_size)],
            "object_amplitude_scale": object_amplitude_scale,
            "object_amplitude_scale_applied": (
                simulation.object.patch_amplitude_normalization
                == "mean_patch_max"
            ),
            "object_gauge": {
                "inference_canvas_before_publication": (
                    "split_normalized"
                    if simulation.object.patch_amplitude_normalization
                    == "mean_patch_max"
                    else "raw_source"
                ),
                "published_canvas": "raw_source",
                "published_scale_factor": object_amplitude_scale,
                "count_diagnostics_canvas": (
                    "split_normalized"
                    if simulation.object.patch_amplitude_normalization
                    == "mean_patch_max"
                    else "raw_source"
                ),
            },
        }
    )
    return TiledReconstructionResult(
        complex_canvas=canvas_snapshot,
        measurement_gauge_canvas=measurement_gauge_snapshot,
        amplitude=amplitude_snapshot,
        phase=phase_snapshot,
        prescale_canvas=prescale_snapshot,
        effective_data_config=base.effective_data_config,
        canvas_weights=weight_snapshot,
        canvas_anchor=canvas_anchor,
        channel_indices=base.channel_indices,
        channel_coordinates=base.channel_coordinates,
        reassembly=reassembly,
    )


def resolve_device_and_precision(execution_config) -> tuple[str, str]:
    """Map a resolved execution config to (torch device, Lightning precision).

    Single owner of the accelerator->device mapping and precision defaulting
    used by every reconstruction door (both ``cli_main`` arms and the
    installed ``ptycho_inference`` dispatcher) — previously three drifting
    inline copies.
    """
    import torch

    device_map = {
        'cpu': 'cpu',
        'gpu': 'cuda',
        'cuda': 'cuda',
        'mps': 'mps',
        'auto': 'cuda' if torch.cuda.is_available() else 'cpu',
    }
    device = device_map.get(getattr(execution_config, 'accelerator', None), 'cpu')
    precision = getattr(execution_config, 'precision', None)
    if precision not in {'32-true', '16-mixed', 'bf16-mixed'}:
        precision = '32-true'
    return device, precision


@scoped_legacy_params
def cli_main():
    """
    CLI entrypoint for PyTorch Lightning checkpoint inference (the config-factory contract (docs/specs/spec-ptycho-config-bridge.md) Phase D.C thin wrapper).

    Thin wrapper that delegates to shared helpers (ptycho_torch.cli.shared) for validation,
    execution config construction, and device resolution, then reconstructs the full scan
    through the barycentric kernel.

    Usage:
        python -m ptycho_torch.inference \\
            --model_path <training_output_dir> \\
            --test_data <npz_file> \\
            --output_dir <inference_output_dir> \\
            --accelerator cpu \\
            [--quiet]

    Expected Output Artifacts:
        - <output_dir>/reconstructed_amplitude.png
        - <output_dir>/reconstructed_phase.png

    References:
        - Blueprint: docs/findings.md (see git history for the originating plan)
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
      --device cpu

  # Run with quiet output
  python -m ptycho_torch.inference \\
      --model_path training_outputs \\
      --test_data test.npz \\
      --output_dir outputs \\
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
        '--patch-weighting',
        choices=['uniform', 'probe'],
        default='uniform',
        dest='patch_weighting',
        help=(
            "Stitching weight for patch reassembly on the barycentric kernel "
            "path: 'uniform' (default) or 'probe' (|P|^2-weighted)."
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

    # Execution config flags (the config-factory contract (docs/specs/spec-ptycho-config-bridge.md))
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

    # The barycentric kernel is the sole reconstruction entry; the deprecated
    # uniform-stitching route was deleted at the Phase 4 closeout.
    if args.probe_mask:
        raise ValueError(
            "--probe-mask cannot be honored on the barycentric kernel path "
            "(the kernel uses the checkpoint's own probe configuration)."
        )

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

    # Delegate to factory for CONFIG-001 compliance (config-factory contract; docs/specs/spec-ptycho-config-bridge.md)
    # Replaces manual checkpoint loading and config construction with centralized
    # factory pattern. The factory handles:
    # 1. Path validation and checkpoint discovery
    # 2. CONFIG-001 bridging (update_legacy_dict before any IO)
    # 3. Config translation (PyTorch → TensorFlow canonical dataclasses)
    # 4. Execution config merging with override precedence

    from ptycho_torch.config_factory import create_inference_payload

    # Convert paths to Path objects
    model_path = Path(args.model_path)
    test_data_path = Path(args.test_data)
    output_dir = Path(args.output_dir)

    # Build overrides dict for factory
    overrides = {
        # inference_groups is the canonical inference group-count key; the
        # barycentric kernel reconstructs the full scan and ignores this count.
        'inference_groups': 32,
        'probe_mask': args.probe_mask,
        'probe_mask_sigma': args.probe_mask_sigma,
        'probe_mask_diameter': args.probe_mask_diameter,
        'log_patch_stats': args.log_patch_stats,
        'patch_stats_limit': args.patch_stats_limit,
        # docs/specs/spec-ptycho-conformance.md (D4): thread the stitching/scaling knobs so the resolved
        # pt_inference_config matches the routing decision above.
        'patch_weighting': args.patch_weighting,
        'varpro_scaling': args.varpro_scaling,
    }
    if args.scale_contract_version is not None:
        overrides['scale_contract_version'] = args.scale_contract_version
    if args.measurement_domain is not None:
        overrides['measurement_domain'] = args.measurement_domain

    # Resolve the Torch-owned configs without mutating legacy params.cfg.
    try:
        payload = create_inference_payload(
            model_path=model_path,
            test_data_file=test_data_path,
            output_dir=output_dir,
            overrides=overrides,
            execution_config=execution_request,
        )

        # Reuse the factory-owned resolved records.
        tf_inference_config = payload.tf_inference_config
        execution_config = payload.execution_config

        if not args.quiet:
            print("Loaded configuration from model checkpoint")
            print(f"Test data: {test_data_path}")
            print(f"Output directory: {output_dir}")
            print(f"N groups: {tf_inference_config.inference_groups}")
            print(f"Execution config: accelerator={execution_config.accelerator}, "
                  f"num_workers={execution_config.num_workers}")

    except Exception as e:
        raise RuntimeError(
            f"Failed to create inference payload.\n"
            f"Error: {e}\n"
            "Ensure model_path contains wts.h5.zip and test_data conforms to DATA-001."
        )

    try:
        import torch

        device, precision = resolve_device_and_precision(execution_config)
        result = reconstruct(
            model_path,
            test_data_path,
            work_dir=output_dir,
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


if __name__ == '__main__':
    sys.exit(cli_main())
