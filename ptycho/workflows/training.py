"""Shared generic training workflow used by public training entry points.

The boundary resolves public configuration, selects and groups flat raw
acquisitions, resolves the Torch factory exactly once (delegating to
``ptycho_torch.workflows.orchestration``), and delegates model and Trainer
construction to the existing backend implementations.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import dataclass, replace
import logging
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ptycho.config import (
    TrainingConfig,
    validate_runnable_training_config,
    validate_training_config_structure,
)
from ptycho.metadata import MetadataManager
from ptycho.workflows.backend_selector import run_cdi_example_with_backend
from ptycho.workflows.config_cli import load_data, setup_configuration
from ptycho.workflows.workflow_orchestration import save_outputs


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainingWorkflowRequest:
    """One legacy-CLI or resolved-synthetic training request."""

    legacy_args: argparse.Namespace | None = None
    raw_argv: tuple[str, ...] = ()
    resolved_synthetic_workflow: Any | None = None
    train_data_file: Path | None = None
    test_data_file: Path | None = None
    output_dir: Path | None = None
    do_stitching: bool = False
    torch_training_seed: int | None = None
    batch_order_recipe: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "raw_argv", tuple(self.raw_argv))
        if self.torch_training_seed is not None:
            if (
                isinstance(self.torch_training_seed, bool)
                or not isinstance(self.torch_training_seed, int)
            ):
                raise TypeError(
                    "torch_training_seed must be a nonnegative integer"
                )
            if self.torch_training_seed < 0:
                raise ValueError(
                    "torch_training_seed must be a nonnegative integer"
                )
        if self.batch_order_recipe is not None:
            from ptycho_torch.batch_order import validate_batch_order_recipe

            validate_batch_order_recipe(self.batch_order_recipe)
        legacy = self.legacy_args is not None
        synthetic = self.resolved_synthetic_workflow is not None
        if legacy == synthetic:
            raise ValueError(
                "exactly one of legacy_args or resolved_synthetic_workflow "
                "must be supplied"
            )
        if synthetic:
            missing = [
                name
                for name in ("train_data_file", "test_data_file", "output_dir")
                if getattr(self, name) is None
            ]
            if missing:
                raise ValueError(
                    "resolved synthetic training requires " + ", ".join(missing)
                )
            if self.do_stitching:
                raise ValueError(
                    "resolved synthetic training requires do_stitching=False; "
                    "barycentric reconstruction is a separate stage"
                )


@dataclass(frozen=True)
class TrainingWorkflowResult:
    """Structured training output and persisted scientific identity."""

    public_config: TrainingConfig
    backend_results: Mapping[str, Any]
    reconstruction_amplitude: Any | None
    reconstruction_phase: Any | None
    bundle_path: Path | None
    rect_s1s2_initialization: Mapping[str, Any] | None
    training_summary_path: Path | None
    pt_data_config: Any | None
    pt_model_config: Any | None
    model_spec: Any | None
    train_group_count: int
    validation_group_count: int
    amplitude_physics_gain_record: Any | None
    amplitude_physics_gain_metadata: Mapping[str, Any] | None
    torch_training_seed: int | None
    batch_order_recipe: str | None


def resolve_training_payload(**kwargs):
    """Lazy adapter to the pure Torch training factory."""

    from ptycho_torch.config_factory import resolve_training_payload as resolve

    return resolve(**kwargs)


def resolve_amplitude_physics_gain(*args, **kwargs):
    """Lazy adapter to the selected-data scaling contract."""

    from ptycho_torch.scaling_contract import resolve_amplitude_physics_gain as resolve

    return resolve(*args, **kwargs)


def load_inference_bundle_torch(*args, **kwargs):
    """Lazy adapter to strict Torch bundle reload."""

    from ptycho_torch.workflows.bundle_io import load_inference_bundle_torch as load

    return load(*args, **kwargs)


def interpret_n_images_parameter(n_images: int, gridsize: int) -> tuple[int, str]:
    """Retain the legacy CLI explanation for ``--n-images``."""

    if gridsize == 1:
        return (
            n_images,
            f"Parameter interpretation: --n-images={n_images} refers to "
            "individual images (gridsize=1)",
        )
    total_patterns = n_images * gridsize * gridsize
    return (
        n_images,
        f"Parameter interpretation: --n-images={n_images} refers to neighbor "
        f"groups (gridsize={gridsize}, total patterns={total_patterns})",
    )


def interpret_sampling_parameters(config: TrainingConfig):
    """Resolve raw selection and exact group counts without side effects."""

    gridsize = config.model.gridsize
    if config.train_raw_selection is not None:
        raw_selection = config.train_raw_selection
        group_count = config.training_groups
        if gridsize == 1:
            message = (
                f"Independent sampling control: subsampling {raw_selection} images, "
                f"using {group_count} groups for training"
            )
        else:
            message = (
                f"Independent sampling control: subsampling {raw_selection} images, "
                f"creating {group_count} groups (approx "
                f"{group_count * gridsize * gridsize} patterns from groups)"
            )
    else:
        raw_selection = config.training_groups
        group_count = config.training_groups
        if gridsize == 1:
            message = f"Legacy mode: using {group_count} groups (gridsize=1)"
        else:
            message = (
                f"Legacy mode: --training-groups={group_count} refers to neighbor groups "
                f"(gridsize={gridsize}, approx "
                f"{group_count * gridsize * gridsize} patterns)"
            )
    # The oversampling/pool flags are retired: exact unique centers only.
    # Keep the legacy 5-tuple shape for callers that still unpack it.
    return (
        raw_selection,
        group_count,
        False,
        None,
        message,
    )


def _metadata_photon_count(path: Path) -> float | None:
    try:
        _, metadata = MetadataManager.load_with_metadata(str(path))
    except Exception as error:
        logger.debug("No metadata found or error reading metadata: %s", error)
        return None
    if not metadata:
        return None
    if "nphotons" in metadata:
        return float(metadata["nphotons"])
    physics = metadata.get("physics_parameters")
    if isinstance(physics, Mapping) and "nphotons" in physics:
        return float(physics["nphotons"])
    return None


def _resolve_metadata_photons(config: TrainingConfig) -> TrainingConfig:
    resolved = _metadata_photon_count(Path(config.train_data_file))
    if resolved is None:
        return config
    logger.info(
        "Overriding nphotons from config (%.1e) with dataset metadata: %.1e",
        config.nphotons,
        resolved,
    )
    return replace(config, nphotons=resolved)


def _torch_model_from_snapshot(resolved: Any):
    import ptycho_torch.workflows.orchestration as orchestration

    return orchestration._torch_model_from_snapshot(resolved)


def _resolve_public_config(request: TrainingWorkflowRequest) -> TrainingConfig:
    if request.legacy_args is not None:
        args = argparse.Namespace(**vars(request.legacy_args))
        if hasattr(args, "train_data_file_path"):
            args.train_data_file = args.train_data_file_path
            delattr(args, "train_data_file_path")
        config = setup_configuration(args, getattr(args, "config", None))
    else:
        import ptycho_torch.workflows.orchestration as orchestration

        config = orchestration._public_config_from_synthetic(request)
    config = _resolve_metadata_photons(config)
    validate_training_config_structure(config)
    validate_runnable_training_config(config)
    return config


def _group_raw_data(
    raw_data: Any,
    config: TrainingConfig,
    path: Path,
    *,
    group_count: int | None = None,
) -> dict:
    expected_groups = config.training_groups if group_count is None else group_count
    grouped = raw_data.generate_grouped_data(
        N=config.model.N,
        K=config.neighbor_count,
        nsamples=expected_groups,
        dataset_path=str(path),
        seed=config.subsample_seed,
        sequential_sampling=config.sequential_sampling,
        gridsize=config.model.gridsize,
    )
    actual = int(np.asarray(grouped["nn_indices"]).shape[0])
    if actual != expected_groups:
        raise ValueError(
            f"grouping produced {actual} groups; expected exactly {expected_groups}"
        )
    return grouped


def _materialize_backend_container(
    grouped: dict,
    raw_data: Any,
    config: TrainingConfig,
    *,
    data_adapter: str | None = None,
):
    if config.backend == "tensorflow":
        from ptycho import loader, params
        from ptycho.config.config import update_legacy_dict
        from ptycho.config.legacy_state import legacy_params_scope

        with legacy_params_scope():
            update_legacy_dict(params.cfg, config)
            return loader.load(
                lambda: grouped,
                raw_data.probeGuess,
                which=None,
                create_split=False,
            )
    import ptycho_torch.workflows.orchestration as orchestration

    return orchestration._materialize_torch_container(
        grouped,
        raw_data,
        data_adapter=data_adapter,
    )


@contextmanager
def _legacy_data_preparation_scope(config: TrainingConfig):
    """Project one request while legacy data consumers run, then restore it."""

    from ptycho import params
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import legacy_params_scope

    with legacy_params_scope():
        update_legacy_dict(params.cfg, config)
        yield


def _legacy_execution_and_patch(request: TrainingWorkflowRequest, config):
    import ptycho_torch.workflows.orchestration as orchestration

    return orchestration._legacy_execution_and_patch(request, config)


def _selected_truth_patches(train_raw: Any, *, N: int) -> np.ndarray:
    """Return one raw truth patch for every finalized selected frame.

    ``RawData.Y`` is already frame-aligned after ``load_data`` subsampling.  A
    flat acquisition may instead carry only the full object; in that case use
    the same singleton-coordinate convention as ``RawData.from_simulation`` so
    gain identity is independent of later neighbor grouping.
    """

    sample_count = int(np.asarray(train_raw.diff3d).shape[0])
    if getattr(train_raw, "Y", None) is not None:
        patches = np.asarray(train_raw.Y)
        if patches.shape[0] != sample_count:
            raise ValueError(
                "selected RawData.Y must contain one truth patch per selected "
                f"frame; got {patches.shape[0]} patches for {sample_count} frames"
            )
        return np.ascontiguousarray(patches)

    object_guess = getattr(train_raw, "objectGuess", None)
    if object_guess is None:
        raise ValueError(
            "selected training data requires RawData.Y or objectGuess to derive "
            "amplitude_physics_gain"
        )
    from ptycho.raw_data import get_image_patches, get_relative_coords

    coords = np.zeros((sample_count, 1, 2, 1), dtype=np.float64)
    coords[:, 0, 0, 0] = np.asarray(train_raw.xcoords)
    coords[:, 0, 1, 0] = np.asarray(train_raw.ycoords)
    global_offsets, local_offsets = get_relative_coords(coords)
    patches = get_image_patches(
        object_guess,
        global_offsets,
        local_offsets,
        N=N,
        gridsize=1,
    )
    return np.ascontiguousarray(np.asarray(patches))


def _resolve_gain(resolved: Any, train_raw: Any):
    """Resolve gain from the finalized raw training selection, never groups."""

    model = resolved.model
    override = (
        model.amplitude_physics_gain
        if model.amplitude_physics_gain_provenance == "explicit"
        else None
    )
    needs_truth = (
        model.physics_forward_mode == "amplitude" and override is None
    )
    return resolve_amplitude_physics_gain(
        np.ascontiguousarray(np.asarray(train_raw.diff3d)),
        (
            _selected_truth_patches(train_raw, N=resolved.data.N)
            if needs_truth
            else None
        ),
        np.ascontiguousarray(np.asarray(train_raw.probeGuess)),
        probe_scale=resolved.data.probe_scale,
        probe_mask=model.probe_mask,
        probe_mask_tensor=model.probe_mask_tensor,
        probe_mask_sigma=model.probe_mask_sigma,
        probe_mask_diameter=model.probe_mask_diameter,
        override=override,
        physics_forward_mode=model.physics_forward_mode,
    )


def _validate_selected_raw_count(train_raw: Any, *, expected: int) -> None:
    """Reject a clamped synthetic selection before deriving its identity."""

    actual = int(np.asarray(train_raw.diff3d).shape[0])
    if actual != expected:
        raise ValueError(
            "synthetic training requested exactly "
            f"{expected} raw frames but selected {actual}"
        )


def _persist_tensorflow_outputs(
    config: TrainingConfig,
    amplitude: Any,
    phase: Any,
    results: Mapping[str, Any],
) -> None:
    from ptycho import model_manager, params
    from ptycho.config.config import update_legacy_dict
    from ptycho.config.legacy_state import configured_params_scope, legacy_params_scope

    with legacy_params_scope():
        with configured_params_scope():
            update_legacy_dict(params.cfg, config)
            model_manager.save(str(config.output_dir))
    save_outputs(amplitude, phase, dict(results), str(config.output_dir))


def _resolve_workflow_torch_seed(
    request: TrainingWorkflowRequest,
    *,
    subsample_seed: int | None,
) -> int | None:
    """Resolve explicit historical seed or the default independent child stream."""

    explicit = request.torch_training_seed
    if request.resolved_synthetic_workflow is None:
        return explicit

    from ptycho.simulation.flat_acquisition import derive_seed_lineage

    configured = request.resolved_synthetic_workflow.training.torch_training_seed
    if explicit is not None and explicit != configured:
        raise ValueError(
            "request torch_training_seed conflicts with resolved synthetic "
            f"identity: expected {configured!r}, got {explicit!r}"
        )
    if configured is not None:
        return configured
    derived = derive_seed_lineage(
        request.resolved_synthetic_workflow.simulation.train.seed
    )["torch"]
    if derived == subsample_seed:
        raise ValueError(
            "the derived synthetic Torch and grouping seed streams must be distinct"
        )
    return derived


def run_training_workflow(
    request: TrainingWorkflowRequest,
) -> TrainingWorkflowResult:
    """Execute one generic training workflow through the existing backend."""

    if not isinstance(request, TrainingWorkflowRequest):
        raise TypeError("request must be a TrainingWorkflowRequest")
    config = _resolve_public_config(request)
    torch_training_seed = _resolve_workflow_torch_seed(
        request,
        subsample_seed=config.subsample_seed,
    )
    batch_order_recipe = None
    if config.backend != "tensorflow":
        import ptycho_torch.workflows.orchestration as orchestration

        batch_order_recipe = orchestration._resolve_workflow_batch_order_recipe(
            request
        )
    raw_selection, group_count, _, _, message = interpret_sampling_parameters(config)
    logger.info(message)
    config = replace(config, training_groups=group_count)

    with _legacy_data_preparation_scope(config):
        train_raw = load_data(
            str(config.train_data_file),
            n_images=group_count,
            n_subsample=raw_selection,
            subsample_seed=config.subsample_seed,
        )
    if request.resolved_synthetic_workflow is not None:
        _validate_selected_raw_count(train_raw, expected=raw_selection)
    gain_record = None
    payload = None
    execution_request = None
    factory_overrides = None
    if config.backend != "tensorflow":
        import ptycho_torch.workflows.orchestration as orchestration

        if request.resolved_synthetic_workflow is not None:
            resolved = request.resolved_synthetic_workflow
            gain_record = _resolve_gain(resolved, train_raw)
            execution_request = orchestration._synthetic_execution_request(
                resolved
            )
            factory_overrides = orchestration._synthetic_factory_overrides(
                resolved,
                config,
            )
            factory_overrides.update(gain_record.factory_overrides())
        else:
            execution_request, cli_patch = _legacy_execution_and_patch(
                request,
                config,
            )
            explicit_fields = getattr(
                execution_request,
                "explicit_fields",
                None,
            )
            if explicit_fields is not None and not explicit_fields:
                logger.info(
                    "POLICY-001: No --torch-* execution flags provided. "
                    "Backend will use GPU-first defaults (auto-detects CUDA "
                    "if available, else CPU). CPU-only users should pass "
                    "--torch-accelerator cpu."
                )
            # Preserve the established precedence: reconstruct the complete
            # public-config baseline, then overlay only explicit CLI aliases.
            factory_overrides = orchestration._base_factory_overrides(config)
            factory_overrides.update(cli_patch)
        payload = resolve_training_payload(
            train_data_file=Path(config.train_data_file),
            output_dir=Path(config.output_dir),
            overrides=factory_overrides,
            execution_config=execution_request,
            training_baseline=config,
        )
        payload_config = payload.tf_training_config
        orchestration._validate_payload_selection_identity(config, payload_config)
        if request.resolved_synthetic_workflow is not None:
            orchestration._validate_synthetic_payload_identity(
                resolved,
                payload,
                gain_record,
            )
        config = payload_config
        validate_training_config_structure(config)
        validate_runnable_training_config(config)

    with _legacy_data_preparation_scope(config):
        test_raw = None
        if config.test_data_file is not None:
            test_raw = load_data(
                str(config.test_data_file),
                n_images=None,
                n_subsample=None,
            )

        train_grouped = _group_raw_data(
            train_raw,
            config,
            Path(config.train_data_file),
        )
        validation_grouped = None
        if test_raw is not None:
            validation_grouped = _group_raw_data(
                test_raw,
                config,
                Path(config.test_data_file),
                group_count=(
                    request.resolved_synthetic_workflow.training.validation_groups
                    if request.resolved_synthetic_workflow is not None
                    else None
                ),
            )
        train_container = _materialize_backend_container(
            train_grouped,
            train_raw,
            config,
        )
        validation_container = (
            _materialize_backend_container(validation_grouped, test_raw, config)
            if validation_grouped is not None
            else None
        )
        if request.resolved_synthetic_workflow is not None:
            import ptycho_torch.workflows.orchestration as orchestration

            adapter_name = resolved.training.data_adapter
            orchestration._apply_backend_data_adapter(
                train_container,
                train_grouped,
                adapter_name,
            )
            if validation_container is not None:
                orchestration._apply_backend_data_adapter(
                    validation_container,
                    validation_grouped,
                    adapter_name,
                )

    amplitude, phase, backend_results = run_cdi_example_with_backend(
        train_container,
        validation_container,
        config,
        do_stitching=request.do_stitching,
        torch_execution_config=None,
        torch_factory_overrides=None,
        torch_resolved_payload=payload,
        torch_amplitude_physics_gain_record=gain_record,
        torch_training_seed=torch_training_seed,
        torch_batch_order_recipe=batch_order_recipe,
    )
    if config.backend == "tensorflow":
        _persist_tensorflow_outputs(config, amplitude, phase, backend_results)

    bundle_path = backend_results.get("bundle_path")
    if bundle_path is not None:
        bundle_path = Path(bundle_path)
    elif config.backend != "tensorflow":
        candidate = Path(config.output_dir) / "wts.h5.zip"
        if candidate.is_file():
            bundle_path = candidate

    rect_s1s2_initialization = backend_results.get("rect_s1s2_initialization")
    if rect_s1s2_initialization is not None:
        if not isinstance(rect_s1s2_initialization, Mapping):
            raise TypeError("rect_s1s2_initialization must be a mapping")
        rect_s1s2_initialization = dict(rect_s1s2_initialization)
    training_summary_path = backend_results.get("training_summary_path")
    if training_summary_path is not None:
        training_summary_path = Path(training_summary_path)
    elif config.backend != "tensorflow":
        candidate = Path(config.output_dir) / "training_summary.json"
        if candidate.is_file():
            training_summary_path = candidate

    if gain_record is not None:
        if bundle_path is None or not bundle_path.is_file():
            raise FileNotFoundError(
                f"synthetic training bundle was not written: {bundle_path}"
            )
        from ptycho.config.legacy_state import isolated_archived_params_scope

        with isolated_archived_params_scope():
            _, loaded = load_inference_bundle_torch(
                Path(config.output_dir),
            )
            if loaded.get("amplitude_physics_gain_record") is None:
                raise ValueError(
                    "strict reload did not return the persisted "
                    "amplitude_physics_gain_record"
                )

    return TrainingWorkflowResult(
        public_config=config,
        backend_results=backend_results,
        reconstruction_amplitude=amplitude,
        reconstruction_phase=phase,
        bundle_path=bundle_path,
        rect_s1s2_initialization=rect_s1s2_initialization,
        training_summary_path=training_summary_path,
        pt_data_config=(payload.pt_data_config if payload is not None else None),
        pt_model_config=(payload.pt_model_config if payload is not None else None),
        model_spec=(payload.model_spec if payload is not None else None),
        train_group_count=int(np.asarray(train_grouped["nn_indices"]).shape[0]),
        validation_group_count=(
            int(np.asarray(validation_grouped["nn_indices"]).shape[0])
            if validation_grouped is not None
            else 0
        ),
        amplitude_physics_gain_record=gain_record,
        amplitude_physics_gain_metadata=(
            gain_record.to_metadata() if gain_record is not None else None
        ),
        torch_training_seed=torch_training_seed,
        batch_order_recipe=batch_order_recipe,
    )


__all__ = [
    "TrainingWorkflowRequest",
    "TrainingWorkflowResult",
    "interpret_n_images_parameter",
    "interpret_sampling_parameters",
    "run_training_workflow",
]
