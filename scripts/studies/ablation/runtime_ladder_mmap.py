"""Generic mmap-loader wiring for the bridge ladder (Task 21b).

The mmap rungs train through the SAME canonical Lightning entry the
dictionary path uses (``ptycho_torch.workflows.components._train_with_lightning``,
which accepts a ``PtychoDataset`` container), so the loader/schema swap stays
the only variable: TF-side configs come from the runner's own
``setup_torch_configs`` and the torch-only overrides mirror
``run_torch_training``'s verbatim. The real generic loader
(``ptycho_torch.dataloader.PtychoDataset``) performs ingestion, probe
normalization, bounds filtering, and grouping; its observable state (the
stored effective probe tensor, scaling constants, grouping indices) becomes
sealed rung evidence.

Wiring pins: ``DataConfig.n_subsample`` defaults to 7 but the endpoint arm
declares 1 — the wiring pins the endpoint value (machine-checked against the
endpoint spec in tests). Runtime ``strategy='auto'`` selects the direct
TensorDictDataLoader path for these single-device ladder runs.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Literal, Mapping

import numpy as np

from .dataset_provenance import canonical_array_sha256
from .runtime_errors import RuntimeExecutionError
from .runtime_ladder_injections import (  # noqa: F401  (re-exported for tests)
    _flatten_probe_batch,
    _probe_layout_injection,
    _train_sampler_injection,
)

__all__ = [
    "BOUNDS_FILTER_MODES",
    "ENDPOINT_PINNED_N_SUBSAMPLE",
    "build_generic_loader_dataset",
    "build_mmap_dataset_payload",
    "extract_loader_evidence",
    "load_grouped_generic_test_dict",
    "recipe_stitch_metadata",
    "resolve_mmap_normalize_mode",
    "runner_torch_overrides",
    "train_via_generic_loader",
]

#: DataConfig.n_subsample defaults to 7; the endpoint arm pins 1 (each scan
#: is grouped exactly once). Machine-checked against the endpoint TOML.
ENDPOINT_PINNED_N_SUBSAMPLE = 1

#: mmap_bounds_filter modes -> (x_bounds, y_bounds) DataConfig overrides.
#: "off" keeps every scan (the dictionary path's effective behavior);
#: "endpoint" is the endpoint arm's declared range filter (machine-checked
#: against hybrid_resnet_ci_compatibility.toml in tests).
BOUNDS_FILTER_MODES: dict[str, tuple[tuple[float, float], tuple[float, float]]] = {
    "off": ((0.0, 1.0), (0.0, 1.0)),
    "endpoint": ((0.1, 0.9), (0.1, 0.9)),
}

_MMAP_NORMALIZE_MODES: dict[str, Literal["None", "Batch"]] = {
    "dictionary_parity": "None",
    "loader": "Batch",
}


def resolve_mmap_normalize_mode(
    config: Mapping[str, Any],
) -> Literal["None", "Batch"]:
    """Map the closed ladder convention to the actual DataConfig mode."""
    convention = config.get("mmap_scale_convention")
    try:
        return _MMAP_NORMALIZE_MODES[str(convention)]
    except KeyError as error:
        raise RuntimeExecutionError(
            "mmap_normalization",
            "mmap_scale_convention must be one of "
            f"{sorted(_MMAP_NORMALIZE_MODES)}; got {convention!r}",
        ) from error


def recipe_stitch_metadata(recipe: Any) -> dict[str, Any]:
    """Grid-lines stitch metadata synthesized from the pinned recipe."""
    return {
        "additional_parameters": {
            "size": recipe.size,
            "offset": recipe.offset,
            "outer_offset_train": recipe.outer_offset_train,
            "outer_offset_test": recipe.outer_offset_test,
            "nimgs_train": recipe.nimgs_train,
            "nimgs_test": recipe.nimgs_test,
            "probe_smoothing_sigma": recipe.probe_smoothing_sigma,
        }
    }


def build_mmap_dataset_payload(
    config: Mapping[str, Any], recipe: Any, train_npz: Path, payload_dir: Path
) -> Any:
    """Resolve the torch config payload for the generic loader.

    Uses the canonical ``config_factory.create_training_payload`` — the same
    factory ``_train_with_lightning`` uses internally — so the dataset is
    constructed under configs identical to the ones training will re-derive.
    """
    from ptycho_torch.config_factory import create_training_payload
    from ptycho_torch.execution_request import ExecutionRequest

    x_bounds, y_bounds = BOUNDS_FILTER_MODES[str(config["mmap_bounds_filter"])]
    overrides = {
        # Required by the factory; unused by PtychoDataset and by the direct
        # TensorDictDataLoader path the ladder runs.
        "n_groups": 1,
        "N": int(config["N"]),
        "gridsize": int(config["gridsize"]),
        "nphotons": float(recipe.nphotons),
        "model_type": "Unsupervised",
        "architecture": str(config["architecture"]),
        "batch_size": int(config["batch_size"]),
        "max_epochs": int(config["epochs"]),
        "n_subsample": ENDPOINT_PINNED_N_SUBSAMPLE,
        "x_bounds": x_bounds,
        "y_bounds": y_bounds,
        "probe_normalize": bool(config["probe_normalize"]),
        "scale_contract_version": str(config["scale_contract_version"]),
        "measurement_domain": str(config["measurement_domain"]),
        "physics_forward_mode": str(config["physics_forward_mode"]),
        "rect_s1s2_trainable": bool(config["rect_s1s2_trainable"]),
        "torch_loss_mode": str(config["torch_loss_mode"]),
        "training_patch_weighting": str(config["training_patch_weighting"]),
        "normalize": resolve_mmap_normalize_mode(config),
    }
    payload_dir = Path(payload_dir)
    payload_dir.mkdir(parents=True, exist_ok=True)
    return create_training_payload(
        train_data_file=Path(train_npz),
        output_dir=payload_dir,
        execution_config=ExecutionRequest(
            values={"strategy": "auto"},
            explicit_fields={"strategy"},
        ),
        overrides=overrides,
    )


def build_generic_loader_dataset(npz_path: Path, payload: Any, work_dir: Path) -> Any:
    """Instantiate the REAL generic mmap loader over one staged NPZ.

    RAM/IO discipline (2026-07-12 investigation): the staging is a HARDLINK
    when possible (byte-identical by inode, zero extra bytes written or
    cached; falls back to a copy across filesystems), always refreshed so a
    stale byte-different NPZ can never be silently reused, and removed after
    construction — PtychoDataset is self-contained post-init (in-memory
    data_dict + on-disk memmap) and never re-reads the source NPZ.
    """
    from ptycho_torch.dataloader import PtychoDataset

    npz_path = Path(npz_path)
    work = Path(work_dir)
    staged = work / "staged" / npz_path.stem
    staged.mkdir(parents=True, exist_ok=True)
    target = staged / npz_path.name
    if target.exists():
        target.unlink()
    try:
        os.link(npz_path, target)
    except OSError:
        shutil.copyfile(npz_path, target)
    try:
        return PtychoDataset(
            str(staged),
            payload.pt_model_config,
            payload.pt_data_config,
            training_config=payload.pt_training_config,
            data_dir=str(work / "memmap" / npz_path.stem),
            remake_map=True,
        )
    finally:
        # Unconditional staging cleanup (m3): the loader is self-contained
        # post-init, and a failed construction must not leak the link either.
        target.unlink(missing_ok=True)


def extract_loader_evidence(dataset: Any) -> dict[str, Any]:
    """Sealed evidence from the loader's OWN stored state (never recomputed).

    ``data_dict['probes'][0, 0]`` is the effective probe tensor the loader
    stored (normalized via ``helper.normalize_probe_like_tf`` when
    ``probe_normalize`` is active, passthrough otherwise) — its hash is the
    ladder's effective-probe identity. Normalization statistics come from the
    loader's legacy (``scaling_constant``) or CI (``ci_statistics``) regime.
    """
    data_dict = getattr(dataset, "data_dict", None)
    if not isinstance(data_dict, Mapping) or "probes" not in data_dict:
        raise RuntimeExecutionError(
            "loader_evidence",
            "the generic loader did not expose data_dict['probes']; cannot "
            "record the effective probe identity",
        )
    probe = np.asarray(data_dict["probes"][0, 0].cpu().numpy())
    evidence: dict[str, Any] = {
        "effective_probe_sha256": canonical_array_sha256(probe),
        # The tensor itself rides along (in-memory only, never sealed) so the
        # count/VarPro physics can use the loader's effective probe gauge.
        "effective_probe": probe,
        "probe_scaling": float(np.asarray(data_dict["probe_scaling"][0])),
    }
    normalization: dict[str, Any] = {}
    if bool(getattr(dataset, "ci_contract_active", False)):
        normalization["regime"] = "ci"
        stats = data_dict.get("ci_statistics") or {}
        for key in ("rms_input_scale", "mean_measured_intensity"):
            value = stats.get(key) if isinstance(stats, Mapping) else None
            normalization[key] = (
                None
                if value is None
                else float(np.asarray(value).reshape(-1)[0])
            )
    else:
        normalization["regime"] = "legacy"
        constant = data_dict.get("scaling_constant")
        normalization["scaling_constant"] = (
            None if constant is None else float(np.asarray(constant).reshape(-1)[0])
        )
    normalization["probe_scaling"] = evidence["probe_scaling"]
    evidence["training_normalization"] = normalization
    return evidence


def runner_torch_overrides(runner_cfg: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    """Torch-only ``_train_with_lightning`` overrides for the mmap flow.

    Mirrors the dictionary path's override keys verbatim
    (``grid_lines_torch_runner.run_torch_training``) — including the explicit
    ``amplitude_physics_gain`` (PROBE-RANK-001 §3.3, calibrated in plan
    Task 26) — plus the loader-owned pins so the internal payload matches the
    dataset's. Factored out so tests can pin the mirror against the runner.
    """
    return {
        "training_patch_weighting": runner_cfg.training_patch_weighting,
        "physics_forward_mode": runner_cfg.physics_forward_mode,
        "cnn_output_mode": runner_cfg.cnn_output_mode,
        "rect_s1s2_trainable": runner_cfg.rect_s1s2_trainable,
        "scale_contract_version": runner_cfg.scale_contract_version,
        "measurement_domain": runner_cfg.measurement_domain,
        "amplitude_physics_gain": getattr(
            runner_cfg, "amplitude_physics_gain", 1.0
        ),
        # Loader-owned pins so the internal payload matches the dataset's.
        "normalize": resolve_mmap_normalize_mode(config),
        "probe_normalize": bool(config["probe_normalize"]),
        "n_subsample": ENDPOINT_PINNED_N_SUBSAMPLE,
    }


def train_via_generic_loader(
    runner_cfg: Any,
    config: Mapping[str, Any],
    recipe: Any,
    train_npz: Path,
    test_npz: Path,
    work: Path,
) -> tuple[Any, Path, dict[str, Any]]:
    """Train through the canonical Lightning entry with the REAL mmap loader.

    Mirrors ``run_torch_training``: seeds torch/numpy from the rung seed,
    TF-side configs from ``setup_torch_configs`` and the same torch-only
    override keys, plus the loader-owned pins (resolved ``normalize``,
    ``probe_normalize``, ``n_subsample``). The held-out pair is
    loaded as a REAL ``PtychoDataset`` validation container: the direct loader path
    wraps the test container unconditionally, so a None container would put
    a None data_source under Lightning's sanity check (the rung-1 bring-up
    blocker) — and val_loss checkpoint selection needs it anyway, exactly
    like the dictionary path.
    """
    import torch

    from ptycho_torch.workflows import components

    from scripts.studies import grid_lines_torch_runner as runner_mod

    work = Path(work)
    seed = int(config["seed"])
    # Seed parity with the dictionary path (run_torch_training seeds both).
    torch.manual_seed(seed)
    np.random.seed(seed)
    training_config, execution_config = runner_mod.setup_torch_configs(runner_cfg)
    payload = build_mmap_dataset_payload(config, recipe, train_npz, work / "payload")
    dataset = build_generic_loader_dataset(train_npz, payload, work)
    val_dataset = build_generic_loader_dataset(test_npz, payload, work)
    if len(val_dataset) < 1:
        raise RuntimeExecutionError(
            "mmap_training", "held-out validation dataset resolved to 0 groups"
        )
    evidence = extract_loader_evidence(dataset)
    overrides = runner_torch_overrides(runner_cfg, config)
    sampler_regime = str(config["mmap_train_sampler"])
    probe_layout = str(config["mmap_probe_batch_shape"])
    try:
        with _probe_layout_injection(probe_layout), \
                _train_sampler_injection(dataset, sampler_regime, seed):
            results = components._train_with_lightning(
                dataset,
                val_dataset,
                training_config,
                execution_config=execution_config,
                overrides=overrides,
            )
    except UnboundLocalError as error:
        # components.py swallows loader-construction exceptions and then hits
        # an UnboundLocalError; surface the real failure site instead.
        raise RuntimeExecutionError(
            "mmap_training",
            "generic-loader dataloader construction failed upstream (the "
            "real exception was printed and swallowed by "
            "components._build_lightning_dataloaders); check the training "
            "log above for the masked error",
        ) from error
    model = results.get("model")
    if model is None and isinstance(results.get("models"), dict):
        model = results["models"].get("diffraction_to_obj")
    if model is None:
        raise RuntimeExecutionError(
            "training", "generic-loader training produced no model handle"
        )
    payload_dict: dict[str, Any] = dict(evidence)
    payload_dict["seed"] = seed
    payload_dict["train_sampler_regime"] = sampler_regime
    payload_dict["probe_batch_layout"] = probe_layout
    if config["count_scale_mode"] == "auto":
        payload_dict["physics_scaling_constant"] = _derive_auto_count_scale(
            train_npz, float(recipe.nphotons)
        )
    return model, work, payload_dict


def _derive_auto_count_scale(train_npz: Path, nphotons: float) -> float:
    """count_scale_mode=auto constant via the dict path's canonical entry
    (``helper.derive_intensity_scale_from_amplitudes``) over the same
    measurement tensor the loader ingests."""
    import torch
    from ptycho_torch import helper as hh

    with np.load(Path(train_npz), allow_pickle=False) as archive:
        measurements = np.asarray(archive["diff3d"], dtype=np.float32)
    if measurements.ndim == 3:
        measurements = measurements[..., np.newaxis]
    scale = hh.derive_intensity_scale_from_amplitudes(
        torch.as_tensor(measurements), nphotons
    )
    return float(scale)


def _bounds_eligible(
    xcoords: np.ndarray, ycoords: np.ndarray, data_config: Any
) -> np.ndarray:
    """Bounds-eligible scan indices — the loader's own filter, replicated
    verbatim from ``dataloader.PtychoDataset.calculate_length``."""
    xmin, xmax = float(np.min(xcoords)), float(np.max(xcoords))
    ymin, ymax = float(np.min(ycoords)), float(np.max(ycoords))
    x_range = xmax - xmin
    y_range = ymax - ymin
    x_lower = xmin + data_config.x_bounds[0] * x_range
    x_upper = xmin + data_config.x_bounds[1] * x_range
    y_lower = ymin + data_config.y_bounds[0] * y_range
    y_upper = ymin + data_config.y_bounds[1] * y_range
    mask = (
        (xcoords >= x_lower)
        & (xcoords <= x_upper)
        & (ycoords >= y_lower)
        & (ycoords <= y_upper)
    )
    return np.where(mask)[0]


def load_grouped_generic_test_dict(
    test_npz: Path, recipe: Any, config: Mapping[str, Any], work: Path
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Group the held-out generic data with the REAL mmap loader (C > 1).

    The loader's persisted ``nn_indices`` define the groups; the runner test
    dict is assembled mechanically from them (values bit-unchanged), with the
    dictionary-path conventions: channelled diffraction ``(G, H, W, C)``,
    global ``coords_offsets`` (index 0 = y/rows, 1 = x/cols), and
    group-centered relative ``coords_nominal``. The grouping record is the
    scan-accounting evidence source.
    """
    payload = build_mmap_dataset_payload(config, recipe, test_npz, Path(work) / "payload")
    dataset = build_generic_loader_dataset(test_npz, payload, Path(work))
    mmap = getattr(dataset, "mmap_ptycho", None)
    if mmap is None or "nn_indices" not in mmap.keys():
        raise RuntimeExecutionError(
            "grouped_ingestion",
            "the generic loader did not persist nn_indices; cannot derive the "
            "grouped held-out inputs",
        )
    nn_indices = np.asarray(mmap["nn_indices"][:].cpu().numpy(), dtype=np.int64)
    with np.load(Path(test_npz), allow_pickle=False) as archive:
        diff3d = np.asarray(archive["diff3d"], dtype=np.float32)
        xcoords = np.asarray(archive["xcoords"], dtype=np.float64)
        ycoords = np.asarray(archive["ycoords"], dtype=np.float64)
        probe = np.asarray(archive["probeGuess"])
        truth = (
            np.asarray(archive["objectGuess"])
            if "objectGuess" in archive.files
            else None
        )
    if diff3d.ndim == 4 and diff3d.shape[-1] == 1:
        diff3d = diff3d[..., 0]
    groups, channels = nn_indices.shape
    total = int(diff3d.shape[0])
    # The KDTree grouping can emit -1 sentinel members; the loader's own
    # fancy indexing (coords_global[nn_indices]) wraps them to the LAST scan,
    # so the adapter mirrors that effective behavior and the accounting
    # records the wrapped slots explicitly — surfacing exactly the silent
    # grouping substitutions the plan's accounting checkbox targets.
    sentinel_slots = int(np.count_nonzero(nn_indices < 0))
    members = np.mod(nn_indices, total)
    diffraction = np.ascontiguousarray(np.transpose(diff3d[members], (0, 2, 3, 1)))
    from ptycho_torch.coords import coords_relative_from_nominal

    offsets = np.zeros((groups, 1, 2, channels), dtype=np.float32)
    offsets[:, 0, 0, :] = ycoords[members]
    offsets[:, 0, 1, :] = xcoords[members]
    # Canonical dict-path relative offsets: sign -1, per-group centering
    # (ptycho_torch.coords.coords_relative_from_nominal == the loader's
    # get_relative_coords semantics in the dictionary layout).
    relative = coords_relative_from_nominal(offsets)
    data: dict[str, Any] = {
        "diffraction": diffraction,
        "coords_nominal": relative.astype(np.float32),
        "coords_offsets": offsets,
        "probeGuess": probe,
    }
    if truth is not None:
        data["YY_ground_truth"] = truth
    eligible = _bounds_eligible(xcoords, ycoords, payload.pt_data_config)
    grouping_record = {
        "expected_scan_ids": [int(i) for i in range(total)],
        "filtered_eligible_scan_ids": [int(i) for i in eligible],
        "used_scan_ids": [int(i) for i in np.unique(members)],
        "group_count": int(groups),
        "participant_slots": int(nn_indices.size),
        "sentinel_wrapped_slots": sentinel_slots,
    }
    return data, recipe_stitch_metadata(recipe), grouping_record
