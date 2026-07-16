"""Grid-based lines workflow orchestration.

Skeleton module for running probe prep → grid simulation → training → inference →
metrics for the deprecated ptycho_lines workflow.

Data contracts: see specs/data_contracts.md
"""

from __future__ import annotations

import contextlib
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import io
from pathlib import Path
import random
import sys
from typing import Any, Dict, Iterable, Optional, Tuple
import gc
import hashlib
import json
import re
import time
import numpy as np

from ptycho.config.config import (
    DetectorSimulationConfig,
    ModelConfig,
    ProbeSimulationConfig,
    ScanSimulationConfig,
    SimulationConfig,
    SyntheticObjectConfig,
    TrainingConfig,
    simulation_config_to_dict,
    simulation_config_sha256,
    update_legacy_dict,
    validate_simulation_config,
)
from ptycho import params as p
from ptycho.config.legacy_state import scoped_legacy_params
from ptycho.simulation import probe_transform as _probe_transform
from ptycho.simulation.identity import (
    array_sha256 as _identity_array_sha256,
    build_simulation_probe_lineage,
    canonical_sha256 as _identity_canonical_sha256,
    file_sha256 as _identity_file_sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class _TeeTextStream(io.TextIOBase):
    def __init__(self, *streams: io.TextIOBase) -> None:
        self._streams = streams

    def write(self, s: str) -> int:
        for stream in self._streams:
            stream.write(s)
            stream.flush()
        return len(s)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _write_log_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@contextlib.contextmanager
def _capture_tf_row_logs(output_dir: Path, model_id: str):
    run_dir = output_dir / "runs" / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    tee_stdout = _TeeTextStream(sys.stdout, stdout_buffer)
    tee_stderr = _TeeTextStream(sys.stderr, stderr_buffer)
    try:
        with contextlib.redirect_stdout(tee_stdout), contextlib.redirect_stderr(tee_stderr):
            print(f"[row:{model_id}] Starting row-local execution")
            try:
                yield
            except Exception as exc:
                print(
                    f"[row:{model_id}] Row execution failed: {exc.__class__.__name__}: {exc}",
                    file=sys.stderr,
                )
                raise
            else:
                print(f"[row:{model_id}] Completed row-local execution")
    finally:
        _write_log_text(run_dir / "stdout.log", stdout_buffer.getvalue())
        _write_log_text(run_dir / "stderr.log", stderr_buffer.getvalue())


@dataclass
class GridLinesConfig:
    """Configuration for grid-based lines workflow.

    See docs/plans/2026-01-27-grid-lines-workflow.md for parameter details.
    """

    N: int | None = None
    gridsize: int | None = None
    output_dir: Path = Path(".")
    probe_npz: Path | None = None
    size: int | None = None
    offset: int | None = None
    outer_offset_train: int | None = None
    outer_offset_test: int | None = None
    nimgs_train: int | None = None
    nimgs_test: int | None = None
    nphotons: float | None = None
    nepochs: int = 60
    batch_size: int = 16
    nll_weight: float = 0.0
    mae_weight: float = 1.0
    realspace_weight: float = 0.0
    probe_smoothing_sigma: float | None = None
    probe_mask_diameter: Optional[int] = None
    probe_source: str | None = None
    probe_scale_mode: str | None = None
    probe_transform_pipeline: Optional[str] = None
    set_phi: bool | None = None
    seed: Optional[int] = None
    simulation: SimulationConfig | None = None

    def __post_init__(self) -> None:
        self.output_dir = Path(self.output_dir)
        nested_simulation_supplied = self.simulation is not None
        explicit = {
            "N": self.N,
            "gridsize": self.gridsize,
            "probe_npz": self.probe_npz,
            "size": self.size,
            "offset": self.offset,
            "outer_offset_train": self.outer_offset_train,
            "outer_offset_test": self.outer_offset_test,
            "nimgs_train": self.nimgs_train,
            "nimgs_test": self.nimgs_test,
            "nphotons": self.nphotons,
            "probe_smoothing_sigma": self.probe_smoothing_sigma,
            "probe_mask_diameter": self.probe_mask_diameter,
            "probe_source": self.probe_source,
            "probe_scale_mode": self.probe_scale_mode,
            "probe_transform_pipeline": self.probe_transform_pipeline,
            "set_phi": self.set_phi,
            "seed": self.seed,
        }
        if self.simulation is None:
            simulation = _simulation_from_flat_grid_lines(explicit)
        else:
            validate_simulation_config(self.simulation)
            _reject_grid_lines_simulation_conflicts(self.simulation, explicit)
            simulation = self.simulation

        self.simulation = simulation
        self.N = simulation.N
        self.gridsize = simulation.scan.grid_size[0]
        self.probe_npz = (
            Path(explicit["probe_npz"])
            if explicit["probe_npz"] is not None
            else simulation.probe.source_path
        )
        self.size = simulation.object.image_size[0]
        self.offset = simulation.scan.offset
        self.outer_offset_train = simulation.scan.outer_offset_train
        self.outer_offset_test = simulation.scan.outer_offset_test
        self.nimgs_train = simulation.scan.train_groups
        self.nimgs_test = simulation.scan.test_groups
        self.nphotons = simulation.detector.photons_per_pattern
        self.probe_mask_diameter = simulation.probe.mask_diameter
        self.probe_source = (
            "ideal_disk" if simulation.probe.source == "ideal" else "custom"
        )
        self.probe_transform_pipeline = simulation.probe.transform_pipeline
        self.probe_scale_mode = explicit["probe_scale_mode"] or (
            "pipeline"
            if nested_simulation_supplied
            or explicit["probe_transform_pipeline"] is not None
            else "pad_preserve"
        )
        self.probe_smoothing_sigma = (
            explicit["probe_smoothing_sigma"]
            if explicit["probe_smoothing_sigma"] is not None
            else _pipeline_smoothing_sigma(simulation.probe.transform_pipeline)
        )
        self.set_phi = simulation.object.set_phi
        self.seed = simulation.seed


def _pipeline_smoothing_sigma(pipeline: str) -> float:
    for segment in pipeline.split("|"):
        op, separator, value = segment.partition(":")
        if op.strip() == "smooth" and separator:
            return float(value)
    return 0.0


def _legacy_grid_lines_pipeline(
    *,
    N: int,
    scale_mode: str,
    smoothing_sigma: float,
) -> str:
    smooth = f"smooth:{smoothing_sigma:g}"
    if scale_mode == "interpolate":
        steps = [f"interp:{N}"]
        if smoothing_sigma > 0:
            steps.append(smooth)
    elif scale_mode == "pad_preserve":
        steps = []
        if smoothing_sigma > 0:
            steps.append(smooth)
        steps.append(f"pad_preserve:{N}")
    elif scale_mode == "pad_extrapolate":
        steps = [f"pad_extrapolate:{N}"]
        if smoothing_sigma > 0:
            steps.append(smooth)
    elif scale_mode == "pipeline":
        raise ValueError(
            "probe_transform_pipeline must be provided when probe_scale_mode='pipeline'"
        )
    else:
        raise ValueError(f"Unknown probe_scale_mode {scale_mode!r}")
    return "|".join(steps)


def _simulation_from_flat_grid_lines(explicit: dict[str, object]) -> SimulationConfig:
    if explicit["probe_transform_pipeline"] is not None:
        if explicit["probe_scale_mode"] not in {None, "pipeline"}:
            raise ValueError(
                "probe_transform_pipeline conflicts with probe_scale_mode"
            )
        if explicit["probe_smoothing_sigma"] not in {None, 0.0}:
            raise ValueError(
                "probe_transform_pipeline conflicts with probe_smoothing_sigma"
            )
    N = int(explicit["N"] if explicit["N"] is not None else 64)
    gridsize = int(
        explicit["gridsize"] if explicit["gridsize"] is not None else 1
    )
    smoothing_sigma = float(
        explicit["probe_smoothing_sigma"]
        if explicit["probe_smoothing_sigma"] is not None
        else 0.5
    )
    scale_mode = str(explicit["probe_scale_mode"] or "pad_preserve")
    pipeline = explicit["probe_transform_pipeline"] or _legacy_grid_lines_pipeline(
        N=N,
        scale_mode=scale_mode,
        smoothing_sigma=smoothing_sigma,
    )
    source_name = str(explicit["probe_source"] or "custom")
    if source_name not in {"custom", "ideal", "ideal_disk"}:
        raise ValueError(f"Unsupported probe_source {source_name!r}")
    source = "ideal" if source_name in {"ideal", "ideal_disk"} else "custom"
    source_path = (
        None
        if source == "ideal"
        else Path(explicit["probe_npz"])
        if explicit["probe_npz"] is not None
        else None
    )
    simulation = SimulationConfig(
        N=N,
        probe=ProbeSimulationConfig(
            source=source,
            source_path=source_path,
            transform_pipeline=str(pipeline),
            mask_diameter=explicit["probe_mask_diameter"],
        ),
        object=SyntheticObjectConfig(
            kind="lines",
            image_size=(
                int(explicit["size"] if explicit["size"] is not None else 392),
            )
            * 2,
            set_phi=bool(explicit["set_phi"] or False),
        ),
        scan=ScanSimulationConfig(
            kind="grid",
            grid_size=(gridsize, gridsize),
            offset=int(explicit["offset"] if explicit["offset"] is not None else 4),
            outer_offset_train=int(
                explicit["outer_offset_train"]
                if explicit["outer_offset_train"] is not None
                else 8
            ),
            outer_offset_test=int(
                explicit["outer_offset_test"]
                if explicit["outer_offset_test"] is not None
                else 20
            ),
            train_groups=int(
                explicit["nimgs_train"]
                if explicit["nimgs_train"] is not None
                else 2
            ),
            test_groups=int(
                explicit["nimgs_test"]
                if explicit["nimgs_test"] is not None
                else 2
            ),
        ),
        detector=DetectorSimulationConfig(
            photons_per_pattern=float(
                explicit["nphotons"]
                if explicit["nphotons"] is not None
                else 1e9
            )
        ),
        seed=explicit["seed"],
    )
    validate_simulation_config(simulation)
    return simulation


def _reject_grid_lines_simulation_conflicts(
    simulation: SimulationConfig,
    explicit: dict[str, object],
) -> None:
    comparisons = {
        "N": (simulation.N, "simulation.N"),
        "gridsize": (simulation.scan.grid_size[0], "simulation.scan.grid_size"),
        "probe_npz": (simulation.probe.source_path, "simulation.probe.source_path"),
        "size": (simulation.object.image_size[0], "simulation.object.image_size"),
        "offset": (simulation.scan.offset, "simulation.scan.offset"),
        "outer_offset_train": (
            simulation.scan.outer_offset_train,
            "simulation.scan.outer_offset_train",
        ),
        "outer_offset_test": (
            simulation.scan.outer_offset_test,
            "simulation.scan.outer_offset_test",
        ),
        "nimgs_train": (simulation.scan.train_groups, "simulation.scan.train_groups"),
        "nimgs_test": (simulation.scan.test_groups, "simulation.scan.test_groups"),
        "nphotons": (
            simulation.detector.photons_per_pattern,
            "simulation.detector.photons_per_pattern",
        ),
        "probe_mask_diameter": (
            simulation.probe.mask_diameter,
            "simulation.probe.mask_diameter",
        ),
        "probe_transform_pipeline": (
            simulation.probe.transform_pipeline,
            "simulation.probe.transform_pipeline",
        ),
        "set_phi": (simulation.object.set_phi, "simulation.object.set_phi"),
        "seed": (simulation.seed, "simulation.seed"),
    }
    for flat_path, (nested_value, nested_path) in comparisons.items():
        flat_value = explicit[flat_path]
        if flat_value is None:
            continue
        if flat_path == "probe_npz":
            flat_value = Path(flat_value)
        if flat_value != nested_value:
            raise ValueError(
                f"{flat_path}={flat_value!r} conflicts with {nested_path}={nested_value!r}"
            )

    if explicit["probe_source"] is not None:
        flat_source = str(explicit["probe_source"])
        flat_source = "ideal" if flat_source in {"ideal", "ideal_disk"} else flat_source
        if flat_source != simulation.probe.source:
            raise ValueError(
                f"probe_source={explicit['probe_source']!r} conflicts with "
                f"simulation.probe.source={simulation.probe.source!r}"
            )

    if (
        explicit["probe_transform_pipeline"] is None
        and (
            explicit["probe_scale_mode"] is not None
            or explicit["probe_smoothing_sigma"] is not None
        )
    ):
        candidate = _legacy_grid_lines_pipeline(
            N=simulation.N,
            scale_mode=str(explicit["probe_scale_mode"] or "pad_preserve"),
            smoothing_sigma=float(
                explicit["probe_smoothing_sigma"]
                if explicit["probe_smoothing_sigma"] is not None
                else 0.5
            ),
        )
        if candidate != simulation.probe.transform_pipeline:
            raise ValueError(
                "probe_scale_mode/probe_smoothing_sigma resolves to "
                f"{candidate!r}, conflicting with simulation.probe.transform_pipeline="
                f"{simulation.probe.transform_pipeline!r}"
            )


# ---------------------------------------------------------------------------
# Probe Extraction + Upscaling Helpers (Task 2)
# ---------------------------------------------------------------------------


def load_probe_guess(npz_path: Path) -> np.ndarray:
    """Load probeGuess from NPZ file."""
    data = np.load(npz_path)
    if "probeGuess" not in data:
        raise KeyError("probeGuess missing from probe npz")
    return data["probeGuess"]


def load_ideal_disk_probe(N: int) -> np.ndarray:
    """Return the idealized disk probe at size N as complex64."""
    from ptycho import probe as probe_mod

    probe_np = probe_mod.get_default_probe(N, fmt="np")
    return np.asarray(probe_np, dtype=np.complex64)


def _load_configured_probe(cfg: GridLinesConfig) -> np.ndarray:
    if cfg.simulation.probe.source == "ideal":
        return load_ideal_disk_probe(cfg.N)
    if cfg.simulation.probe.source_path is None:
        raise ValueError(
            "simulation.probe.source_path is required when "
            "simulation.probe.source='custom'"
        )
    return load_probe_guess(cfg.simulation.probe.source_path)


def scale_probe(
    probe: np.ndarray,
    target_N: int,
    smoothing_sigma: float,
    scale_mode: str = "pad_extrapolate",
    probe_transform_pipeline: str | None = None,
) -> np.ndarray:
    """Resize probe to target_N and optionally smooth.

    Modes:
        - interpolate: cubic spline interpolation on real/imag parts.
        - pad_preserve: smooth at source resolution, then center-pad the complex probe.
        - pad_extrapolate: edge-pad amplitude + quadratic phase extrapolation.
    """
    normalized_pipeline, steps = normalize_probe_transform_pipeline(
        target_N=target_N,
        probe_shape=probe.shape,
        probe_scale_mode=scale_mode,
        probe_smoothing_sigma=smoothing_sigma,
        probe_transform_pipeline=probe_transform_pipeline,
    )
    _ = normalized_pipeline
    return apply_probe_transform_pipeline(probe, steps)


def scale_probe_with_mode(
    probe: np.ndarray,
    target_N: int,
    smoothing_sigma: float,
    scale_mode: str = "pad_extrapolate",
    probe_transform_pipeline: str | None = None,
) -> np.ndarray:
    """Resize probe to target_N and optionally smooth using specified mode.

    Modes:
        - interpolate: cubic spline interpolation on real/imag parts.
        - pad_preserve: smooth at source resolution, then center-pad the complex probe.
        - pad_extrapolate: edge-pad amplitude + quadratic phase extrapolation.
    """
    return scale_probe(
        probe,
        target_N,
        smoothing_sigma,
        scale_mode=scale_mode,
        probe_transform_pipeline=probe_transform_pipeline,
    )


# The reusable implementation lives under ptycho.simulation. Keep these names
# as workflow-level re-exports for historical callers and cached study code.
parse_probe_transform_pipeline = _probe_transform.parse_probe_transform_pipeline
_serialize_probe_transform_pipeline = _probe_transform.serialize_probe_transform_pipeline
normalize_probe_transform_pipeline = _probe_transform.normalize_probe_transform_pipeline
apply_probe_transform_pipeline = _probe_transform.apply_probe_transform_pipeline
apply_probe_transform_pipeline_with_metadata = (
    _probe_transform.apply_probe_transform_pipeline_with_metadata
)
smooth_complex_array = _probe_transform.smooth_complex_array
interpolate_array = _probe_transform.interpolate_array
make_disk_mask = _probe_transform.make_disk_mask
apply_probe_mask = _probe_transform.apply_probe_mask


# ---------------------------------------------------------------------------
# Simulation + Dataset Persistence (Task 3)
# ---------------------------------------------------------------------------


def configure_legacy_params(cfg: GridLinesConfig, probe_np: np.ndarray) -> TrainingConfig:
    """Configure legacy params.cfg and return a TrainingConfig.

    Must be called before generate_data() to set up legacy global state.
    """
    simulation = cfg.simulation
    if simulation.object.kind != "lines" or simulation.scan.kind != "grid":
        raise ValueError(
            "GridLinesConfig generation requires simulation.object.kind='lines' "
            "and simulation.scan.kind='grid'"
        )
    update_legacy_dict(p.cfg, simulation)

    config = TrainingConfig(
        model=ModelConfig(N=cfg.N, gridsize=cfg.gridsize, object_big=False),
        nphotons=cfg.nphotons,
        nepochs=cfg.nepochs,
        batch_size=cfg.batch_size,
        nll_weight=cfg.nll_weight,
        mae_weight=cfg.mae_weight,
        realspace_weight=cfg.realspace_weight,
    )
    update_legacy_dict(p.cfg, config)
    p.set("sim_jitter_scale", 0.0)
    from ptycho import probe as probe_mod

    probe_mod.set_probe_guess(probe_guess=probe_np)
    return config


def _capture_simulation_probe() -> np.ndarray | None:
    """Capture the set_probe-normalized simulation probe from legacy params.

    Returns params['probe'] (the post-set_probe illumination actually used by the
    simulator) as a complex64 array with the trailing singleton channel squeezed
    to match the probeGuess convention ((N, N, 1) -> (N, N)). Returns None when no
    probe is registered, keeping the key optional.
    """
    probe = p.get("probe", None)
    if probe is None:
        return None
    probe = np.asarray(probe)
    if probe.ndim == 3 and probe.shape[-1] == 1:
        probe = probe[:, :, 0]
    return probe.astype(np.complex64)


@scoped_legacy_params
def simulate_grid_data(cfg: GridLinesConfig, probe_np: np.ndarray) -> Dict[str, Any]:
    """Run simulation via data_preprocessing.generate_data and return split data."""
    configure_legacy_params(cfg, probe_np)
    from ptycho import data_preprocessing

    (
        X_tr, YI_tr, Yphi_tr,
        X_te, YI_te, Yphi_te,
        YY_gt, dataset, YY_full, norm_Y_I
    ) = data_preprocessing.generate_data()

    # Capture the set_probe-normalized illumination before any backend/param reset.
    probe_simulated = _capture_simulation_probe()

    def _build_scan_positions(
        container: Any,
        *,
        n_repeats: int,
        outer_offset: int,
        n_samples: int,
        channels: int,
    ) -> np.ndarray | None:
        coords_offsets = getattr(container, "global_offsets", None)
        if coords_offsets is not None:
            coords_offsets = np.asarray(coords_offsets)
            if coords_offsets.ndim == 4 and coords_offsets.shape[0] == n_samples:
                if np.unique(coords_offsets).size > 1:
                    return coords_offsets

        # Simulated gridsize=1 data frequently carries degenerate relative coordinates.
        # Reconstruct global scan positions for interop from the simulation geometry.
        from ptycho import diffsim

        ix, iy = diffsim.extract_coords(
            size=cfg.size,
            repeats=n_repeats,
            coord_type="global",
            outer_offset=outer_offset,
        )
        ix = np.asarray(ix)
        iy = np.asarray(iy)
        if ix.ndim != 4 or iy.ndim != 4:
            return None

        coords_global = np.zeros((ix.shape[0], 1, 2, ix.shape[3]), dtype=np.float32)
        coords_global[:, 0, 0, :] = iy[:, 0, 0, :]
        coords_global[:, 0, 1, :] = ix[:, 0, 0, :]

        if coords_global.shape[0] < n_samples:
            return None
        coords_global = coords_global[:n_samples]

        if coords_global.shape[3] < channels:
            return None
        coords_global = coords_global[..., :channels]
        return coords_global

    train_offsets = _build_scan_positions(
        dataset.train_data,
        n_repeats=cfg.nimgs_train,
        outer_offset=cfg.outer_offset_train,
        n_samples=int(np.asarray(X_tr).shape[0]),
        channels=int(np.asarray(X_tr).shape[-1]),
    )
    test_offsets = _build_scan_positions(
        dataset.test_data,
        n_repeats=cfg.nimgs_test,
        outer_offset=cfg.outer_offset_test,
        n_samples=int(np.asarray(X_te).shape[0]),
        channels=int(np.asarray(X_te).shape[-1]),
    )

    return {
        "train": {
            "X": X_tr,
            "Y_I": YI_tr,
            "Y_phi": Yphi_tr,
            "coords_nominal": dataset.train_data.coords_nominal,
            "coords_true": dataset.train_data.coords_true,
            "coords_offsets": train_offsets,
            "YY_full": dataset.train_data.YY_full,
            "probe_simulated": probe_simulated,
            "container": dataset.train_data,
        },
        "test": {
            "X": X_te,
            "Y_I": YI_te,
            "Y_phi": Yphi_te,
            "coords_nominal": dataset.test_data.coords_nominal,
            "coords_true": dataset.test_data.coords_true,
            "coords_offsets": test_offsets,
            "YY_full": dataset.test_data.YY_full,
            "YY_ground_truth": YY_gt,
            "norm_Y_I": norm_Y_I,
            "probe_simulated": probe_simulated,
            "container": dataset.test_data,
        },
        "intensity_scale": p.get("intensity_scale"),
    }


def dataset_out_dir(cfg: GridLinesConfig) -> Path:
    """Return the simulation-identity-bearing dataset output directory."""
    simulation_digest = simulation_config_sha256(cfg.simulation)
    return (
        cfg.output_dir
        / "datasets"
        / f"N{cfg.N}"
        / f"gs{cfg.gridsize}"
        / f"simulation-{simulation_digest}"
    )


def _sha256_file(path: Path) -> str:
    return _identity_file_sha256(path)


def _sha256_array(value: np.ndarray) -> str:
    return _identity_array_sha256(value)


def _canonical_sha256(value: object) -> str:
    return _identity_canonical_sha256(value)


def _build_probe_lineage(
    cfg: GridLinesConfig,
    *,
    raw_probe: np.ndarray,
    normalized_pipeline: str,
    transformed_probe: np.ndarray,
    transform_metadata: dict[str, object],
) -> dict[str, object]:
    """Build stable recipe identity plus measured transform evidence."""
    return build_simulation_probe_lineage(
        cfg.simulation,
        raw_probe=raw_probe,
        normalized_pipeline=normalized_pipeline,
        transformed_probe=transformed_probe,
        transform_metadata=transform_metadata,
    )


def _reject_mismatched_dataset_reuse(
    path: Path,
    expected_simulation_digest: str,
    expected_recipe_digest: str | None,
) -> None:
    if not path.exists():
        return
    from ptycho.metadata import MetadataManager

    _, metadata = MetadataManager.load_with_metadata(str(path))
    additional = {} if metadata is None else metadata.get("additional_parameters", {})
    existing_simulation = additional.get("simulation_config_sha256")
    existing_recipe = additional.get("dataset_recipe_sha256")
    if existing_simulation != expected_simulation_digest or (
        expected_recipe_digest is not None
        and existing_recipe != expected_recipe_digest
    ):
        raise ValueError(
            f"existing dataset {path} has simulation_config_sha256="
            f"{existing_simulation!r}, dataset_recipe_sha256={existing_recipe!r}; "
            f"requested {expected_simulation_digest!r}/{expected_recipe_digest!r}. "
            "Use a distinct output identity."
        )


def _preflight_dataset_pair_reuse(
    cfg: GridLinesConfig,
    probe_lineage: dict[str, object],
) -> None:
    """Validate both split destinations before simulation or either write."""

    output_dir = dataset_out_dir(cfg)
    simulation_digest = str(probe_lineage["simulation_config_sha256"])
    recipe_value = probe_lineage.get("dataset_recipe_sha256")
    recipe_digest = None if recipe_value is None else str(recipe_value)
    for split in ("train", "test"):
        _reject_mismatched_dataset_reuse(
            output_dir / f"{split}.npz",
            simulation_digest,
            recipe_digest,
        )


def _derive_complete_probe_lineage(
    cfg: GridLinesConfig,
    *,
    normalized_pipeline: str,
    normalized_steps: list[dict[str, object]],
    stored_probe: np.ndarray | None,
) -> dict[str, object] | None:
    """Reconstruct complete lineage for established direct-save callers."""

    if cfg.simulation.probe.source == "ideal":
        raw_probe = load_ideal_disk_probe(cfg.N)
    else:
        source_path = cfg.simulation.probe.source_path
        if source_path is None or not source_path.is_file():
            return None
        raw_probe = load_probe_guess(source_path)
    transform_result = apply_probe_transform_pipeline_with_metadata(
        raw_probe, normalized_steps
    )
    transformed_probe = apply_probe_mask(
        transform_result.probe, cfg.simulation.probe.mask_diameter
    )
    if stored_probe is not None and not np.array_equal(
        np.asarray(stored_probe), transformed_probe
    ):
        raise ValueError(
            "stored probeGuess conflicts with the resolved simulation probe recipe"
        )
    return _build_probe_lineage(
        cfg,
        raw_probe=raw_probe,
        normalized_pipeline=normalized_pipeline,
        transformed_probe=transformed_probe,
        transform_metadata=transform_result.metadata,
    )


def save_split_npz(
    cfg: GridLinesConfig,
    split: str,
    data: Dict[str, Any],
    config: TrainingConfig,
    *,
    probe_transform_pipeline: str | None = None,
    probe_transform_steps: list[dict[str, object]] | None = None,
    probe_lineage: dict[str, object] | None = None,
) -> Path:
    """Save train or test split as NPZ with metadata."""
    from ptycho.metadata import MetadataManager

    out_dir = dataset_out_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.npz"

    payload = {
        "diffraction": data["X"],
        "Y_I": data["Y_I"],
        "Y_phi": data["Y_phi"],
        "coords_nominal": data["coords_nominal"],
        "coords_true": data["coords_true"],
        "YY_full": data["YY_full"],
    }
    if data.get("coords_offsets") is not None:
        payload["coords_offsets"] = data["coords_offsets"]
    if data.get("probeGuess") is not None:
        payload["probeGuess"] = data["probeGuess"]
    if data.get("probe_simulated") is not None:
        payload["probe_simulated"] = np.asarray(data["probe_simulated"], dtype=np.complex64)
    if split == "test":
        if data.get("YY_ground_truth") is not None:
            payload["YY_ground_truth"] = data["YY_ground_truth"]
        if data.get("norm_Y_I") is not None:
            payload["norm_Y_I"] = np.array(data["norm_Y_I"])

    if probe_transform_pipeline is None or probe_transform_steps is None:
        if cfg.probe_transform_pipeline:
            normalized_steps = parse_probe_transform_pipeline(cfg.probe_transform_pipeline)
            normalized_pipeline = _serialize_probe_transform_pipeline(normalized_steps)
        else:
            probe_guess = np.asarray(data.get("probeGuess")) if data.get("probeGuess") is not None else None
            normalized_pipeline, normalized_steps = normalize_probe_transform_pipeline(
                target_N=cfg.N,
                probe_shape=probe_guess.shape if probe_guess is not None else (cfg.N, cfg.N),
                probe_scale_mode=cfg.probe_scale_mode,
                probe_smoothing_sigma=cfg.probe_smoothing_sigma,
                probe_transform_pipeline=cfg.probe_transform_pipeline,
            )
    else:
        normalized_pipeline = probe_transform_pipeline
        normalized_steps = probe_transform_steps

    if probe_lineage is None:
        transformed = data.get("probeGuess")
        probe_lineage = _derive_complete_probe_lineage(
            cfg,
            normalized_pipeline=normalized_pipeline,
            normalized_steps=normalized_steps,
            stored_probe=(None if transformed is None else np.asarray(transformed)),
        )
        if probe_lineage is None:
            simulation_payload = simulation_config_to_dict(cfg.simulation)
            simulation_digest = _canonical_sha256(simulation_payload)
            probe_lineage = {
                "simulation_config": simulation_payload,
                "simulation_config_sha256": simulation_digest,
                "dataset_recipe_sha256": None,
                "probe_lineage": {
                    "source_kind": cfg.simulation.probe.source,
                    "source_path": (
                        str(cfg.simulation.probe.source_path)
                        if cfg.simulation.probe.source_path is not None
                        else None
                    ),
                    "source_file_sha256": None,
                    "raw_probe_sha256": None,
                    "normalized_transform_pipeline": normalized_pipeline,
                    "transformed_probe_sha256": (
                        _sha256_array(transformed) if transformed is not None else None
                    ),
                },
            }
    _preflight_dataset_pair_reuse(cfg, probe_lineage)

    metadata = MetadataManager.create_metadata(
        config,
        script_name="grid_lines_workflow",
        size=cfg.size,
        offset=cfg.offset,
        outer_offset_train=cfg.outer_offset_train,
        outer_offset_test=cfg.outer_offset_test,
        nimgs_train=cfg.nimgs_train,
        nimgs_test=cfg.nimgs_test,
        probe_mask_diameter=cfg.probe_mask_diameter,
        probe_source=cfg.probe_source,
        probe_scale_mode=cfg.probe_scale_mode,
        probe_smoothing_sigma=cfg.probe_smoothing_sigma,
        probe_transform_pipeline=normalized_pipeline,
        probe_transform_steps=normalized_steps,
        probe_npz=str(cfg.probe_npz),
        set_phi=cfg.set_phi,
        coords_type="relative",
        **probe_lineage,
    )
    MetadataManager.save_with_metadata(str(path), payload, metadata)
    return path


def build_grid_lines_datasets(
    cfg: GridLinesConfig,
    dataset_tag: str | None = None,
    canonical_gt_label: str = "gt",
) -> Dict[str, str]:
    """Build train/test NPZ datasets and persist a canonical GT recon artifact."""
    probe_guess = _load_configured_probe(cfg)
    normalized_pipeline, normalized_steps = normalize_probe_transform_pipeline(
        target_N=cfg.N,
        probe_shape=probe_guess.shape,
        probe_scale_mode=cfg.probe_scale_mode,
        probe_smoothing_sigma=cfg.probe_smoothing_sigma,
        probe_transform_pipeline=cfg.probe_transform_pipeline,
    )
    transform_result = apply_probe_transform_pipeline_with_metadata(
        probe_guess, normalized_steps
    )
    probe_scaled = transform_result.probe
    probe_scaled = apply_probe_mask(probe_scaled, cfg.probe_mask_diameter)
    probe_lineage = _build_probe_lineage(
        cfg,
        raw_probe=probe_guess,
        normalized_pipeline=normalized_pipeline,
        transformed_probe=probe_scaled,
        transform_metadata=transform_result.metadata,
    )
    _preflight_dataset_pair_reuse(cfg, probe_lineage)

    sim = simulate_grid_data(cfg, probe_scaled)
    config = configure_legacy_params(cfg, probe_scaled)

    sim["train"]["probeGuess"] = probe_scaled
    sim["test"]["probeGuess"] = probe_scaled
    train_npz = save_split_npz(
        cfg,
        "train",
        sim["train"],
        config,
        probe_transform_pipeline=normalized_pipeline,
        probe_transform_steps=normalized_steps,
        probe_lineage=probe_lineage,
    )
    test_npz = save_split_npz(
        cfg,
        "test",
        sim["test"],
        config,
        probe_transform_pipeline=normalized_pipeline,
        probe_transform_steps=normalized_steps,
        probe_lineage=probe_lineage,
    )

    tag = dataset_tag or f"N{cfg.N}"
    gt_path = cfg.output_dir / "recons" / canonical_gt_label / "recon.npz"
    gt_complex = np.squeeze(sim["test"]["YY_ground_truth"])
    if gt_path.exists():
        with np.load(gt_path) as existing:
            existing_gt = np.squeeze(existing["YY_pred"])
        if existing_gt.shape != gt_complex.shape or not np.allclose(
            existing_gt,
            gt_complex,
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(
                "Canonical GT mismatch across N builds; enforce shared synthetic object identity/seed"
            )
    else:
        gt_path = save_recon_artifact(cfg.output_dir, canonical_gt_label, gt_complex)

    return {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "gt_recon": str(gt_path),
        "tag": tag,
    }


def build_grid_lines_datasets_by_n(
    base_cfg: GridLinesConfig,
    required_ns: Iterable[int],
) -> Dict[int, Dict[str, str]]:
    """Build dataset bundles for each unique required N value."""
    bundles: Dict[int, Dict[str, str]] = {}
    for n_value in sorted(set(required_ns)):
        _reset_backend_state()
        resized_steps = [
            ({**step, "target_N": n_value} if "target_N" in step else dict(step))
            for step in parse_probe_transform_pipeline(
                base_cfg.simulation.probe.transform_pipeline
            )
        ]
        simulation_n = replace(
            base_cfg.simulation,
            N=n_value,
            probe=replace(
                base_cfg.simulation.probe,
                transform_pipeline=_serialize_probe_transform_pipeline(resized_steps),
            ),
        )
        cfg_n = replace(
            base_cfg,
            N=None,
            gridsize=None,
            probe_npz=None,
            size=None,
            offset=None,
            outer_offset_train=None,
            outer_offset_test=None,
            nimgs_train=None,
            nimgs_test=None,
            nphotons=None,
            probe_smoothing_sigma=None,
            probe_mask_diameter=None,
            probe_source=None,
            probe_scale_mode=None,
            probe_transform_pipeline=None,
            set_phi=None,
            seed=None,
            simulation=simulation_n,
        )
        bundles[n_value] = build_grid_lines_datasets(
            cfg_n,
            dataset_tag=f"N{n_value}",
            canonical_gt_label="gt",
        )
    return bundles


def _reset_backend_state() -> None:
    """Best-effort backend cleanup between heavy multi-N dataset builds."""
    try:
        import tensorflow as tf

        tf.keras.backend.clear_session()
    except Exception:
        pass
    gc.collect()


# ---------------------------------------------------------------------------
# Stitching Helper (Task 4) - gridsize=1 safe
# ---------------------------------------------------------------------------


def stitch_predictions(predictions: np.ndarray, norm_Y_I: float, part: str = "amp") -> np.ndarray:
    """Stitch model predictions, bypassing the incorrect gridsize=1 guard.

    NOTE: This function exists because data_preprocessing.stitch_data() has an
    incorrect ValueError guard for gridsize=1. The original stitching math works
    fine for gridsize=1 (produces 1x1 grid).

    Bug ref: STITCH-GRIDSIZE-001

    Contract: STITCH-001
    - Handles both gridsize=1 and gridsize>1
    - For gridsize>1, reshapes channels into spatial grid before stitching
    - Uses outer_offset_test from params.cfg for border clipping
    - Returns stitched array with last dimension = 1

    Args:
        predictions: Model output, shape (batch, N, N, gridsize^2) or complex
        norm_Y_I: Normalization factor from simulation
        part: 'amp', 'phase', or 'complex'

    Returns:
        Stitched images, shape (n_test, H, W, 1)
    """
    nimgs = p.get("nimgs_test")
    outer_offset = p.get("outer_offset_test")
    N = p.cfg["N"]
    gridsize = p.cfg["gridsize"]

    if part == "amp":
        getpart = np.absolute
    elif part == "phase":
        getpart = np.angle
    else:
        getpart = lambda x: x

    # Apply part extraction
    processed = getpart(predictions)

    # Handle gridsize>1: reshape channels to spatial grid
    # Input: (batch, N, N, gridsize^2)
    # Output: (batch*gridsize^2, N, N, 1) with patches reordered spatially
    if gridsize > 1 and len(processed.shape) == 4 and processed.shape[-1] == gridsize**2:
        batch = processed.shape[0]
        # Reshape to (batch, N, N, gridsize, gridsize)
        processed = processed.reshape(batch, N, N, gridsize, gridsize)
        # Transpose to (batch, gridsize, gridsize, N, N)
        processed = processed.transpose(0, 3, 4, 1, 2)
        # Reshape to (batch * gridsize * gridsize, N, N, 1)
        processed = processed.reshape(batch * gridsize**2, N, N, 1)
        # Update effective number of images for stitching calculation
        nimgs_effective = nimgs * gridsize**2
    else:
        # Ensure 4D with trailing 1
        if len(processed.shape) == 3:
            processed = processed[..., np.newaxis]
        nimgs_effective = nimgs

    # Calculate number of segments
    nsegments = int(np.sqrt((processed.size / nimgs_effective) / (N**2)))

    img_recon = np.reshape(
        norm_Y_I * processed, (-1, nsegments, nsegments, N, N, 1)
    )

    # Border clipping (from data_preprocessing.get_clip_sizes)
    bordersize = (N - outer_offset / 2) / 2
    borderleft = int(np.ceil(bordersize))
    borderright = int(np.floor(bordersize))

    img_recon = img_recon[:, :, :, borderleft:-borderright, borderleft:-borderright, :]
    tmp = img_recon.transpose(0, 1, 3, 2, 4, 5)
    stitched = tmp.reshape(-1, np.prod(tmp.shape[1:3]), np.prod(tmp.shape[1:3]), 1)
    return stitched


# ---------------------------------------------------------------------------
# Training + Inference Helpers (Task 5)
# ---------------------------------------------------------------------------


def train_pinn_model(train_data):
    """Train PtychoPINN model and return model + history."""
    from ptycho import train_pinn

    model, history = train_pinn.train(train_data)
    return model, history


def save_pinn_model(cfg: GridLinesConfig) -> None:
    """Save trained PINN model to output directory."""
    from ptycho import model_manager

    out_dir = cfg.output_dir / "pinn"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_manager.save(str(out_dir))


def select_baseline_channels(X, Y_I, Y_phi):
    """Select channel 0 only for baseline when gridsize > 1."""
    if X.shape[-1] > 1:
        return X[..., :1], Y_I[..., :1], Y_phi[..., :1]
    return X, Y_I, Y_phi


def train_baseline_model(X_train, Y_I_train, Y_phi_train):
    """Train baseline model (channel 0 only for gridsize > 1)."""
    from ptycho import baselines

    Xb, YIb, Yphib = select_baseline_channels(X_train, Y_I_train, Y_phi_train)
    model, history = baselines.train(Xb, YIb, Yphib)
    return model, history


def _history_loss_series(history: object) -> list[float]:
    if isinstance(history, dict):
        loss = history.get("loss", [])
        if isinstance(loss, list):
            return [float(value) for value in loss]
    raw_history = getattr(history, "history", None)
    if isinstance(raw_history, dict):
        loss = raw_history.get("loss", [])
        if isinstance(loss, list):
            return [float(value) for value in loss]
    return []


def _history_final_epoch(history: object, *, fallback_epochs: int) -> int:
    loss_series = _history_loss_series(history)
    if loss_series:
        return int(len(loss_series))
    epoch_series = getattr(history, "epoch", None)
    if isinstance(epoch_series, list) and epoch_series:
        return int(max(epoch_series) + 1)
    return int(fallback_epochs)


def _history_final_loss(history: object) -> float | None:
    loss_series = _history_loss_series(history)
    if loss_series:
        return float(loss_series[-1])
    return None


def _history_validation_loss(history: object) -> Dict[str, object]:
    if isinstance(history, dict):
        val_loss = history.get("val_loss", [])
        if isinstance(val_loss, list) and val_loss:
            return {"status": "emitted", "value": float(val_loss[-1])}
    raw_history = getattr(history, "history", None)
    if isinstance(raw_history, dict):
        val_loss = raw_history.get("val_loss", [])
        if isinstance(val_loss, list) and val_loss:
            return {"status": "emitted", "value": float(val_loss[-1])}
    return {"status": "no_validation_series", "value": None}


def _count_model_parameters(model: object) -> int | None:
    count_params = getattr(model, "count_params", None)
    if callable(count_params):
        try:
            return int(count_params())
        except Exception:
            return None
    return None


def _tf_hardware_summary() -> Dict[str, object]:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    accelerator = "cpu"
    if gpus:
        accelerator = getattr(gpus[0], "name", None) or "gpu"
    return {
        "backend": "tensorflow",
        "accelerator": accelerator,
    }


def _build_tf_row_payload(
    *,
    model_id: str,
    model_label: str,
    model: object,
    history: object,
    metrics: Dict[str, object],
    N: Optional[int],
    epoch_budget: int,
    train_wall_time_sec: float,
    inference_time_sec: float,
) -> Dict[str, object]:
    return {
        "model_label": model_label,
        "architecture_id": "cnn",
        "training_procedure": "supervised" if model_id == "baseline" else "pinn",
        "N": int(N) if N is not None else None,
        "parameter_count": _count_model_parameters(model),
        "epoch_budget": int(epoch_budget),
        "final_completed_epoch": _history_final_epoch(history, fallback_epochs=epoch_budget),
        "final_train_loss": _history_final_loss(history),
        "validation_loss": _history_validation_loss(history),
        "runtime_summary": {
            "train_wall_time_sec": float(train_wall_time_sec),
            "inference_time_sec": float(inference_time_sec),
        },
        "hardware_summary": _tf_hardware_summary(),
        "row_status": "paper_grade",
        "caveats": [],
        "metrics": dict(metrics),
    }


def _serialize_history_payload(history: object) -> Dict[str, object]:
    if isinstance(history, dict):
        return dict(history)
    payload: Dict[str, object] = {}
    history_dict = getattr(history, "history", None)
    if isinstance(history_dict, dict):
        payload.update(history_dict)
    epoch_series = getattr(history, "epoch", None)
    if isinstance(epoch_series, list):
        payload["epoch"] = list(epoch_series)
    return payload


def _build_tf_row_invocation_argv(
    *,
    cfg: GridLinesConfig,
    model_id: str,
    train_npz: Path,
    test_npz: Path,
) -> list[str]:
    argv = [
        "--model-id",
        model_id,
        "--N",
        str(cfg.N),
        "--gridsize",
        str(cfg.gridsize),
        "--output-dir",
        str(cfg.output_dir),
        "--probe-npz",
        str(cfg.probe_npz),
        "--train-npz",
        str(train_npz),
        "--test-npz",
        str(test_npz),
        "--nimgs-train",
        str(cfg.nimgs_train),
        "--nimgs-test",
        str(cfg.nimgs_test),
        "--nphotons",
        str(cfg.nphotons),
        "--nepochs",
        str(cfg.nepochs),
        "--batch-size",
        str(cfg.batch_size),
        "--probe-source",
        str(cfg.probe_source),
        "--probe-scale-mode",
        str(cfg.probe_scale_mode),
        "--probe-smoothing-sigma",
        str(cfg.probe_smoothing_sigma),
    ]
    if cfg.seed is not None:
        argv.extend(["--seed", str(cfg.seed)])
    return argv


def _apply_execution_seed(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(int(seed))
    np.random.seed(int(seed))
    try:
        import tensorflow as tf
    except Exception:
        return
    tf.keras.utils.set_random_seed(int(seed))


def _write_tf_row_provenance(
    *,
    cfg: GridLinesConfig,
    model_id: str,
    row_payload: Dict[str, object],
    history: object,
    train_npz: Path,
    test_npz: Path,
    model_artifact: Path,
    recon_path: Path,
) -> None:
    from scripts.studies.invocation_logging import (
        capture_runtime_provenance,
        get_git_commit,
        update_invocation_artifacts,
        write_invocation_artifacts,
    )

    run_dir = cfg.output_dir / "runs" / model_id
    run_dir.mkdir(parents=True, exist_ok=True)
    invocation_extra = {
        "runtime_provenance": capture_runtime_provenance(),
        "git_commit": get_git_commit(REPO_ROOT),
        "invocation_mode": "library",
        "row_model_id": model_id,
        "shared_root_output_dir": str(cfg.output_dir),
    }
    invocation_json, invocation_sh = write_invocation_artifacts(
        output_dir=run_dir,
        script_path="ptycho/workflows/grid_lines_workflow.py",
        argv=_build_tf_row_invocation_argv(
            cfg=cfg,
            model_id=model_id,
            train_npz=train_npz,
            test_npz=test_npz,
        ),
        parsed_args={
            "grid_lines_config": asdict(cfg),
            "row_model_id": model_id,
            "train_npz": str(train_npz),
            "test_npz": str(test_npz),
        },
        extra=invocation_extra,
    )
    update_invocation_artifacts(
        invocation_json,
        status="completed",
        exit_code=0,
        finished_at_utc=datetime.now(timezone.utc).isoformat(),
        run_dir=str(run_dir),
    )
    config_payload = {
        "grid_lines_config": asdict(cfg),
        "row_model_id": model_id,
        "model_label": row_payload.get("model_label"),
        "training_procedure": row_payload.get("training_procedure"),
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "model_artifact": str(model_artifact),
        "recon_npz": str(recon_path),
    }
    (run_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2, default=_json_default),
        encoding="utf-8",
    )
    (run_dir / "history.json").write_text(
        json.dumps(_serialize_history_payload(history), indent=2, default=_json_default),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(row_payload.get("metrics", {}), indent=2, default=_json_default),
        encoding="utf-8",
    )

    row_payload["invocation"] = {
        "json": str(invocation_json.relative_to(cfg.output_dir)),
        "shell": str(invocation_sh.relative_to(cfg.output_dir)),
    }
    row_payload["config"] = {
        "json": str((run_dir / "config.json").relative_to(cfg.output_dir)),
    }
    row_payload["git"] = {
        "commit": invocation_extra["git_commit"],
    }
    row_payload["environment"] = dict(invocation_extra["runtime_provenance"])
    row_payload["dataset"] = {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "probe_npz": str(cfg.probe_npz),
        "probe_source": str(cfg.probe_source),
        "probe_scale_mode": str(cfg.probe_scale_mode),
    }
    row_payload["splits"] = {
        "nimgs_train": int(cfg.nimgs_train),
        "nimgs_test": int(cfg.nimgs_test),
        "gridsize": int(cfg.gridsize),
        "set_phi": bool(cfg.set_phi),
        "seed": int(cfg.seed) if cfg.seed is not None else None,
    }
    randomness_payload: Dict[str, object] = {"seed_policy": "shared_wrapper_seed_contract"}
    if cfg.seed is not None:
        randomness_payload["seed"] = int(cfg.seed)
        randomness_payload["requested_seed"] = int(cfg.seed)
    row_payload["randomness"] = randomness_payload
    row_payload["outputs"] = {
        "metrics_json": str((run_dir / "metrics.json").relative_to(cfg.output_dir)),
        "history_json": str((run_dir / "history.json").relative_to(cfg.output_dir)),
        "recon_npz": str(recon_path.relative_to(cfg.output_dir)),
        "stdout_log": str((run_dir / "stdout.log").relative_to(cfg.output_dir)),
        "stderr_log": str((run_dir / "stderr.log").relative_to(cfg.output_dir)),
        "model_artifact": str(model_artifact.relative_to(cfg.output_dir)),
    }
    row_payload["visuals"] = {
        "amp_phase_png": f"visuals/amp_phase_{model_id}.png",
        "amp_phase_error_png": f"visuals/amp_phase_error_{model_id}.png",
    }


def run_pinn_inference(model, X_test, coords_nominal):
    """Run PINN inference on test data.

    Returns the reconstructed complex object (first output of model.predict).
    Native inference failures propagate to the caller.
    """
    intensity_scale = p.get("intensity_scale")
    prediction = model.predict([X_test * intensity_scale, coords_nominal])
    if prediction is None:
        raise ValueError("PINN inference returned no outputs")
    if isinstance(prediction, (list, tuple)):
        if not prediction:
            raise ValueError("PINN inference returned no outputs")
        return prediction[0]
    return prediction


def run_baseline_inference(model, X_test):
    """Run baseline inference on test data (channel 0 only)."""
    Xb, _, _ = select_baseline_channels(X_test, X_test, X_test)
    pred_amp, pred_phase = model.predict(Xb)
    pred_complex = pred_amp * np.exp(1j * pred_phase)
    return pred_complex


# ---------------------------------------------------------------------------
# Orchestrator + Outputs (Task 6)
# ---------------------------------------------------------------------------


def save_comparison_png(
    cfg: GridLinesConfig,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    pinn_amp: np.ndarray,
    pinn_phase: np.ndarray,
    base_amp: np.ndarray,
    base_phase: np.ndarray,
) -> Path:
    """Save 2x3 comparison plot (amp/phase rows x GT/PINN/Baseline cols)."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 0: Amplitude
    axes[0, 0].imshow(gt_amp, cmap="viridis")
    axes[0, 0].set_title("GT Amplitude")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(pinn_amp, cmap="viridis")
    axes[0, 1].set_title("PINN Amplitude")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(base_amp, cmap="viridis")
    axes[0, 2].set_title("Baseline Amplitude")
    axes[0, 2].axis("off")

    # Row 1: Phase
    axes[1, 0].imshow(gt_phase, cmap="twilight")
    axes[1, 0].set_title("GT Phase")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(pinn_phase, cmap="twilight")
    axes[1, 1].set_title("PINN Phase")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(base_phase, cmap="twilight")
    axes[1, 2].set_title("Baseline Phase")
    axes[1, 2].axis("off")

    fig.suptitle(f"N={cfg.N}, gridsize={cfg.gridsize}", fontsize=14)
    plt.tight_layout()

    visuals_dir = cfg.output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    out_path = visuals_dir / "compare_amp_phase.png"
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


_LABEL_TITLES = {
    "pinn": "PINN",
    "baseline": "Baseline",
    "pinn_fno": "FNO",
    "gt": "GT",
}


def _infer_patch_size_from_output_dir(output_dir: Path) -> Optional[int]:
    """Infer N from path tokens like 'n64'/'n128' in study output directories."""
    path_text = str(output_dir).lower()
    matches = re.findall(r"(?:^|[_\-/])n(\d{2,4})(?:[_\-/]|$)", path_text)
    if not matches:
        return None
    try:
        return int(matches[-1])
    except ValueError:
        return None


def _resolve_display_border_pixels(output_dir: Path, override: Optional[int] = None) -> int:
    """Resolve border size used for color-limit estimation."""
    if override is not None:
        return max(0, int(override))
    inferred_n = _infer_patch_size_from_output_dir(output_dir)
    if inferred_n is None:
        try:
            inferred_n = int(p.get("N"))
        except Exception:
            inferred_n = 0
    # Keep the auto-crop conservative; N//2 can over-tighten color limits.
    # Use a small fraction of N, capped to avoid aggressive clipping on large N.
    border = max(0, int(inferred_n) // 16)
    return min(border, 32)


def _inner_crop_for_display_bounds(array: np.ndarray, border_pixels: int) -> np.ndarray:
    """Crop outer border when computing display bounds; fallback to full array if too small."""
    arr = np.asarray(array)
    if arr.ndim != 2 or border_pixels <= 0:
        return arr
    h, w = arr.shape
    border = min(int(border_pixels), (h - 1) // 2, (w - 1) // 2)
    if border <= 0:
        return arr
    return arr[border:h - border, border:w - border]


def _safe_min_max(array: np.ndarray) -> Tuple[float, float] | None:
    if array is None or array.size == 0:
        return None
    if not np.any(np.isfinite(array)):
        return None
    vmin = np.nanmin(array)
    vmax = np.nanmax(array)
    if np.isnan(vmin) or np.isnan(vmax):
        return None
    return float(vmin), float(vmax)


def _display_bounds(array: np.ndarray, border_pixels: int = 0) -> Tuple[float, float] | None:
    """Compute min/max for plotting, optionally excluding an outer artifact band."""
    cropped = _inner_crop_for_display_bounds(array, border_pixels=border_pixels)
    bounds = _safe_min_max(cropped)
    if bounds is None and border_pixels > 0:
        return _safe_min_max(array)
    return bounds


def _resolve_probe_for_visuals(output_dir: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load probe amplitude/phase from a dataset NPZ for compare visualizations."""
    def _candidate_paths_from_run_params(run_params_path: Path) -> list[Path]:
        try:
            payload = json.loads(run_params_path.read_text())
        except Exception:
            return []

        resolved: list[Path] = []
        for key in ("train_npz", "test_npz"):
            raw_value = payload.get(key)
            if not isinstance(raw_value, str) or not raw_value.strip():
                continue
            raw_path = Path(raw_value)
            candidate = raw_path if raw_path.is_absolute() else (output_dir / raw_path)
            if candidate.exists():
                resolved.append(candidate)
            elif raw_path.exists():
                resolved.append(raw_path)
        return resolved

    candidates: list[Path] = []
    dataset_root = output_dir / "datasets"
    if dataset_root.exists():
        candidates.extend(
            sorted(dataset_root.glob("N*/gs*/simulation-*/train.npz"))
        )
        candidates.extend(
            sorted(dataset_root.glob("N*/gs*/simulation-*/test.npz"))
        )
        candidates.extend(sorted(dataset_root.glob("N*/gs*/train.npz")))
        candidates.extend(sorted(dataset_root.glob("N*/gs*/test.npz")))

    for params_name in ("run_params.json", "runparams.json"):
        params_path = output_dir / params_name
        if params_path.exists():
            candidates.extend(_candidate_paths_from_run_params(params_path))

    if not candidates:
        return None

    seen = set()
    for npz_path in candidates:
        npz_key = str(npz_path)
        if npz_key in seen:
            continue
        seen.add(npz_key)
        try:
            with np.load(npz_path) as data:
                probe = None
                for key in ("probeGuess", "probe", "probe_guess"):
                    if key in data:
                        probe = np.asarray(data[key])
                        break
                if probe is None:
                    continue
        except Exception:
            continue

        probe = np.squeeze(probe)
        if probe.ndim > 2:
            probe = probe[0]
        if probe.ndim != 2:
            continue
        probe = np.asarray(probe, dtype=np.complex64)
        return {
            "amp": np.abs(probe),
            "phase": np.angle(probe),
        }
    return None


def _imshow_scaled_probe(
    ax,
    image: np.ndarray,
    *,
    cmap: str,
    object_side: float,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
):
    """Render probe image centered and scaled to object-side units."""
    probe_h, probe_w = image.shape
    x0 = (object_side - probe_w) / 2.0
    y0 = (object_side - probe_h) / 2.0
    x1 = x0 + probe_w
    y1 = y0 + probe_h

    kwargs: Dict[str, float] = {}
    if vmin is not None:
        kwargs["vmin"] = vmin
    if vmax is not None:
        kwargs["vmax"] = vmax

    mappable = ax.imshow(
        image,
        cmap=cmap,
        extent=(x0, x1, y0, y1),
        origin="upper",
        **kwargs,
    )
    ax.set_xlim(0.0, object_side)
    ax.set_ylim(object_side, 0.0)
    ax.set_aspect("equal")
    return mappable


def _pixel_ticks_for_width(width_pixels: int, max_ticks: int = 6) -> np.ndarray:
    width = max(1, int(width_pixels))
    if width == 1:
        return np.array([0], dtype=int)
    count = max(2, min(max_ticks, width))
    ticks = np.linspace(0, width - 1, num=count)
    return np.unique(np.round(ticks).astype(int))


def _apply_x_pixel_axis(ax, width_pixels: int) -> None:
    """Show numeric x-axis tick labels to indicate pixel dimensions."""
    ticks = _pixel_ticks_for_width(width_pixels)
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks])
    ax.set_yticks([])
    ax.tick_params(axis="x", labelsize=8)


def _should_share_colorbar(
    arrays: Tuple[np.ndarray, ...] | list[np.ndarray],
    rtol: float = 1e-6,
    atol: float = 1e-8,
) -> bool:
    ranges = []
    for arr in arrays:
        bounds = _safe_min_max(arr)
        if bounds is None:
            return False
        ranges.append(bounds)
    if len(ranges) <= 1:
        return True
    first_min, first_max = ranges[0]
    for vmin, vmax in ranges[1:]:
        if not (np.isclose(vmin, first_min, rtol=rtol, atol=atol) and
                np.isclose(vmax, first_max, rtol=rtol, atol=atol)):
            return False
    return True


def _add_row_colorbars(fig, axes_row, mappables, arrays) -> None:
    share = _should_share_colorbar(arrays)
    if share:
        fig.colorbar(mappables[0], ax=list(axes_row), shrink=0.8)
    else:
        for ax, mappable in zip(axes_row, mappables):
            fig.colorbar(mappable, ax=ax, shrink=0.8)


def save_recon_artifact(output_dir: Path, label: str, recon_complex: np.ndarray) -> Path:
    """Save stitched complex reconstruction as NPZ artifact."""
    recon_dir = output_dir / "recons" / label
    recon_dir.mkdir(parents=True, exist_ok=True)
    recon = np.squeeze(recon_complex)
    if recon.ndim > 2:
        recon = recon[0]
    recon = recon.astype(np.complex64)
    amp = np.abs(recon)
    phase = np.angle(recon)
    path = recon_dir / "recon.npz"
    np.savez(path, YY_pred=recon, amp=amp, phase=phase)
    return path


def save_comparison_png_dynamic(
    output_dir: Path,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    recons: Dict[str, Dict[str, np.ndarray]],
    order: Tuple[str, ...],
    border_pixels: Optional[int] = None,
    probe: Optional[Dict[str, np.ndarray]] = None,
    amp_bounds: Tuple[float, float] | None = None,
    phase_bounds: Tuple[float, float] | None = None,
) -> Path:
    """Save comparison plot with GT plus available model reconstructions."""
    import matplotlib.pyplot as plt

    labels = [label for label in order if label in recons]
    include_probe = (
        probe is not None
        and isinstance(probe, dict)
        and "amp" in probe
        and "phase" in probe
    )
    ncols = 1 + len(labels) + (1 if include_probe else 0)
    fig, axes = plt.subplots(2, ncols, figsize=(5 * ncols, 8), squeeze=False)
    resolved_border = _resolve_display_border_pixels(output_dir, override=border_pixels)

    amp_arrays = [gt_amp]
    phase_arrays = [gt_phase]
    amp_gt_bounds = amp_bounds or _display_bounds(gt_amp, border_pixels=resolved_border)
    phase_gt_bounds = phase_bounds or _display_bounds(gt_phase, border_pixels=resolved_border)

    amp_gt_kwargs = {}
    if amp_gt_bounds is not None:
        amp_gt_kwargs = {"vmin": amp_gt_bounds[0], "vmax": amp_gt_bounds[1]}
    amp_mappables = [axes[0, 0].imshow(gt_amp, cmap="viridis", **amp_gt_kwargs)]
    axes[0, 0].set_title("GT Amplitude")
    _apply_x_pixel_axis(axes[0, 0], gt_amp.shape[1])

    phase_gt_kwargs = {}
    if phase_gt_bounds is not None:
        phase_gt_kwargs = {"vmin": phase_gt_bounds[0], "vmax": phase_gt_bounds[1]}
    phase_mappables = [axes[1, 0].imshow(gt_phase, cmap="twilight", **phase_gt_kwargs)]
    axes[1, 0].set_title("GT Phase")
    _apply_x_pixel_axis(axes[1, 0], gt_phase.shape[1])

    for idx, label in enumerate(labels, start=1):
        amp = recons[label]["amp"]
        phase = recons[label]["phase"]
        title = _LABEL_TITLES.get(label, label)
        amp_label_bounds = amp_bounds or _display_bounds(amp, border_pixels=resolved_border)
        phase_label_bounds = phase_bounds or _display_bounds(phase, border_pixels=resolved_border)
        amp_kwargs = {}
        if amp_label_bounds is not None:
            amp_kwargs = {"vmin": amp_label_bounds[0], "vmax": amp_label_bounds[1]}
        phase_kwargs = {}
        if phase_label_bounds is not None:
            phase_kwargs = {"vmin": phase_label_bounds[0], "vmax": phase_label_bounds[1]}

        amp_arrays.append(amp)
        phase_arrays.append(phase)

        amp_mappables.append(
            axes[0, idx].imshow(amp, cmap="viridis", **amp_kwargs)
        )
        axes[0, idx].set_title(f"{title} Amplitude")
        _apply_x_pixel_axis(axes[0, idx], amp.shape[1])

        phase_mappables.append(
            axes[1, idx].imshow(phase, cmap="twilight", **phase_kwargs)
        )
        axes[1, idx].set_title(f"{title} Phase")
        _apply_x_pixel_axis(axes[1, idx], phase.shape[1])

    if include_probe:
        probe_idx = ncols - 1
        probe_amp = np.asarray(probe["amp"])
        probe_phase = np.asarray(probe["phase"])
        object_side = float(max(gt_amp.shape))

        probe_amp_bounds = _display_bounds(probe_amp, border_pixels=0)
        probe_phase_bounds = _display_bounds(probe_phase, border_pixels=0)

        amp_mappables.append(
            _imshow_scaled_probe(
                axes[0, probe_idx],
                probe_amp,
                cmap="viridis",
                object_side=object_side,
                vmin=(probe_amp_bounds[0] if probe_amp_bounds is not None else None),
                vmax=(probe_amp_bounds[1] if probe_amp_bounds is not None else None),
            )
        )
        axes[0, probe_idx].set_title(f"Probe Amplitude ({probe_amp.shape[0]}x{probe_amp.shape[1]})")
        _apply_x_pixel_axis(axes[0, probe_idx], int(round(object_side)))
        amp_arrays.append(probe_amp)

        phase_mappables.append(
            _imshow_scaled_probe(
                axes[1, probe_idx],
                probe_phase,
                cmap="twilight",
                object_side=object_side,
                vmin=(probe_phase_bounds[0] if probe_phase_bounds is not None else None),
                vmax=(probe_phase_bounds[1] if probe_phase_bounds is not None else None),
            )
        )
        axes[1, probe_idx].set_title(f"Probe Phase ({probe_phase.shape[0]}x{probe_phase.shape[1]})")
        _apply_x_pixel_axis(axes[1, probe_idx], int(round(object_side)))
        phase_arrays.append(probe_phase)

    _add_row_colorbars(fig, axes[0], amp_mappables, amp_arrays)
    _add_row_colorbars(fig, axes[1], phase_mappables, phase_arrays)

    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    out_path = visuals_dir / "compare_amp_phase.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_amp_phase_png(
    visuals_dir: Path,
    label: str,
    amp: np.ndarray,
    phase: np.ndarray,
    border_pixels: int = 0,
    amp_bounds: Tuple[float, float] | None = None,
    phase_bounds: Tuple[float, float] | None = None,
) -> Path:
    """Save per-model amplitude/phase visualization."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(6, 8), squeeze=False)
    title = _LABEL_TITLES.get(label, label)
    amp_bounds = amp_bounds or _display_bounds(amp, border_pixels=border_pixels)
    phase_bounds = phase_bounds or _display_bounds(phase, border_pixels=border_pixels)
    amp_kwargs = {}
    if amp_bounds is not None:
        amp_kwargs = {"vmin": amp_bounds[0], "vmax": amp_bounds[1]}
    phase_kwargs = {}
    if phase_bounds is not None:
        phase_kwargs = {"vmin": phase_bounds[0], "vmax": phase_bounds[1]}

    amp_mappable = axes[0, 0].imshow(amp, cmap="viridis", **amp_kwargs)
    axes[0, 0].set_title(f"{title} Amplitude")
    axes[0, 0].axis("off")

    phase_mappable = axes[1, 0].imshow(phase, cmap="twilight", **phase_kwargs)
    axes[1, 0].set_title(f"{title} Phase")
    axes[1, 0].axis("off")

    _add_row_colorbars(fig, axes[0], [amp_mappable], [amp])
    _add_row_colorbars(fig, axes[1], [phase_mappable], [phase])

    out_path = visuals_dir / f"amp_phase_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _shared_display_bounds(
    arrays: Iterable[np.ndarray],
    *,
    border_pixels: int,
) -> Tuple[float, float] | None:
    bounds = [
        _display_bounds(np.asarray(array), border_pixels=border_pixels)
        for array in arrays
    ]
    finite_bounds = [item for item in bounds if item is not None]
    if not finite_bounds:
        return None
    return (
        min(item[0] for item in finite_bounds),
        max(item[1] for item in finite_bounds),
    )


def save_amp_phase_error_png(
    visuals_dir: Path,
    label: str,
    amp: np.ndarray,
    phase: np.ndarray,
    gt_amp: np.ndarray,
    gt_phase: np.ndarray,
    *,
    amp_bounds: Tuple[float, float] | None,
    phase_bounds: Tuple[float, float] | None,
    amp_error_bounds: Tuple[float, float] | None,
    phase_error_bounds: Tuple[float, float] | None,
) -> Path:
    import matplotlib.pyplot as plt

    title = _LABEL_TITLES.get(label, label)
    amp_abs_error = np.abs(np.asarray(amp) - np.asarray(gt_amp))
    phase_abs_error = np.abs(np.asarray(phase) - np.asarray(gt_phase))

    fig, axes = plt.subplots(2, 2, figsize=(10, 8), squeeze=False)
    amp_kwargs = {}
    if amp_bounds is not None:
        amp_kwargs = {"vmin": amp_bounds[0], "vmax": amp_bounds[1]}
    phase_kwargs = {}
    if phase_bounds is not None:
        phase_kwargs = {"vmin": phase_bounds[0], "vmax": phase_bounds[1]}
    amp_error_kwargs = {}
    if amp_error_bounds is not None:
        amp_error_kwargs = {"vmin": amp_error_bounds[0], "vmax": amp_error_bounds[1]}
    phase_error_kwargs = {}
    if phase_error_bounds is not None:
        phase_error_kwargs = {"vmin": phase_error_bounds[0], "vmax": phase_error_bounds[1]}

    amp_mappable = axes[0, 0].imshow(amp, cmap="viridis", **amp_kwargs)
    axes[0, 0].set_title(f"{title} Amplitude")
    axes[0, 0].axis("off")
    amp_err_mappable = axes[0, 1].imshow(amp_abs_error, cmap="magma", **amp_error_kwargs)
    axes[0, 1].set_title(f"{title} |Amp Error|")
    axes[0, 1].axis("off")
    phase_mappable = axes[1, 0].imshow(phase, cmap="twilight", **phase_kwargs)
    axes[1, 0].set_title(f"{title} Phase")
    axes[1, 0].axis("off")
    phase_err_mappable = axes[1, 1].imshow(phase_abs_error, cmap="magma", **phase_error_kwargs)
    axes[1, 1].set_title(f"{title} |Phase Error|")
    axes[1, 1].axis("off")

    fig.colorbar(amp_mappable, ax=axes[0, 0], shrink=0.8)
    fig.colorbar(amp_err_mappable, ax=axes[0, 1], shrink=0.8)
    fig.colorbar(phase_mappable, ax=axes[1, 0], shrink=0.8)
    fig.colorbar(phase_err_mappable, ax=axes[1, 1], shrink=0.8)

    out_path = visuals_dir / f"amp_phase_error_{label}.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def save_frc_curves_png(
    visuals_dir: Path,
    frc_by_label: Dict[str, Tuple[np.ndarray, np.ndarray]],
) -> Path:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), squeeze=False)
    amp_ax = axes[0, 0]
    phase_ax = axes[0, 1]
    for label, (amp_curve, phase_curve) in frc_by_label.items():
        title = _LABEL_TITLES.get(label, label)
        amp_x = np.linspace(0.0, 1.0, num=len(amp_curve))
        phase_x = np.linspace(0.0, 1.0, num=len(phase_curve))
        amp_ax.plot(amp_x, amp_curve, label=title)
        phase_ax.plot(phase_x, phase_curve, label=title)
    for ax, ax_title in ((amp_ax, "Amplitude FRC"), (phase_ax, "Phase FRC")):
        ax.axhline(0.5, color="black", linestyle="--", linewidth=1.0)
        ax.axhline(1.0 / 7.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(ax_title)
        ax.set_xlabel("Normalized Spatial Frequency")
        ax.set_ylabel("Correlation")
        ax.set_ylim(0.0, 1.05)
        ax.legend()
    out_path = visuals_dir / "frc_curves.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def render_grid_lines_visuals(output_dir: Path, order: Tuple[str, ...]) -> Dict[str, str]:
    """Render composite and per-model visuals from recon artifacts."""
    visuals_dir = output_dir / "visuals"
    visuals_dir.mkdir(parents=True, exist_ok=True)
    resolved_border = _resolve_display_border_pixels(output_dir)

    recons: Dict[str, Dict[str, np.ndarray]] = {}
    per_model_paths: Dict[str, str] = {}
    for label in order:
        recon_path = output_dir / "recons" / label / "recon.npz"
        if not recon_path.exists():
            continue
        with np.load(recon_path) as data:
            if "amp" not in data or "phase" not in data:
                continue
            amp = data["amp"]
            phase = data["phase"]
        recons[label] = {"amp": amp, "phase": phase}
        label_border = 0 if label == "gt" else resolved_border
        per_model_paths[label] = str(
            save_amp_phase_png(visuals_dir, label, amp, phase, border_pixels=label_border)
        )

    outputs: Dict[str, str] = {}
    amp_bounds = _shared_display_bounds(
        [data["amp"] for data in recons.values()],
        border_pixels=resolved_border,
    )
    phase_bounds = _shared_display_bounds(
        [data["phase"] for data in recons.values()],
        border_pixels=resolved_border,
    )
    for label, path in per_model_paths.items():
        outputs[f"amp_phase_{label}"] = path

    gt = recons.get("gt")
    if gt is None:
        return outputs

    for label, data in recons.items():
        label_border = 0 if label == "gt" else resolved_border
        per_model_paths[label] = str(
            save_amp_phase_png(
                visuals_dir,
                label,
                data["amp"],
                data["phase"],
                border_pixels=label_border,
                amp_bounds=amp_bounds,
                phase_bounds=phase_bounds,
            )
        )
        outputs[f"amp_phase_{label}"] = per_model_paths[label]

    error_labels = [label for label in order if label in recons and label != "gt"]
    amp_error_bounds = _shared_display_bounds(
        [np.abs(recons[label]["amp"] - gt["amp"]) for label in error_labels],
        border_pixels=resolved_border,
    )
    phase_error_bounds = _shared_display_bounds(
        [np.abs(recons[label]["phase"] - gt["phase"]) for label in error_labels],
        border_pixels=resolved_border,
    )
    for label in error_labels:
        outputs[f"amp_phase_error_{label}"] = str(
            save_amp_phase_error_png(
                visuals_dir,
                label,
                recons[label]["amp"],
                recons[label]["phase"],
                gt["amp"],
                gt["phase"],
                amp_bounds=amp_bounds,
                phase_bounds=phase_bounds,
                amp_error_bounds=amp_error_bounds,
                phase_error_bounds=phase_error_bounds,
            )
        )

    compare = save_comparison_png_dynamic(
        output_dir,
        gt["amp"],
        gt["phase"],
        {label: data for label, data in recons.items() if label != "gt"},
        order=tuple(label for label in order if label != "gt"),
        border_pixels=resolved_border,
        probe=_resolve_probe_for_visuals(output_dir),
        amp_bounds=amp_bounds,
        phase_bounds=phase_bounds,
    )
    outputs["compare"] = str(compare)

    metrics_path = output_dir / "metrics.json"
    if metrics_path.exists():
        with metrics_path.open("r", encoding="utf-8") as handle:
            metrics_payload = json.load(handle)
        frc_by_label: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        if isinstance(metrics_payload, dict):
            for label in error_labels:
                payload = metrics_payload.get(label)
                if not isinstance(payload, dict):
                    continue
                frc_payload = payload.get("frc")
                if (
                    isinstance(frc_payload, (list, tuple))
                    and len(frc_payload) == 2
                ):
                    amp_curve = np.asarray(frc_payload[0], dtype=np.float64)
                    phase_curve = np.asarray(frc_payload[1], dtype=np.float64)
                    if amp_curve.ndim == 1 and phase_curve.ndim == 1:
                        frc_by_label[label] = (amp_curve, phase_curve)
        if frc_by_label:
            outputs["frc_curves"] = str(save_frc_curves_png(visuals_dir, frc_by_label))
    return outputs


@scoped_legacy_params
def run_grid_lines_workflow(
    cfg: GridLinesConfig,
    tf_models: Tuple[str, ...] = ("pinn", "baseline"),
) -> Dict[str, Any]:
    """Orchestrate probe prep → sim → train → infer → stitch → metrics.

    Steps:
    1. Load and scale probe to target N
    2. Configure legacy params and simulate grid data
    3. Save train/test datasets as NPZ
    4. Train selected TF models
    5. Run selected inference paths on test data
    6. Stitch selected predictions and compute metrics
    7. Save comparison PNG and metrics JSON

    Returns:
        Dict with train_npz, test_npz, metrics paths and values.
    """
    from ptycho.evaluation import eval_reconstruction

    selected_models = tuple(tf_models)
    if not selected_models:
        raise ValueError("tf_models must include at least one of {'pinn', 'baseline'}")
    unsupported_models = sorted(set(selected_models) - {"pinn", "baseline"})
    if unsupported_models:
        raise ValueError(f"Unsupported tf_models entries: {unsupported_models}")
    # Preserve user order while removing accidental duplicates.
    deduped = []
    seen_models = set()
    for model_id in selected_models:
        if model_id not in seen_models:
            deduped.append(model_id)
            seen_models.add(model_id)
    selected_models = tuple(deduped)

    print(f"[grid_lines_workflow] Starting N={cfg.N}, gridsize={cfg.gridsize}")
    _apply_execution_seed(cfg.seed)

    # Step 1: Probe preparation
    print("[1/7] Loading and scaling probe...")
    probe_guess = _load_configured_probe(cfg)
    normalized_pipeline, normalized_steps = normalize_probe_transform_pipeline(
        target_N=cfg.N,
        probe_shape=probe_guess.shape,
        probe_scale_mode=cfg.probe_scale_mode,
        probe_smoothing_sigma=cfg.probe_smoothing_sigma,
        probe_transform_pipeline=cfg.probe_transform_pipeline,
    )
    transform_result = apply_probe_transform_pipeline_with_metadata(
        probe_guess, normalized_steps
    )
    probe_scaled = transform_result.probe
    probe_scaled = apply_probe_mask(probe_scaled, cfg.probe_mask_diameter)
    probe_lineage = _build_probe_lineage(
        cfg,
        raw_probe=probe_guess,
        normalized_pipeline=normalized_pipeline,
        transformed_probe=probe_scaled,
        transform_metadata=transform_result.metadata,
    )
    _preflight_dataset_pair_reuse(cfg, probe_lineage)

    # Step 2: Simulation
    print("[2/7] Running grid simulation...")
    sim = simulate_grid_data(cfg, probe_scaled)
    config = configure_legacy_params(cfg, probe_scaled)

    # Step 3: Save datasets
    print("[3/7] Saving datasets...")
    sim["train"]["probeGuess"] = probe_scaled
    sim["test"]["probeGuess"] = probe_scaled
    train_npz = save_split_npz(
        cfg,
        "train",
        sim["train"],
        config,
        probe_transform_pipeline=normalized_pipeline,
        probe_transform_steps=normalized_steps,
        probe_lineage=probe_lineage,
    )
    test_npz = save_split_npz(
        cfg,
        "test",
        sim["test"],
        config,
        probe_transform_pipeline=normalized_pipeline,
        probe_transform_steps=normalized_steps,
        probe_lineage=probe_lineage,
    )

    # Step 4-6: Execute row-local train -> infer -> evaluate segments.
    print(f"[4/7] Executing selected TF rows with row-local provenance: {selected_models}...")
    norm_Y_I = sim["test"]["norm_Y_I"]
    YY_gt = sim["test"]["YY_ground_truth"]
    metrics_payload: Dict[str, Any] = {}
    recons: Dict[str, Dict[str, np.ndarray]] = {}
    row_payloads: Dict[str, Dict[str, object]] = {}
    if "pinn" in selected_models:
        with _capture_tf_row_logs(cfg.output_dir, "pinn"):
            _apply_execution_seed(cfg.seed)
            print("[4/7][row:pinn] Training PINN model...")
            pinn_train_start = time.perf_counter()
            pinn_model, pinn_history = train_pinn_model(sim["train"]["container"])
            pinn_train_time_s = time.perf_counter() - pinn_train_start
            save_pinn_model(cfg)

            print("[5/7][row:pinn] Running PINN inference...")
            pinn_infer_start = time.perf_counter()
            pinn_pred = run_pinn_inference(
                pinn_model, sim["test"]["X"], sim["test"]["coords_nominal"]
            )
            pinn_inference_time_s = time.perf_counter() - pinn_infer_start
            print("[6/7][row:pinn] Stitching and computing metrics...")
            pinn_amp = stitch_predictions(pinn_pred, norm_Y_I, part="amp")
            pinn_phase = stitch_predictions(pinn_pred, norm_Y_I, part="phase")
            pinn_stitched = pinn_amp * np.exp(1j * pinn_phase)
            metrics_payload["pinn"] = eval_reconstruction(
                pinn_stitched,
                YY_gt,
                label="pinn",
            )
            row_payloads["pinn"] = _build_tf_row_payload(
                model_id="pinn",
                model_label="CDI CNN + PINN",
                model=pinn_model,
                history=pinn_history,
                metrics=metrics_payload["pinn"],
                N=cfg.N,
                epoch_budget=int(cfg.nepochs),
                train_wall_time_sec=float(pinn_train_time_s),
                inference_time_sec=float(pinn_inference_time_s),
            )
            recons["pinn"] = {
                "amp": pinn_amp[0, :, :, 0],
                "phase": pinn_phase[0, :, :, 0],
            }
            pinn_recon_path = save_recon_artifact(cfg.output_dir, "pinn", pinn_stitched)
            if train_npz is not None and test_npz is not None:
                _write_tf_row_provenance(
                    cfg=cfg,
                    model_id="pinn",
                    row_payload=row_payloads["pinn"],
                    history=pinn_history,
                    train_npz=Path(train_npz),
                    test_npz=Path(test_npz),
                    model_artifact=cfg.output_dir / "pinn" / "wts.h5.zip",
                    recon_path=pinn_recon_path,
                )

    if "baseline" in selected_models:
        with _capture_tf_row_logs(cfg.output_dir, "baseline"):
            _apply_execution_seed(cfg.seed)
            print("[4/7][row:baseline] Training baseline model...")
            base_train_start = time.perf_counter()
            base_model, base_history = train_baseline_model(
                sim["train"]["X"], sim["train"]["Y_I"], sim["train"]["Y_phi"]
            )
            base_train_time_s = time.perf_counter() - base_train_start
            base_dir = cfg.output_dir / "baseline"
            base_dir.mkdir(parents=True, exist_ok=True)
            base_model.save(base_dir / "baseline.keras")

            print("[5/7][row:baseline] Running baseline inference...")
            base_infer_start = time.perf_counter()
            base_pred = run_baseline_inference(base_model, sim["test"]["X"])
            base_inference_time_s = time.perf_counter() - base_infer_start

            print("[6/7][row:baseline] Stitching and computing metrics...")
            base_amp = stitch_predictions(base_pred, norm_Y_I, part="amp")
            base_phase = stitch_predictions(base_pred, norm_Y_I, part="phase")
            base_stitched = base_amp * np.exp(1j * base_phase)
            metrics_payload["baseline"] = eval_reconstruction(
                base_stitched,
                YY_gt,
                label="baseline",
            )
            row_payloads["baseline"] = _build_tf_row_payload(
                model_id="baseline",
                model_label="CDI CNN + supervised",
                model=base_model,
                history=base_history,
                metrics=metrics_payload["baseline"],
                N=cfg.N,
                epoch_budget=int(cfg.nepochs),
                train_wall_time_sec=float(base_train_time_s),
                inference_time_sec=float(base_inference_time_s),
            )
            recons["baseline"] = {
                "amp": base_amp[0, :, :, 0],
                "phase": base_phase[0, :, :, 0],
            }
            baseline_recon_path = save_recon_artifact(cfg.output_dir, "baseline", base_stitched)
            if train_npz is not None and test_npz is not None:
                _write_tf_row_provenance(
                    cfg=cfg,
                    model_id="baseline",
                    row_payload=row_payloads["baseline"],
                    history=base_history,
                    train_npz=Path(train_npz),
                    test_npz=Path(test_npz),
                    model_artifact=cfg.output_dir / "baseline" / "baseline.keras",
                    recon_path=baseline_recon_path,
                )

    # Step 7: Save outputs
    print("[7/7] Saving merged outputs...")

    metrics_path = cfg.output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics_payload, indent=2, default=str))

    # Comparison PNG - squeeze any singleton dims from GT.
    gt_squeezed = np.squeeze(YY_gt)
    gt_amp_2d = np.abs(gt_squeezed)
    gt_phase_2d = np.angle(gt_squeezed)

    save_recon_artifact(cfg.output_dir, "gt", gt_squeezed)

    png_path = save_comparison_png_dynamic(
        cfg.output_dir,
        gt_amp_2d,
        gt_phase_2d,
        recons,
        order=tuple(model_id for model_id in selected_models if model_id in {"pinn", "baseline"}),
    )

    print(f"[grid_lines_workflow] Complete. Outputs in {cfg.output_dir}")

    return {
        "train_npz": str(train_npz),
        "test_npz": str(test_npz),
        "metrics_json": str(metrics_path),
        "comparison_png": str(png_path),
        "metrics": metrics_payload,
        "row_payloads": row_payloads,
    }
