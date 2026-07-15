"""Recipe-pinned identity for the procedurally generated grid-lines reference.

The N=128/C=1 grid-lines reference dataset (the executable condition in
``tests/torch/test_grid_lines_hybrid_resnet_integration.py``) is generated
procedurally, so its immutable identity is the generation *recipe*: the pinned
Run1084 probe archive/array hashes, the deterministic pad-then-smooth probe
transform hash, and the closed set of grid-lines simulation parameters. NPZ
content is not bit-reproducible across environments (the integration fixture
tolerates dataset-statistic drift), so materialized train/test archives are
fingerprinted at execution time and recorded as sealed run evidence rather
than being pinned in the checked spec.

Facade: re-exported through :mod:`scripts.studies.ablation.datasets`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np

from ptycho.config import (
    DetectorSimulationConfig,
    ProbeSimulationConfig,
    ScanSimulationConfig,
    SimulationConfig,
    SyntheticObjectConfig,
    simulation_config_from_mapping,
    simulation_config_to_dict,
)

from .dataset_content import file_sha256
from .dataset_provenance import canonical_array_sha256
from .dataset_schema import DatasetError

__all__ = [
    "GRID_LINES_REFERENCE_GENERATOR",
    "LADDER_DATASET_EXPRESSIONS",
    "GridLinesReferenceRecipe",
    "LadderDatasetRecipe",
    "MaterializedReferenceDataset",
    "parse_grid_lines_reference_recipe",
    "parse_grid_lines_reference_simulation",
    "parse_ladder_dataset",
    "validate_ladder_npz_pair",
    "validate_reference_npz_pair",
]

#: The only supported generator for reference recipes; names the exact
#: procedural pipeline (``ptycho.workflows.grid_lines_workflow``).
GRID_LINES_REFERENCE_GENERATOR = "grid_lines_workflow_v1"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
_RECIPE_FIELDS = (
    "id",
    "generator",
    "probe_archive",
    "probe_archive_sha256",
    "raw_probe_array_sha256",
    "transformed_probe_sha256",
    "probe_scale_mode",
    "probe_smoothing_sigma",
    "set_phi",
    "N",
    "gridsize",
    "size",
    "offset",
    "outer_offset_train",
    "outer_offset_test",
    "nimgs_train",
    "nimgs_test",
    "nphotons",
)
_REQUIRED_SPLIT_KEYS = ("diffraction", "Y_I", "Y_phi", "coords_nominal", "probeGuess")


@dataclass(frozen=True)
class GridLinesReferenceRecipe:
    """Immutable, fingerprintable identity of the grid-lines reference data."""

    id: str
    generator: str
    probe_archive: Path
    probe_archive_declared: str
    probe_archive_sha256: str
    raw_probe_array_sha256: str
    transformed_probe_sha256: str
    probe_scale_mode: str
    probe_smoothing_sigma: float
    set_phi: bool
    N: int
    gridsize: int
    size: int
    offset: int
    outer_offset_train: int
    outer_offset_test: int
    nimgs_train: int
    nimgs_test: int
    nphotons: float
    schema_version: int = 1
    simulation: SimulationConfig | None = None

    @property
    def fingerprint_sha256(self) -> str:
        """Hash the declared recipe identity (path as declared, not resolved)."""
        if self.schema_version == 2:
            if self.simulation is None:
                raise RuntimeError("schema-v2 recipe is missing SimulationConfig")
            payload = {
                "id": self.id,
                "generator": self.generator,
                "probe_archive_sha256": self.probe_archive_sha256,
                "raw_probe_array_sha256": self.raw_probe_array_sha256,
                "transformed_probe_sha256": self.transformed_probe_sha256,
                "simulation": simulation_config_to_dict(self.simulation),
            }
            encoded = json.dumps(
                payload, sort_keys=True, separators=(",", ":"), allow_nan=False
            )
            return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        payload = {
            name: getattr(self, name)
            for name in _RECIPE_FIELDS
            if name != "probe_archive"
        }
        payload["probe_archive"] = self.probe_archive_declared
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class MaterializedReferenceDataset:
    """Content fingerprints of one materialized reference NPZ pair."""

    recipe_fingerprint_sha256: str
    train_path: Path
    test_path: Path
    train_sha256: str
    test_sha256: str
    probe_sha256: str
    n_train: int
    n_test: int


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise DatasetError(f"reference recipe {name} must be nonempty text")
    return value


def _sha(value: Any, name: str) -> str:
    text = _text(value, name)
    if not _SHA256_RE.fullmatch(text) or text == "0" * 64:
        raise DatasetError(
            f"reference recipe {name} must be a non-sentinel lowercase SHA-256"
        )
    return text


def _positive_int(value: Any, name: str) -> int:
    if type(value) is not int or value <= 0:
        raise DatasetError(f"reference recipe {name} must be a positive integer")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise DatasetError(f"reference recipe {name} must be a nonnegative integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise DatasetError(f"reference recipe {name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise DatasetError(f"reference recipe {name} must be finite")
    return number


def parse_grid_lines_reference_recipe(
    dataset_id: str,
    value: Mapping[str, Any],
    *,
    base_dir: Path,
) -> GridLinesReferenceRecipe:
    """Parse the closed recipe table; no artifact I/O happens here."""
    if not isinstance(value, Mapping):
        raise DatasetError("reference recipe must be a table")
    if set(value) != set(_RECIPE_FIELDS):
        missing = sorted(set(_RECIPE_FIELDS) - set(value))
        extra = sorted(set(value) - set(_RECIPE_FIELDS))
        raise DatasetError(
            f"reference recipe fields do not match the closed schema "
            f"(missing={missing}, unexpected={extra})"
        )
    declared_id = _text(value["id"], "id")
    if declared_id != dataset_id or not _ID_RE.fullmatch(declared_id):
        raise DatasetError(
            f"reference recipe id {declared_id!r} must match {dataset_id!r}"
        )
    generator = _text(value["generator"], "generator")
    if generator != GRID_LINES_REFERENCE_GENERATOR:
        raise DatasetError(
            f"reference recipe generator {generator!r} is not "
            f"{GRID_LINES_REFERENCE_GENERATOR!r}"
        )
    set_phi = value["set_phi"]
    if type(set_phi) is not bool:
        raise DatasetError("reference recipe set_phi must be boolean")
    declared_archive = _text(value["probe_archive"], "probe_archive")
    nphotons = _finite_float(value["nphotons"], "nphotons")
    if nphotons <= 0:
        raise DatasetError("reference recipe nphotons must be positive")
    sigma = _finite_float(value["probe_smoothing_sigma"], "probe_smoothing_sigma")
    if sigma < 0:
        raise DatasetError("reference recipe probe_smoothing_sigma must be >= 0")
    N = _positive_int(value["N"], "N")
    gridsize = _positive_int(value["gridsize"], "gridsize")
    size = _positive_int(value["size"], "size")
    offset = _positive_int(value["offset"], "offset")
    outer_offset_train = _nonnegative_int(
        value["outer_offset_train"], "outer_offset_train"
    )
    outer_offset_test = _nonnegative_int(
        value["outer_offset_test"], "outer_offset_test"
    )
    nimgs_train = _positive_int(value["nimgs_train"], "nimgs_train")
    nimgs_test = _positive_int(value["nimgs_test"], "nimgs_test")
    declared_probe = (Path(base_dir) / declared_archive).resolve()
    scale_mode = _text(value["probe_scale_mode"], "probe_scale_mode")
    transform = f"{scale_mode}:{N}"
    if sigma > 0:
        if scale_mode == "pad_preserve":
            transform = f"smooth:{sigma:g}|{transform}"
        else:
            transform = f"{transform}|smooth:{sigma:g}"
    simulation = SimulationConfig(
        N=N,
        probe=ProbeSimulationConfig(
            source="custom",
            source_path=declared_probe,
            transform_pipeline=transform,
        ),
        object=SyntheticObjectConfig(
            kind="lines",
            image_size=(size, size),
            set_phi=set_phi,
        ),
        scan=ScanSimulationConfig(
            kind="grid",
            grid_size=(gridsize, gridsize),
            offset=offset,
            outer_offset_train=outer_offset_train,
            outer_offset_test=outer_offset_test,
            train_groups=nimgs_train,
            test_groups=nimgs_test,
        ),
        detector=DetectorSimulationConfig(photons_per_pattern=nphotons),
    )
    return GridLinesReferenceRecipe(
        id=declared_id,
        generator=generator,
        probe_archive=declared_probe,
        probe_archive_declared=declared_archive,
        probe_archive_sha256=_sha(
            value["probe_archive_sha256"], "probe_archive_sha256"
        ),
        raw_probe_array_sha256=_sha(
            value["raw_probe_array_sha256"], "raw_probe_array_sha256"
        ),
        transformed_probe_sha256=_sha(
            value["transformed_probe_sha256"], "transformed_probe_sha256"
        ),
        probe_scale_mode=scale_mode,
        probe_smoothing_sigma=sigma,
        set_phi=set_phi,
        N=N,
        gridsize=gridsize,
        size=size,
        offset=offset,
        outer_offset_train=outer_offset_train,
        outer_offset_test=outer_offset_test,
        nimgs_train=nimgs_train,
        nimgs_test=nimgs_test,
        nphotons=nphotons,
        schema_version=1,
        simulation=simulation,
    )


_V2_IDENTITY_FIELDS = {
    "id",
    "generator",
    "probe_archive_sha256",
    "raw_probe_array_sha256",
    "transformed_probe_sha256",
}


def parse_grid_lines_reference_simulation(
    identity: Mapping[str, Any],
    simulation_values: Mapping[str, Any],
    *,
    base_dir: Path,
) -> GridLinesReferenceRecipe:
    """Parse a schema-v2 reference recipe with canonical nested ownership."""
    if not isinstance(identity, Mapping):
        raise DatasetError("dataset identity must be a table")
    unknown = set(identity) - _V2_IDENTITY_FIELDS
    missing = _V2_IDENTITY_FIELDS - set(identity)
    if unknown or missing:
        raise DatasetError(
            "dataset identity fields do not match schema v2 "
            f"(missing={sorted(missing)}, unexpected={sorted(unknown)})"
        )
    identifier = _text(identity["id"], "id")
    if not _ID_RE.fullmatch(identifier):
        raise DatasetError(f"simulation recipe id {identifier!r} is invalid")
    generator = _text(identity["generator"], "generator")
    if generator != GRID_LINES_REFERENCE_GENERATOR:
        raise DatasetError(
            f"simulation recipe generator must be {GRID_LINES_REFERENCE_GENERATOR!r}"
        )
    if not isinstance(simulation_values, Mapping):
        raise DatasetError("simulation recipe must be a table")
    try:
        simulation = simulation_config_from_mapping(simulation_values)
    except (TypeError, ValueError) as exc:
        raise DatasetError(f"simulation recipe config is invalid: {exc}") from exc
    if simulation.object.kind != "lines" or simulation.scan.kind != "grid":
        raise DatasetError(
            "grid-lines reference requires simulation.object.kind='lines' and "
            "simulation.scan.kind='grid'"
        )
    if simulation.probe.source != "custom" or simulation.probe.source_path is None:
        raise DatasetError(
            "grid-lines reference requires a custom simulation.probe.source_path"
        )
    declared_archive = str(simulation.probe.source_path)
    resolved_archive = (Path(base_dir) / simulation.probe.source_path).resolve()
    steps = simulation.probe.transform_pipeline.split("|")
    smoothing = 0.0
    for step in steps:
        if step.startswith("smooth:"):
            try:
                smoothing = float(step.split(":", 1)[1])
            except ValueError as exc:
                raise DatasetError(
                    "simulation probe smoothing must be numeric"
                ) from exc
            if not math.isfinite(smoothing) or smoothing < 0:
                raise DatasetError(
                    "simulation probe smoothing must be finite and nonnegative"
                )
    grid_size = simulation.scan.grid_size[0]
    return GridLinesReferenceRecipe(
        id=identifier,
        generator=generator,
        probe_archive=resolved_archive,
        probe_archive_declared=declared_archive,
        probe_archive_sha256=_sha(
            identity["probe_archive_sha256"], "probe_archive_sha256"
        ),
        raw_probe_array_sha256=_sha(
            identity["raw_probe_array_sha256"], "raw_probe_array_sha256"
        ),
        transformed_probe_sha256=_sha(
            identity["transformed_probe_sha256"], "transformed_probe_sha256"
        ),
        probe_scale_mode="pipeline",
        probe_smoothing_sigma=smoothing,
        set_phi=simulation.object.set_phi,
        N=simulation.N,
        gridsize=grid_size,
        size=simulation.object.image_size[0],
        offset=simulation.scan.offset,
        outer_offset_train=simulation.scan.outer_offset_train,
        outer_offset_test=simulation.scan.outer_offset_test,
        nimgs_train=simulation.scan.train_groups,
        nimgs_test=simulation.scan.test_groups,
        nphotons=simulation.detector.photons_per_pattern,
        schema_version=2,
        simulation=simulation,
    )


def _validate_split(
    recipe: GridLinesReferenceRecipe, path: Path, split: str
) -> tuple[str, str, int]:
    """Return (file_sha256, probe canonical sha256, sample count) for a split."""
    npz_path = Path(path)
    if not npz_path.is_file():
        raise DatasetError(f"reference {split} NPZ is missing: {npz_path}")
    digest = file_sha256(npz_path)
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            diffraction = (
                np.asarray(archive["diffraction"]) if "diffraction" in keys else None
            )
            probe = np.asarray(archive["probeGuess"]) if "probeGuess" in keys else None
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise DatasetError(
            f"cannot read reference {split} NPZ {npz_path}: {exc}"
        ) from exc
    missing = [key for key in _REQUIRED_SPLIT_KEYS if key not in keys]
    if missing:
        raise DatasetError(f"reference {split} NPZ is missing required keys {missing}")
    if split == "test" and not keys & {"YY_ground_truth", "YY_full"}:
        raise DatasetError("reference test NPZ must carry YY_ground_truth or YY_full")
    assert diffraction is not None and probe is not None
    if diffraction.ndim < 3 or diffraction.shape[1:3] != (recipe.N, recipe.N):
        raise DatasetError(
            f"reference {split} diffraction shape {diffraction.shape} does not "
            f"match recipe detector size N={recipe.N}"
        )
    if diffraction.shape[0] < 1:
        raise DatasetError(f"reference {split} NPZ contains no samples")
    if probe.shape != (recipe.N, recipe.N) or not np.iscomplexobj(probe):
        raise DatasetError(
            f"reference {split} probeGuess must be complex ({recipe.N}, {recipe.N}); "
            f"got {probe.dtype} {probe.shape}"
        )
    probe_sha = canonical_array_sha256(probe)
    if probe_sha != recipe.transformed_probe_sha256:
        raise DatasetError(
            f"reference {split} probeGuess hash {probe_sha} does not match the "
            f"recipe transformed probe {recipe.transformed_probe_sha256}"
        )
    return digest, probe_sha, int(diffraction.shape[0])


def _validate_probe_archive(recipe: GridLinesReferenceRecipe) -> None:
    """Fail closed unless the probe archive matches its pinned identity."""
    if not recipe.probe_archive.is_file():
        raise DatasetError(
            f"reference probe archive is missing: {recipe.probe_archive}"
        )
    archive_sha = file_sha256(recipe.probe_archive)
    if archive_sha != recipe.probe_archive_sha256:
        raise DatasetError(
            f"reference probe archive hash {archive_sha} does not match the "
            f"pinned identity {recipe.probe_archive_sha256}"
        )
    try:
        with np.load(recipe.probe_archive, allow_pickle=False) as archive:
            raw_probe = np.asarray(archive["probeGuess"])
    except (OSError, KeyError, ValueError) as exc:
        raise DatasetError(
            f"cannot read probeGuess from reference probe archive "
            f"{recipe.probe_archive}: {exc}"
        ) from exc
    raw_sha = canonical_array_sha256(raw_probe)
    if raw_sha != recipe.raw_probe_array_sha256:
        raise DatasetError(
            f"reference raw probe array hash {raw_sha} does not match the "
            f"pinned identity {recipe.raw_probe_array_sha256}"
        )


def validate_reference_npz_pair(
    recipe: GridLinesReferenceRecipe,
    train_npz: Path,
    test_npz: Path,
) -> MaterializedReferenceDataset:
    """Anchor a materialized NPZ pair to the recipe and fingerprint its content.

    Fails closed on a missing/tampered probe archive, a probe that does not
    hash to the pinned transformed-probe identity, wrong detector size, or
    missing required keys. Content hashes become sealed run evidence.
    """
    if not isinstance(recipe, GridLinesReferenceRecipe):
        raise DatasetError("recipe must be a GridLinesReferenceRecipe")
    _validate_probe_archive(recipe)
    train_sha, train_probe_sha, n_train = _validate_split(recipe, train_npz, "train")
    test_sha, test_probe_sha, n_test = _validate_split(recipe, test_npz, "test")
    if train_probe_sha != test_probe_sha:
        raise DatasetError("reference train/test NPZ probes disagree")
    return MaterializedReferenceDataset(
        recipe_fingerprint_sha256=recipe.fingerprint_sha256,
        train_path=Path(train_npz).resolve(),
        test_path=Path(test_npz).resolve(),
        train_sha256=train_sha,
        test_sha256=test_sha,
        probe_sha256=train_probe_sha,
        n_train=n_train,
        n_test=n_test,
    )


# ---------------------------------------------------------------------------
# Bridge-ladder dataset recipes (plan Task 21): a grid-lines base recipe plus
# the measurement expression the rung consumes. ``dictionary`` is the historic
# cached-NPZ schema; ``generic_amplitude`` is its deterministic generic-schema
# twin (diff3d/xcoords/ycoords, values unchanged); ``generic_count_intensity``
# is the aligned count twin (scripts/studies/make_aligned_count_twin.py):
# diffraction re-expressed as physical counts, all other keys verbatim.
# ---------------------------------------------------------------------------

LADDER_DATASET_EXPRESSIONS = (
    "dictionary",
    "generic_amplitude",
    "generic_count_intensity",
)
_GENERIC_REQUIRED_KEYS = ("diff3d", "xcoords", "ycoords", "probeGuess")


@dataclass(frozen=True)
class LadderDatasetRecipe:
    """One rung dataset: pinned grid-lines recipe plus measurement expression."""

    id: str
    expression: str
    recipe: GridLinesReferenceRecipe

    @property
    def fingerprint_sha256(self) -> str:
        encoded = json.dumps(
            {
                "expression": self.expression,
                "recipe_fingerprint_sha256": self.recipe.fingerprint_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def parse_ladder_dataset(
    dataset_id: str,
    value: Mapping[str, Any],
    *,
    base_dir: Path,
) -> LadderDatasetRecipe:
    """Parse one closed ladder dataset table; no artifact I/O happens here."""
    if not isinstance(value, Mapping):
        raise DatasetError("ladder dataset must be a table")
    if set(value) != {"expression", "recipe"}:
        raise DatasetError(
            "ladder dataset fields must be exactly {expression, recipe}; got "
            f"{sorted(value)}"
        )
    expression = value["expression"]
    if expression not in LADDER_DATASET_EXPRESSIONS:
        raise DatasetError(
            f"ladder dataset expression {expression!r} must be one of "
            f"{LADDER_DATASET_EXPRESSIONS}"
        )
    recipe = parse_grid_lines_reference_recipe(
        dataset_id, value["recipe"], base_dir=base_dir
    )
    return LadderDatasetRecipe(id=dataset_id, expression=expression, recipe=recipe)


def _validate_generic_split(
    dataset: LadderDatasetRecipe, path: Path, split: str
) -> tuple[str, str, int]:
    """Return (file sha256, probe canonical sha256, sample count) for a split."""
    recipe = dataset.recipe
    npz_path = Path(path)
    if not npz_path.is_file():
        raise DatasetError(f"ladder {split} NPZ is missing: {npz_path}")
    digest = file_sha256(npz_path)
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            keys = set(archive.files)
            missing = [key for key in _GENERIC_REQUIRED_KEYS if key not in keys]
            if missing:
                raise DatasetError(
                    f"ladder {split} NPZ is missing generic-schema keys {missing}"
                )
            if split == "test" and "objectGuess" not in keys:
                raise DatasetError("ladder test NPZ must carry objectGuess truth")
            diff3d = np.asarray(archive["diff3d"])
            probe = np.asarray(archive["probeGuess"])
            xcoords = np.asarray(archive["xcoords"])
            ycoords = np.asarray(archive["ycoords"])
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        raise DatasetError(f"cannot read ladder {split} NPZ {npz_path}: {exc}") from exc
    if diff3d.ndim == 4 and diff3d.shape[-1] == 1:
        diff3d = diff3d[..., 0]
    if diff3d.ndim != 3 or diff3d.shape[1:] != (recipe.N, recipe.N):
        raise DatasetError(
            f"ladder {split} diff3d shape {diff3d.shape} does not match the "
            f"recipe detector size N={recipe.N}"
        )
    if diff3d.shape[0] < 1:
        raise DatasetError(f"ladder {split} NPZ contains no samples")
    for name, coords in (("xcoords", xcoords), ("ycoords", ycoords)):
        if coords.ndim != 1 or coords.shape[0] != diff3d.shape[0]:
            raise DatasetError(
                f"ladder {split} {name} must be 1D with one entry per pattern"
            )
    if dataset.expression == "generic_count_intensity" and np.any(diff3d < 0):
        raise DatasetError(
            f"ladder {split} count-intensity measurements contain negative values"
        )
    if probe.shape != (recipe.N, recipe.N) or not np.iscomplexobj(probe):
        raise DatasetError(
            f"ladder {split} probeGuess must be complex ({recipe.N}, {recipe.N}); "
            f"got {probe.dtype} {probe.shape}"
        )
    probe_sha = canonical_array_sha256(probe)
    if probe_sha != recipe.transformed_probe_sha256:
        raise DatasetError(
            f"ladder {split} probeGuess hash {probe_sha} does not match the "
            f"recipe transformed probe {recipe.transformed_probe_sha256}"
        )
    return digest, probe_sha, int(diff3d.shape[0])


def validate_ladder_npz_pair(
    dataset: LadderDatasetRecipe,
    train_npz: Path,
    test_npz: Path,
) -> MaterializedReferenceDataset:
    """Anchor a rung's materialized NPZ pair to its recipe and fingerprint it.

    Dictionary-expression datasets delegate to the reference validator; the
    generic twins validate the generic schema, the pinned probe identity, and
    (for count twins) count nonnegativity. Fails closed, mirroring
    :func:`validate_reference_npz_pair`.
    """
    if not isinstance(dataset, LadderDatasetRecipe):
        raise DatasetError("dataset must be a LadderDatasetRecipe")
    if dataset.expression == "dictionary":
        materialized = validate_reference_npz_pair(dataset.recipe, train_npz, test_npz)
        return MaterializedReferenceDataset(
            recipe_fingerprint_sha256=dataset.fingerprint_sha256,
            train_path=materialized.train_path,
            test_path=materialized.test_path,
            train_sha256=materialized.train_sha256,
            test_sha256=materialized.test_sha256,
            probe_sha256=materialized.probe_sha256,
            n_train=materialized.n_train,
            n_test=materialized.n_test,
        )
    _validate_probe_archive(dataset.recipe)
    train_sha, train_probe_sha, n_train = _validate_generic_split(
        dataset, train_npz, "train"
    )
    test_sha, test_probe_sha, n_test = _validate_generic_split(
        dataset, test_npz, "test"
    )
    if train_probe_sha != test_probe_sha:
        raise DatasetError("ladder train/test NPZ probes disagree")
    return MaterializedReferenceDataset(
        recipe_fingerprint_sha256=dataset.fingerprint_sha256,
        train_path=Path(train_npz).resolve(),
        test_path=Path(test_npz).resolve(),
        train_sha256=train_sha,
        test_sha256=test_sha,
        probe_sha256=train_probe_sha,
        n_train=n_train,
        n_test=n_test,
    )
