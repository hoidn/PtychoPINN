"""Versioned deterministic synthetic-object producer registry."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import io
from numbers import Integral
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np


LINES_OBJECT_RECIPE = "lines-object-v1"
LINES_OBJECT_PRODUCER_SYMBOLS = (
    "ptycho.diffsim.mk_lines_img",
    "ptycho.diffsim.dummy_phi",
)
PER_OBJECT_DUMMY_PHASE_LAW = "per-object-dummy-phi-v1"
DEAD_LEAVES_OBJECT_RECIPE_V1 = "dead-leaves-object-v1"
DEAD_LEAVES_OBJECT_RECIPE = "dead-leaves-object-v2"
DEAD_LEAVES_RNG_CONTRACT_V1 = "dead-leaves-rng-v1"
DEAD_LEAVES_RNG_CONTRACT = "dead-leaves-rng-v2"
DEAD_LEAVES_OBJECT_PRODUCER_SYMBOLS_V1 = (
    "ptycho_torch.datagen.objects.create_dead_leaves",
    "ptycho.diffsim.dummy_phi",
)
DEAD_LEAVES_OBJECT_PRODUCER_SYMBOLS = (
    "ptycho_torch.datagen.objects.create_dead_leaves",
    "ptycho.simulation.object_producers.fixed_dead_leaves_phase",
)
DEAD_LEAVES_PHASE_LAW = "dead-leaves-fixed-phase-v1"
DEAD_LEAVES_PHASE_REFERENCE_MAX = 1.1
DEAD_LEAVES_PHASE_REFERENCE_MEAN = 0.95
DEAD_LEAVES_OBJECT_ARGUMENTS = {
    "max_iters": 700,
    "r_min_frac": 0.02,
    "r_max_frac": 0.18,
    "r_sigma": 3,
}
FROZEN_OBJECT_BANK_RECIPE = "frozen-object-bank-v1"
FROZEN_OBJECT_BANK_SOURCE_VERSION = "frozen-object-bank-source-v1"
FROZEN_OBJECT_BANK_SELECTION_VERSION = "frozen-object-bank-selection-v1"
FROZEN_COMPLEX_PHASE_LAW = "frozen-complex-source-v1"
FROZEN_OBJECT_BANK_PRODUCER_SYMBOLS = (
    "ptycho.simulation.object_producers.load_frozen_object_banks",
    "ptycho.simulation.object_producers._frozen_object_from_array",
)
FROZEN_OBJECT_BANK_KEYS = ("trainObjectGuess", "testObjectGuess")


@dataclass(frozen=True)
class LinesObject:
    """One locked lines-object array and its stable producer identity."""

    array: np.ndarray
    recipe: str = LINES_OBJECT_RECIPE
    producer_symbols: tuple[str, str] = LINES_OBJECT_PRODUCER_SYMBOLS
    rng_identity: Mapping[str, Any] = field(default_factory=dict)
    phase_identity: Mapping[str, Any] = field(default_factory=dict)
    source_identity: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DeadLeavesObject:
    """One locked dead-leaves array and its stable producer identity."""

    array: np.ndarray
    recipe: str = DEAD_LEAVES_OBJECT_RECIPE
    producer_symbols: tuple[str, str] = DEAD_LEAVES_OBJECT_PRODUCER_SYMBOLS
    rng_identity: Mapping[str, Any] = field(default_factory=dict)
    phase_identity: Mapping[str, Any] = field(default_factory=dict)
    source_identity: Mapping[str, Any] = field(default_factory=dict)


ObjectBuildResult = LinesObject | DeadLeavesObject


@dataclass(frozen=True)
class ObjectRandomStreams:
    """Named random streams consumed by one registered object builder."""

    numeric: np.random.Generator
    shape: np.random.Generator | None = None


@dataclass(frozen=True)
class ObjectProducer:
    """One registered kind/recipe pair and its deterministic builder."""

    kind: str
    recipe: str
    producer_symbols: tuple[str, str]
    build: Callable[[ObjectRandomStreams], ObjectBuildResult] | None


def _validate_object_array(array: np.ndarray, *, recipe: str) -> np.ndarray:
    object_guess = np.ascontiguousarray(np.asarray(array, dtype=np.complex64))
    if object_guess.shape != (392, 392):
        raise ValueError(
            f"{recipe} must produce shape (392, 392), got {object_guess.shape}"
        )
    if not np.isfinite(object_guess).all():
        raise ValueError(f"{recipe} produced nonfinite values")
    return object_guess


def fixed_dead_leaves_phase(amplitude: np.ndarray) -> np.ndarray:
    """Apply the v2 object-independent material phase law."""

    values = np.asarray(amplitude, dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("dead-leaves amplitude must contain only finite values")
    return np.asarray(
        np.float32(np.pi)
        * np.tanh(
            (values - np.float32(DEAD_LEAVES_PHASE_REFERENCE_MAX / 2.0))
            / np.float32(3.0 * DEAD_LEAVES_PHASE_REFERENCE_MEAN)
        ),
        dtype=np.float32,
    )


def phase_identity_for_recipe(recipe: str) -> dict[str, Any]:
    """Return the phase law owned by a registered object recipe."""

    if recipe == DEAD_LEAVES_OBJECT_RECIPE:
        return {
            "version": DEAD_LEAVES_PHASE_LAW,
            "reference_max_amplitude": DEAD_LEAVES_PHASE_REFERENCE_MAX,
            "reference_mean_amplitude": DEAD_LEAVES_PHASE_REFERENCE_MEAN,
        }
    if recipe in {LINES_OBJECT_RECIPE, DEAD_LEAVES_OBJECT_RECIPE_V1}:
        return {"version": PER_OBJECT_DUMMY_PHASE_LAW}
    if recipe == FROZEN_OBJECT_BANK_RECIPE:
        return {"version": FROZEN_COMPLEX_PHASE_LAW}
    raise ValueError(f"unsupported object recipe {recipe!r}")


def _build_lines_object(streams: ObjectRandomStreams) -> LinesObject:
    from ptycho import diffsim

    morphology = diffsim.mk_lines_img(784, nlines=400, rng=streams.numeric)
    amplitude = np.asarray(morphology)[196:-196, 196:-196, 0]
    phase = np.asarray(diffsim.dummy_phi(amplitude), dtype=np.float32)
    object_guess = amplitude * np.exp(1j * phase)
    return LinesObject(
        array=_validate_object_array(object_guess, recipe=LINES_OBJECT_RECIPE),
        phase_identity=phase_identity_for_recipe(LINES_OBJECT_RECIPE),
    )


def _build_dead_leaves_object(
    streams: ObjectRandomStreams,
    *,
    recipe: str,
) -> DeadLeavesObject:
    from ptycho import diffsim
    from ptycho_torch.datagen.objects import create_dead_leaves

    raw_object = create_dead_leaves(
        (392, 392),
        DEAD_LEAVES_OBJECT_ARGUMENTS,
        rng=streams.numeric,
        shape_rng=streams.shape,
    )
    amplitude = np.asarray(np.abs(raw_object))
    phase = (
        np.asarray(diffsim.dummy_phi(amplitude), dtype=np.float32)
        if recipe == DEAD_LEAVES_OBJECT_RECIPE_V1
        else fixed_dead_leaves_phase(amplitude)
    )
    object_guess = amplitude * np.exp(1j * phase)
    return DeadLeavesObject(
        array=_validate_object_array(
            object_guess,
            recipe=recipe,
        ),
        recipe=recipe,
        phase_identity=phase_identity_for_recipe(recipe),
    )


def _build_dead_leaves_object_v1(
    streams: ObjectRandomStreams,
) -> DeadLeavesObject:
    result = _build_dead_leaves_object(
        streams,
        recipe=DEAD_LEAVES_OBJECT_RECIPE_V1,
    )
    return replace(
        result,
        producer_symbols=DEAD_LEAVES_OBJECT_PRODUCER_SYMBOLS_V1,
    )


def _build_dead_leaves_object_v2(
    streams: ObjectRandomStreams,
) -> DeadLeavesObject:
    if streams.shape is None or streams.shape is streams.numeric:
        raise ValueError(
            "dead-leaves-object-v2 requires independent shape and numeric streams"
        )
    return _build_dead_leaves_object(
        streams,
        recipe=DEAD_LEAVES_OBJECT_RECIPE,
    )


OBJECT_PRODUCERS = {
    ("lines", LINES_OBJECT_RECIPE): ObjectProducer(
        kind="lines",
        recipe=LINES_OBJECT_RECIPE,
        producer_symbols=LINES_OBJECT_PRODUCER_SYMBOLS,
        build=_build_lines_object,
    ),
    ("dead_leaves", DEAD_LEAVES_OBJECT_RECIPE_V1): ObjectProducer(
        kind="dead_leaves",
        recipe=DEAD_LEAVES_OBJECT_RECIPE_V1,
        producer_symbols=DEAD_LEAVES_OBJECT_PRODUCER_SYMBOLS_V1,
        build=_build_dead_leaves_object_v1,
    ),
    ("dead_leaves", DEAD_LEAVES_OBJECT_RECIPE): ObjectProducer(
        kind="dead_leaves",
        recipe=DEAD_LEAVES_OBJECT_RECIPE,
        producer_symbols=DEAD_LEAVES_OBJECT_PRODUCER_SYMBOLS,
        build=_build_dead_leaves_object_v2,
    ),
    ("lines", FROZEN_OBJECT_BANK_RECIPE): ObjectProducer(
        kind="lines",
        recipe=FROZEN_OBJECT_BANK_RECIPE,
        producer_symbols=FROZEN_OBJECT_BANK_PRODUCER_SYMBOLS,
        build=None,
    ),
    ("dead_leaves", FROZEN_OBJECT_BANK_RECIPE): ObjectProducer(
        kind="dead_leaves",
        recipe=FROZEN_OBJECT_BANK_RECIPE,
        producer_symbols=FROZEN_OBJECT_BANK_PRODUCER_SYMBOLS,
        build=None,
    ),
}

DEFAULT_OBJECT_RECIPES = {
    "lines": LINES_OBJECT_RECIPE,
    "dead_leaves": DEAD_LEAVES_OBJECT_RECIPE,
}


def registered_object_kinds() -> tuple[str, ...]:
    """Return object kinds exposed by the producer registry, in registration order."""

    return tuple(dict.fromkeys(kind for kind, _recipe in OBJECT_PRODUCERS))


def object_recipe_for_kind(kind: str) -> str:
    """Return the current default registered recipe for ``kind``."""

    recipe = DEFAULT_OBJECT_RECIPES.get(kind)
    if recipe is None:
        supported = ", ".join(repr(name) for name in registered_object_kinds())
        raise ValueError(
            f"unsupported simulation.object.kind {kind!r}; expected one of "
            f"{supported}"
        )
    return recipe


def validate_object_recipe(kind: str, recipe: str) -> ObjectProducer:
    """Return the registered producer or fail on an unsupported pair."""

    producer = OBJECT_PRODUCERS.get((kind, recipe))
    if producer is None:
        object_recipe_for_kind(kind)
        expected = tuple(
            registered_recipe
            for registered_kind, registered_recipe in OBJECT_PRODUCERS
            if registered_kind == kind
        )
        raise ValueError(
            f"simulation.object_recipe {recipe!r} does not match "
            f"simulation.object.kind {kind!r}; expected one of {expected!r}"
        )
    return producer


def build_object(
    kind: str,
    recipe: str,
    rng: np.random.Generator,
    *,
    shape_rng: np.random.Generator | None = None,
) -> ObjectBuildResult:
    """Build an object through the registered kind/recipe producer."""

    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")
    if shape_rng is not None and not isinstance(shape_rng, np.random.Generator):
        raise TypeError("shape_rng must be a numpy.random.Generator")
    if recipe == DEAD_LEAVES_OBJECT_RECIPE_V1:
        if shape_rng is not None and shape_rng is not rng:
            raise ValueError(
                "dead-leaves-object-v1 requires one combined random stream"
            )
        shape_rng = rng
    producer = validate_object_recipe(kind, recipe)
    if producer.build is None:
        raise ValueError(
            f"simulation.object_recipe {recipe!r} is source-backed; use "
            "load_frozen_object_banks instead of an RNG builder"
        )
    return producer.build(ObjectRandomStreams(numeric=rng, shape=shape_rng))


def _frozen_object_from_array(
    kind: str,
    array: np.ndarray,
    *,
    source_identity: Mapping[str, Any],
) -> ObjectBuildResult:
    object_type = LinesObject if kind == "lines" else DeadLeavesObject
    return object_type(
        array=np.ascontiguousarray(array),
        recipe=FROZEN_OBJECT_BANK_RECIPE,
        producer_symbols=FROZEN_OBJECT_BANK_PRODUCER_SYMBOLS,
        phase_identity=phase_identity_for_recipe(FROZEN_OBJECT_BANK_RECIPE),
        source_identity=dict(source_identity),
    )


def load_frozen_object_banks(
    kind: str,
    source_path: str | Path,
    *,
    train_count: int,
    test_count: int,
    image_size: tuple[int, int],
    shared_object: bool,
) -> tuple[dict[str, list[ObjectBuildResult]], dict[str, Any]]:
    """Load an exact complex object bank and bind every canvas to its source."""

    from ptycho.simulation.identity import array_sha256

    validate_object_recipe(kind, FROZEN_OBJECT_BANK_RECIPE)
    for name, value in (("train_count", train_count), ("test_count", test_count)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    expected_size = tuple(int(value) for value in image_size)
    if len(expected_size) != 2 or any(value <= 0 for value in expected_size):
        raise ValueError("image_size must contain two positive integers")

    source = Path(source_path)
    if not source.is_file():
        raise FileNotFoundError(f"frozen object bank does not exist: {source}")
    try:
        source_snapshot = source.read_bytes()
        source_digest = hashlib.sha256(source_snapshot).hexdigest()
        with np.load(io.BytesIO(source_snapshot), allow_pickle=False) as archive:
            member_names = tuple(archive.files)
            if (
                len(member_names) != len(FROZEN_OBJECT_BANK_KEYS)
                or set(member_names) != set(FROZEN_OBJECT_BANK_KEYS)
            ):
                raise ValueError(
                    "frozen object bank must contain exactly "
                    f"{FROZEN_OBJECT_BANK_KEYS!r}; got {member_names!r}"
                )
            arrays = {
                name: np.array(archive[name], copy=True)
                for name in FROZEN_OBJECT_BANK_KEYS
            }
    except (OSError, ValueError) as error:
        if isinstance(error, ValueError) and str(error).startswith(
            "frozen object bank must contain exactly"
        ):
            raise
        raise ValueError(f"invalid frozen object bank {source}: {error}") from error

    expected_shapes = {
        "trainObjectGuess": (int(train_count), *expected_size),
        "testObjectGuess": (int(test_count), *expected_size),
    }
    for name, array in arrays.items():
        if array.dtype != np.dtype(np.complex64):
            raise ValueError(
                f"frozen object bank {name} must have dtype complex64, "
                f"got {array.dtype.name}"
            )
        if array.shape != expected_shapes[name]:
            raise ValueError(
                f"frozen object bank {name} shape must be "
                f"{expected_shapes[name]}, got {array.shape}"
            )
        if not np.isfinite(array).all():
            raise ValueError(f"frozen object bank {name} must contain finite values")
        arrays[name] = np.ascontiguousarray(array)
    if shared_object and not np.array_equal(
        arrays["trainObjectGuess"], arrays["testObjectGuess"]
    ):
        raise ValueError(
            "simulation.shared_object=True requires identical frozen train and "
            "test object banks"
        )

    source_record = {
        "version": FROZEN_OBJECT_BANK_SOURCE_VERSION,
        "source_path": str(source),
        "source_file_sha256": source_digest,
        "arrays": {
            name: {
                "array_sha256": array_sha256(array),
                "shape": list(array.shape),
                "dtype": array.dtype.name,
            }
            for name, array in arrays.items()
        },
    }
    banks: dict[str, list[ObjectBuildResult]] = {}
    for split, key in (("train", "trainObjectGuess"), ("test", "testObjectGuess")):
        banks[split] = [
            _frozen_object_from_array(
                kind,
                canvas,
                source_identity={
                    "version": FROZEN_OBJECT_BANK_SELECTION_VERSION,
                    "source_key": key,
                    "source_index": index,
                    "source_file_sha256": source_digest,
                    "source_array_sha256": array_sha256(canvas),
                },
            )
            for index, canvas in enumerate(arrays[key])
        ]
    return banks, source_record


def _seed_sequence_identity(seed_sequence: np.random.SeedSequence) -> dict[str, Any]:
    return {
        "entropy": int(seed_sequence.entropy),
        "spawn_key": [int(value) for value in seed_sequence.spawn_key],
    }


def _validated_parent_seed(seed: int) -> tuple[int, np.random.SeedSequence]:
    if isinstance(seed, bool) or not isinstance(seed, Integral):
        raise TypeError("seed must be a nonnegative integer")
    seed = int(seed)
    if seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return seed, np.random.SeedSequence(seed)


def rng_identity_for_seed(recipe: str, seed: int) -> dict[str, Any]:
    """Return the complete recipe-owned RNG identity without drawing an object."""

    seed, parent = _validated_parent_seed(seed)
    if recipe == DEAD_LEAVES_OBJECT_RECIPE:
        shape_seed, numeric_seed = parent.spawn(2)
        return {
            "version": DEAD_LEAVES_RNG_CONTRACT,
            "parent_seed": seed,
            "streams": {
                "shape": {
                    "bit_generator": "PCG64",
                    "seed_sequence": _seed_sequence_identity(shape_seed),
                },
                "numeric": {
                    "bit_generator": "PCG64",
                    "seed_sequence": _seed_sequence_identity(numeric_seed),
                },
            },
        }
    if recipe == DEAD_LEAVES_OBJECT_RECIPE_V1:
        version = DEAD_LEAVES_RNG_CONTRACT_V1
    elif recipe == LINES_OBJECT_RECIPE:
        version = "lines-object-rng-v1"
    else:
        raise ValueError(f"unsupported object recipe {recipe!r}")
    return {
        "version": version,
        "parent_seed": seed,
        "streams": {
            "combined": {
                "bit_generator": "PCG64",
                "seed_sequence": _seed_sequence_identity(parent),
            }
        },
    }


def build_object_from_seed(
    kind: str,
    recipe: str,
    seed: int,
) -> ObjectBuildResult:
    """Build one object from a recorded parent seed and recipe-owned streams."""

    seed, parent = _validated_parent_seed(seed)
    validate_object_recipe(kind, recipe)

    if recipe == DEAD_LEAVES_OBJECT_RECIPE:
        shape_seed, numeric_seed = parent.spawn(2)
        result = build_object(
            kind,
            recipe,
            np.random.Generator(np.random.PCG64(numeric_seed)),
            shape_rng=np.random.Generator(np.random.PCG64(shape_seed)),
        )
    else:
        numeric_rng = np.random.default_rng(parent)
        result = build_object(kind, recipe, numeric_rng)
    return replace(result, rng_identity=rng_identity_for_seed(recipe, seed))


def build_lines_object(rng: np.random.Generator) -> LinesObject:
    """Build the registered ``lines-object-v1`` producer."""

    result = build_object("lines", LINES_OBJECT_RECIPE, rng)
    assert isinstance(result, LinesObject)
    return result


def build_dead_leaves_object(
    rng: np.random.Generator,
    *,
    shape_rng: np.random.Generator,
) -> DeadLeavesObject:
    """Build the registered ``dead-leaves-object-v2`` producer."""

    result = build_object(
        "dead_leaves",
        DEAD_LEAVES_OBJECT_RECIPE,
        rng,
        shape_rng=shape_rng,
    )
    assert isinstance(result, DeadLeavesObject)
    return result
