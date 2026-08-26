"""Pure fixed selection for representative rectangular dose closure."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
import operator
from typing import Any

from torch.utils.data import Subset

from ptycho_torch.rect_s1s2_initialization import (
    RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED,
)


_UINT64_MODULUS = 1 << 64
_UINT64_MASK = _UINT64_MODULUS - 1
_SPLITMIX64_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX64_MULTIPLIER_1 = 0xBF58476D1CE4E5B9
_SPLITMIX64_MULTIPLIER_2 = 0x94D049BB133111EB


@dataclass(frozen=True, slots=True)
class SelectedDoseClosureRow:
    """Selected channels for one logical row, annotated for physical access."""

    logical_row: int
    base_row: int
    channels: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class DoseClosureSamplePlan:
    """Immutable canonical draws and their physical-read ordering."""

    population_patterns: int
    flat_slots: tuple[int, ...]
    access_rows: tuple[SelectedDoseClosureRow, ...]


def _next_splitmix64(state: int) -> tuple[int, int]:
    state = (state + _SPLITMIX64_GAMMA) & _UINT64_MASK
    candidate = state
    candidate = (
        (candidate ^ (candidate >> 30)) * _SPLITMIX64_MULTIPLIER_1
    ) & _UINT64_MASK
    candidate = (
        (candidate ^ (candidate >> 27)) * _SPLITMIX64_MULTIPLIER_2
    ) & _UINT64_MASK
    candidate = (candidate ^ (candidate >> 31)) & _UINT64_MASK
    return state, candidate


def _map_bounded_candidate(candidate: int, bound: int) -> int | None:
    """Map one uint64 without modulo bias, or reject the biased tail."""

    limit = _UINT64_MODULUS - (_UINT64_MODULUS % bound)
    if candidate >= limit:
        return None
    return candidate % bound


def _draw_flat_slots(population_patterns: int) -> tuple[int, ...]:
    state = RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED & _UINT64_MASK
    selected: list[int] = []
    selected_set: set[int] = set()
    while len(selected) < RECT_S1S2_DOSE_CLOSURE_PATTERNS:
        state, candidate = _next_splitmix64(state)
        mapped = _map_bounded_candidate(candidate, population_patterns)
        if mapped is None or mapped in selected_set:
            continue
        selected.append(mapped)
        selected_set.add(mapped)
    return tuple(selected)


def _subset_index(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("torch Subset index must be an integer, not bool")
    try:
        return operator.index(value)
    except TypeError as error:
        raise TypeError(
            f"torch Subset index must be an integer, got {value!r}"
        ) from error


def _base_row_for_logical(dataset: Any, logical_row: int) -> int:
    current_dataset = dataset
    current_row = logical_row
    while isinstance(current_dataset, Subset):
        try:
            mapped_value = current_dataset.indices[current_row]
        except Exception as error:
            raise ValueError(
                f"torch Subset index {current_row} could not be resolved"
            ) from error
        current_row = _subset_index(mapped_value)
        parent_dataset = current_dataset.dataset
        try:
            parent_length = len(parent_dataset)
        except Exception as error:
            raise ValueError(
                "torch Subset parent dataset must have a valid length"
            ) from error
        if not 0 <= current_row < parent_length:
            raise ValueError(
                "torch Subset index must be within its parent dataset; "
                f"got {current_row} for length {parent_length}"
            )
        current_dataset = parent_dataset
    return current_row


def build_dose_closure_sample_plan(
    dataset: Any,
    *,
    channels: int,
) -> DoseClosureSamplePlan:
    """Select the one fixed 256-slot representative dose-closure sample."""

    if isinstance(channels, bool) or not isinstance(channels, Integral):
        raise TypeError("dose-closure channels must be a positive integer")
    channels = int(channels)
    if channels <= 0:
        raise ValueError("dose-closure channels must be a positive integer")
    try:
        logical_rows = len(dataset)
    except Exception as error:
        raise ValueError("dose-closure dataset must have a valid length") from error
    population_patterns = logical_rows * channels
    if population_patterns < RECT_S1S2_DOSE_CLOSURE_PATTERNS:
        raise ValueError(
            "dose-closure population must contain at least "
            f"{RECT_S1S2_DOSE_CLOSURE_PATTERNS} detector-pattern slots"
        )
    if population_patterns > _UINT64_MODULUS:
        raise ValueError("dose-closure population must not exceed 2**64 slots")

    flat_slots = _draw_flat_slots(population_patterns)
    channels_by_logical_row: dict[int, list[int]] = {}
    for flat_slot in flat_slots:
        logical_row, channel = divmod(flat_slot, channels)
        channels_by_logical_row.setdefault(logical_row, []).append(channel)

    access_rows = tuple(
        sorted(
            (
                SelectedDoseClosureRow(
                    logical_row=logical_row,
                    base_row=_base_row_for_logical(dataset, logical_row),
                    channels=tuple(sorted(selected_channels)),
                )
                for logical_row, selected_channels in channels_by_logical_row.items()
            ),
            key=lambda row: (row.base_row, row.logical_row),
        )
    )
    return DoseClosureSamplePlan(
        population_patterns=population_patterns,
        flat_slots=flat_slots,
        access_rows=access_rows,
    )


__all__ = [
    "DoseClosureSamplePlan",
    "SelectedDoseClosureRow",
    "build_dose_closure_sample_plan",
]
