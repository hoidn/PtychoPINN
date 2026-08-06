"""Pure selection tests for representative rectangular dose closure."""

from __future__ import annotations

from dataclasses import FrozenInstanceError
import hashlib
import inspect
import random
import struct

import numpy as np
import pytest
import torch

from ptycho_torch.rect_s1s2_sampling import (
    _map_bounded_candidate,
    build_dose_closure_sample_plan,
)
from ptycho_torch.rect_s1s2_initialization import (
    RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    RECT_S1S2_DOSE_CLOSURE_SAMPLE_POLICY,
    RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED,
)


class _IdentityDataset(torch.utils.data.Dataset):
    def __init__(self, length: int):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        raise AssertionError("selection must not read or enumerate dataset rows")


def _assert_numpy_random_states_equal(before, after):
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def test_sampling_identity_and_public_plan_api_are_fixed():
    signature = inspect.signature(build_dose_closure_sample_plan)

    assert RECT_S1S2_DOSE_CLOSURE_PATTERNS == 256
    assert RECT_S1S2_DOSE_CLOSURE_SAMPLE_SEED == 20260806
    assert RECT_S1S2_DOSE_CLOSURE_SAMPLE_POLICY == "splitmix64_rejection_v1"
    assert tuple(signature.parameters) == ("dataset", "channels")
    assert signature.parameters["channels"].kind is inspect.Parameter.KEYWORD_ONLY


def test_v2_selection_vector_is_pinned_and_plan_is_immutable():
    plan = build_dose_closure_sample_plan(
        dataset=_IdentityDataset(1024),
        channels=1,
    )
    encoded = b"".join(struct.pack("<Q", value) for value in plan.flat_slots)

    assert plan.population_patterns == 1024
    assert plan.flat_slots[:8] == (705, 359, 847, 532, 312, 814, 888, 27)
    assert hashlib.sha256(encoded).hexdigest() == (
        "6c88290b749ff9ba972a2adeafce93985cb90c91ce3e682fe8f30add20f6e8f1"
    )
    assert len(plan.flat_slots) == len(set(plan.flat_slots)) == 256
    assert not hasattr(plan, "__dict__")
    assert all(not hasattr(row, "__dict__") for row in plan.access_rows)
    with pytest.raises(FrozenInstanceError):
        plan.population_patterns = 1


def test_non_power_of_two_candidate_mapping_rejects_biased_tail():
    bound = 10
    limit = 2**64 - (2**64 % bound)

    assert _map_bounded_candidate(limit - 1, bound) == (limit - 1) % bound
    assert _map_bounded_candidate(limit, bound) is None
    assert _map_bounded_candidate(2**64 - 1, bound) is None


def test_grouped_channels_preserve_every_flat_slot_without_truncation():
    channels = 9
    plan = build_dose_closure_sample_plan(
        dataset=_IdentityDataset(1024),
        channels=channels,
    )
    expected_by_logical_row: dict[int, set[int]] = {}

    for flat_slot in plan.flat_slots:
        logical_row, channel = divmod(flat_slot, channels)
        assert flat_slot == logical_row * channels + channel
        assert 0 <= channel < channels
        expected_by_logical_row.setdefault(logical_row, set()).add(channel)

    assert plan.population_patterns == 1024 * channels
    assert len(plan.flat_slots) == len(set(plan.flat_slots)) == 256
    assert len(plan.access_rows) == len(expected_by_logical_row)
    assert plan.access_rows == tuple(
        sorted(plan.access_rows, key=lambda row: (row.base_row, row.logical_row))
    )
    assert {row.logical_row: row.channels for row in plan.access_rows} == {
        logical_row: tuple(sorted(selected_channels))
        for logical_row, selected_channels in expected_by_logical_row.items()
    }


def test_nested_subsets_sample_outer_population_and_preserve_duplicates():
    base = _IdentityDataset(400)
    inner_indices = list(range(320))
    inner_indices[0] = 91
    inner_indices[2] = 91
    inner = torch.utils.data.Subset(base, inner_indices)
    outer_indices = list(range(300))
    outer = torch.utils.data.Subset(inner, outer_indices)

    plan = build_dose_closure_sample_plan(outer, channels=2)

    assert plan.population_patterns == len(outer) * 2
    assert len(plan.flat_slots) == len(set(plan.flat_slots)) == 256
    assert all(0 <= flat_slot < len(outer) * 2 for flat_slot in plan.flat_slots)
    assert all(0 <= row.logical_row < len(outer) for row in plan.access_rows)
    assert {300, 301, 319}.isdisjoint({row.base_row for row in plan.access_rows})
    assert all(
        row.base_row == inner_indices[outer_indices[row.logical_row]]
        for row in plan.access_rows
    )
    duplicate_members = {
        row.logical_row: row for row in plan.access_rows if row.logical_row in {0, 2}
    }
    assert {
        logical_row: row.base_row for logical_row, row in duplicate_members.items()
    } == {
        0: 91,
        2: 91,
    }
    assert {
        logical_row: row.channels for logical_row, row in duplicate_members.items()
    } == {0: (1,), 2: (0,)}
    assert plan.access_rows == tuple(
        sorted(plan.access_rows, key=lambda row: (row.base_row, row.logical_row))
    )


def test_plan_construction_does_not_consume_global_rng_state():
    python_original = random.getstate()
    numpy_original = np.random.get_state()
    torch_original = torch.random.get_rng_state()
    try:
        random.seed(104729)
        np.random.seed(104729)
        torch.manual_seed(104729)
        python_before = random.getstate()
        numpy_before = np.random.get_state()
        torch_before = torch.random.get_rng_state().clone()

        build_dose_closure_sample_plan(_IdentityDataset(1024), channels=9)

        assert random.getstate() == python_before
        _assert_numpy_random_states_equal(numpy_before, np.random.get_state())
        assert torch.equal(torch.random.get_rng_state(), torch_before)
    finally:
        random.setstate(python_original)
        np.random.set_state(numpy_original)
        torch.random.set_rng_state(torch_original)


@pytest.mark.parametrize("length", [0, 255])
def test_empty_or_too_small_population_is_rejected(length):
    with pytest.raises(ValueError, match="at least 256"):
        build_dose_closure_sample_plan(_IdentityDataset(length), channels=1)


@pytest.mark.parametrize("channels", [True, 0, -1, 1.5, "1"])
def test_invalid_channel_counts_are_rejected_clearly(channels):
    with pytest.raises((TypeError, ValueError), match="channels"):
        build_dose_closure_sample_plan(_IdentityDataset(300), channels=channels)


@pytest.mark.parametrize(
    "bad_index",
    [object(), True, -1, 300],
    ids=("unsupported", "boolean", "negative", "past-end"),
)
def test_broken_subset_indices_are_rejected_clearly(bad_index):
    subset = torch.utils.data.Subset(
        _IdentityDataset(300),
        [bad_index] * RECT_S1S2_DOSE_CLOSURE_PATTERNS,
    )

    with pytest.raises((TypeError, ValueError), match=r"Subset.*index"):
        build_dose_closure_sample_plan(subset, channels=1)
