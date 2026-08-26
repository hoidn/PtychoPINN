"""Owner-level contracts for backend-neutral acquisition grouping plans."""

from __future__ import annotations

import numpy as np
import pytest


def _grid(side: int, *, spacing: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    axis = np.arange(side, dtype=np.float64) * spacing
    xcoords, ycoords = np.meshgrid(axis, axis)
    return xcoords.reshape(-1), ycoords.reshape(-1)


def _assert_ambient_rng_equal(before, after) -> None:
    assert before[0] == after[0]
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]


def _k_pool_threshold(
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    center: int,
    candidates: np.ndarray,
    neighbor_count: int,
) -> float:
    """Distance of the K-th nearest non-center candidate, any tie order."""
    pool = candidates[candidates != center]
    distances = np.hypot(
        xcoords[pool] - xcoords[center], ycoords[pool] - ycoords[center]
    )
    return float(np.sort(distances)[neighbor_count - 1])


def test_nearest_groups_center_first_repeats_and_k_pool():
    from ptycho.grouping import plan_nearest_groups

    xcoords, ycoords = _grid(3)
    candidates = np.arange(6, dtype=np.int64)
    plan = plan_nearest_groups(
        xcoords,
        ycoords,
        center_indices=np.array([4, 1]),
        candidate_indices=candidates,
        group_size=3,
        neighbor_count=4,
        repeats=2,
        seed=7,
    )

    np.testing.assert_array_equal(plan.center_indices, [4, 4, 1, 1])
    np.testing.assert_array_equal(plan.neighbor_indices[:, 0], plan.center_indices)
    assert all(len(np.unique(row)) == 3 for row in plan.neighbor_indices)
    for row in plan.neighbor_indices:
        threshold = _k_pool_threshold(
            xcoords, ycoords, int(row[0]), candidates, 4
        )
        distances = np.hypot(
            xcoords[row[1:]] - xcoords[row[0]], ycoords[row[1:]] - ycoords[row[0]]
        )
        assert np.all(distances <= threshold)


def test_nearest_groups_duplicate_coordinates_remove_center_by_identity():
    from ptycho.grouping import plan_nearest_groups

    # Rows 0, 1, and 2 share the coordinate (0, 0); the center is row 1, so
    # the center row is not necessarily the first coordinate match.
    xcoords = np.array([0.0, 0.0, 0.0, 1.0, 2.0])
    ycoords = np.zeros(5, dtype=np.float64)
    plan = plan_nearest_groups(
        xcoords,
        ycoords,
        center_indices=np.array([1]),
        candidate_indices=np.arange(5),
        group_size=3,
        neighbor_count=2,
        repeats=1,
        seed=7,
    )

    np.testing.assert_array_equal(plan.center_indices, [1])
    np.testing.assert_array_equal(plan.neighbor_indices[:, 0], [1])
    # Only the center identity (row 1) leaves the non-center pool: its
    # coordinate twin (row 0) stays eligible and is drawn, while the center
    # row appears exactly once, at column zero.
    assert set(plan.neighbor_indices[0]) == {0, 1, 2}
    assert len(np.unique(plan.neighbor_indices[0])) == 3


def test_nearest_groups_partition_by_object_and_carry_source_identity():
    from ptycho.grouping import plan_nearest_groups

    xcoords = np.tile(np.arange(6, dtype=np.float64), 2)
    ycoords = np.zeros(12, dtype=np.float64)
    object_index = np.tile(np.arange(2, dtype=np.int64), 6)
    experiment_id = np.arange(12, dtype=np.int64) + 100
    source_indices = np.arange(12, dtype=np.int64) + 7

    plan = plan_nearest_groups(
        xcoords,
        ycoords,
        center_indices=np.array([0, 6]),
        candidate_indices=np.arange(12),
        object_index=object_index,
        experiment_id=experiment_id,
        source_indices=source_indices,
        group_size=3,
        neighbor_count=4,
        repeats=2,
        seed=17,
    )

    assert plan.neighbor_indices.shape == (4, 3)
    np.testing.assert_array_equal(plan.center_indices, [0, 0, 6, 6])
    np.testing.assert_array_equal(plan.neighbor_indices[:, 0], plan.center_indices)
    assert all(np.unique(object_index[row]).size == 1 for row in plan.neighbor_indices)
    np.testing.assert_array_equal(plan.object_index, object_index[plan.center_indices])
    np.testing.assert_array_equal(
        plan.experiment_id, experiment_id[plan.center_indices]
    )
    np.testing.assert_array_equal(plan.source_indices, source_indices)


def test_nearest_groups_c1_is_ordered_identity_without_neighbor_draw():
    from ptycho.grouping import plan_nearest_groups

    xcoords, ycoords = _grid(3)
    generator = np.random.default_rng(20260824)
    state_before = generator.bit_generator.state

    plan = plan_nearest_groups(
        xcoords,
        ycoords,
        center_indices=np.array([4, 1]),
        candidate_indices=np.arange(len(xcoords)),
        group_size=1,
        neighbor_count=1,
        repeats=2,
        rng=generator,
    )

    # C=1 emits each ordered center as a one-column row without any draw, so
    # the supplied generator state is untouched.
    assert generator.bit_generator.state == state_before
    np.testing.assert_array_equal(plan.center_indices, [4, 4, 1, 1])
    np.testing.assert_array_equal(plan.neighbor_indices, [[4], [4], [1], [1]])


def test_nearest_groups_seed_and_generator_are_reproducible_without_ambient_state():
    from ptycho.grouping import plan_nearest_groups

    xcoords, ycoords = _grid(3)
    common = dict(
        center_indices=np.array([4, 1]),
        candidate_indices=np.arange(len(xcoords)),
        group_size=3,
        neighbor_count=4,
        repeats=2,
    )

    np.random.seed(20260824)
    state_before = np.random.get_state()
    seeded_a = plan_nearest_groups(xcoords, ycoords, **common, seed=7)
    _assert_ambient_rng_equal(state_before, np.random.get_state())
    seeded_b = plan_nearest_groups(xcoords, ycoords, **common, seed=7)
    np.testing.assert_array_equal(seeded_a.neighbor_indices, seeded_b.neighbor_indices)

    generator_a = plan_nearest_groups(
        xcoords, ycoords, **common, rng=np.random.default_rng(7)
    )
    generator_b = plan_nearest_groups(
        xcoords, ycoords, **common, rng=np.random.default_rng(7)
    )
    np.testing.assert_array_equal(
        generator_a.neighbor_indices, generator_b.neighbor_indices
    )
    np.testing.assert_array_equal(seeded_a.neighbor_indices, generator_a.neighbor_indices)


@pytest.mark.parametrize(
    ("override", "match"),
    [
        (
            {"center_indices": np.array([3]), "candidate_indices": np.array([0, 1, 2])},
            "subset of candidate_indices",
        ),
        ({"center_indices": np.array([1, 1])}, "center_indices"),
        ({"candidate_indices": np.array([1, 1, 2, 3, 4, 5])}, "candidate_indices"),
        (
            {"center_indices": np.array([True, False, False, False, False, False])},
            "center_indices",
        ),
        ({"candidate_indices": np.array([0.0, 1.0, 2.0])}, "candidate_indices"),
        ({"candidate_indices": np.array([0, 99])}, "candidate_indices"),
        ({"center_indices": np.array([99])}, "center_indices"),
        ({"group_size": 0}, "C=0"),
        ({"repeats": 0}, "repeats=0"),
        ({"group_size": 3, "neighbor_count": 1}, "K=1 must be at least C-1=2"),
        ({"seed": 7, "rng": np.random.default_rng(7)}, "mutually exclusive"),
    ],
    ids=[
        "center-outside-pool",
        "duplicate-center",
        "duplicate-candidates",
        "boolean-mask-center",
        "float-candidates",
        "candidate-out-of-range",
        "center-out-of-range",
        "c-group-zero",
        "repeats-zero",
        "k-below-c-minus-one",
        "seed-and-rng",
    ],
)
def test_nearest_groups_reject_invalid_indices_counts_partitions_and_rng(
    override, match
):
    from ptycho.grouping import plan_nearest_groups

    values = dict(
        xcoords=np.arange(6, dtype=np.float64),
        ycoords=np.zeros(6, dtype=np.float64),
        center_indices=np.array([1]),
        candidate_indices=np.arange(6),
        group_size=3,
        neighbor_count=4,
        repeats=1,
        seed=7,
    )
    values.update(override)

    with pytest.raises(ValueError, match=match):
        plan_nearest_groups(**values)


def test_nearest_groups_reject_starved_object_partition():
    from ptycho.grouping import plan_nearest_groups

    with pytest.raises(ValueError, match="object partition 0 for center 1"):
        plan_nearest_groups(
            np.arange(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            center_indices=np.array([1]),
            candidate_indices=np.array([0, 1, 2]),
            object_index=np.zeros(6, dtype=np.int64),
            group_size=4,
            neighbor_count=4,
            repeats=1,
            seed=7,
        )


def test_nearest_groups_rejects_wrong_rng_type():
    from ptycho.grouping import plan_nearest_groups

    with pytest.raises(TypeError, match="Generator"):
        plan_nearest_groups(
            np.arange(6, dtype=np.float64),
            np.zeros(6, dtype=np.float64),
            center_indices=np.array([1]),
            candidate_indices=np.arange(6),
            group_size=3,
            neighbor_count=4,
            repeats=1,
            rng=object(),
        )


def test_grouping_plan_arrays_are_immutable_and_global_rng_is_untouched():
    from ptycho.grouping import plan_nearest_groups

    xcoords, ycoords = _grid(3)
    np.random.seed(20260813)
    state_before = np.random.get_state()

    plan = plan_nearest_groups(
        xcoords,
        ycoords,
        center_indices=np.array([4, 1]),
        candidate_indices=np.arange(len(xcoords)),
        group_size=2,
        neighbor_count=4,
        repeats=1,
        seed=12,
    )

    _assert_ambient_rng_equal(state_before, np.random.get_state())
    for value in (
        plan.neighbor_indices,
        plan.center_indices,
        plan.source_indices,
        plan.object_index,
        plan.experiment_id,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        plan.neighbor_indices[0, 0] = 0


def test_grouping_plans_preserve_optional_source_index_mapping():
    from ptycho.grouping import plan_nearest_groups

    xcoords = np.arange(4, dtype=np.float64)
    ycoords = np.zeros(4, dtype=np.float64)
    source_indices = np.array([3, 8, 13, 21], dtype=np.int64)

    mapped = plan_nearest_groups(
        xcoords,
        ycoords,
        center_indices=np.array([1, 2]),
        candidate_indices=np.arange(4),
        source_indices=source_indices,
        group_size=2,
        neighbor_count=2,
        repeats=1,
        seed=7,
    )
    fallback = plan_nearest_groups(
        xcoords,
        ycoords,
        center_indices=np.array([1, 2]),
        candidate_indices=np.arange(4),
        group_size=2,
        neighbor_count=2,
        repeats=1,
        seed=7,
    )

    np.testing.assert_array_equal(mapped.source_indices, source_indices)
    np.testing.assert_array_equal(fallback.source_indices, np.arange(4))


@pytest.mark.parametrize(
    "source_indices",
    [
        np.array([1, 2, 3]),
        np.array([1, -2, 3, 4]),
        np.array([1, 2, 2, 4]),
        np.array([1.0, 2.0, 3.0, 4.0]),
    ],
    ids=["wrong-length", "negative", "duplicate", "noninteger"],
)
def test_nearest_groups_rejects_invalid_source_index_mapping(source_indices):
    from ptycho.grouping import plan_nearest_groups

    with pytest.raises(ValueError, match="source_indices"):
        plan_nearest_groups(
            np.arange(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            center_indices=np.array([1, 2]),
            candidate_indices=np.arange(4),
            source_indices=source_indices,
            group_size=2,
            neighbor_count=2,
            repeats=1,
            seed=7,
        )


def test_grouping_plan_has_only_immutable_consumed_fields_and_rejects_invalid_rows():
    """Pin the final five-field GroupingPlan surface and its row validation."""
    import dataclasses

    from ptycho.grouping import GroupingPlan

    assert [field.name for field in dataclasses.fields(GroupingPlan)] == [
        "neighbor_indices",
        "center_indices",
        "source_indices",
        "object_index",
        "experiment_id",
    ]

    plan = GroupingPlan(
        neighbor_indices=np.array([[0, 1], [1, 2]], dtype=np.int64),
        center_indices=np.array([0, 1], dtype=np.int64),
        source_indices=np.arange(3, dtype=np.int64),
        object_index=np.zeros(2, dtype=np.int64),
        experiment_id=np.zeros(2, dtype=np.int64),
    )
    for value in (
        plan.neighbor_indices,
        plan.center_indices,
        plan.source_indices,
        plan.object_index,
        plan.experiment_id,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        plan.neighbor_indices[0, 0] = 9

    values = {
        "neighbor_indices": np.array([[0]]),
        "center_indices": np.array([0]),
        "source_indices": np.array([10, 20]),
        "object_index": np.array([0]),
        "experiment_id": np.array([7]),
    }
    invalid_rows = [
        ({"neighbor_indices": np.array([[-1]])}, "neighbor_indices"),
        ({"neighbor_indices": np.array([[2]])}, "neighbor_indices"),
        ({"center_indices": np.array([2])}, "center_indices"),
        ({"source_indices": np.array([10, -20])}, "source_indices"),
        ({"source_indices": np.array([10, 10])}, "source_indices"),
        ({"object_index": np.array([-1])}, "object_index"),
        ({"experiment_id": np.array([-1])}, "experiment_id"),
    ]
    for override, match in invalid_rows:
        with pytest.raises(ValueError, match=match):
            GroupingPlan(**{**values, **override})
