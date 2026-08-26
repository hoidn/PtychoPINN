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


def test_sample_then_group_matches_current_seeded_and_sequential_rows():
    from ptycho.grouping import plan_sample_then_group

    xcoords, ycoords = _grid(5)
    object_index = np.zeros(len(xcoords), dtype=np.int64)

    seeded = plan_sample_then_group(
        xcoords,
        ycoords,
        object_index=object_index,
        count=6,
        neighbor_count=7,
        group_size=4,
        seed=17,
    )
    sequential_a = plan_sample_then_group(
        xcoords,
        ycoords,
        object_index=object_index,
        count=6,
        neighbor_count=7,
        group_size=4,
        sequential=True,
        seed=17,
    )
    sequential_b = plan_sample_then_group(
        xcoords,
        ycoords,
        object_index=object_index,
        count=6,
        neighbor_count=7,
        group_size=4,
        sequential=True,
        seed=999,
    )

    np.testing.assert_array_equal(
        seeded.neighbor_indices,
        [
            [17, 9, 18, 14],
            [8, 1, 3, 6],
            [1, 8, 9, 7],
            [4, 12, 19, 18],
            [18, 23, 13, 22],
            [15, 0, 16, 11],
        ],
    )
    np.testing.assert_array_equal(
        sequential_a.neighbor_indices,
        [
            [1, 7, 10, 2],
            [6, 3, 7, 11],
            [6, 8, 0, 12],
            [8, 9, 7, 2],
            [2, 3, 13, 14],
            [6, 0, 11, 10],
        ],
    )
    np.testing.assert_array_equal(
        sequential_a.neighbor_indices, sequential_b.neighbor_indices
    )
    np.testing.assert_array_equal(sequential_a.center_indices, np.arange(6))
    assert seeded.policy == "raw_random_sample_then_group"
    assert sequential_a.policy == "raw_sequential_sample_then_group"


def test_sample_then_group_caps_c1_and_requires_oversampling_opt_in():
    from ptycho.grouping import plan_sample_then_group

    xcoords = np.arange(5, dtype=np.float64)
    ycoords = np.zeros(5, dtype=np.float64)
    object_index = np.zeros(5, dtype=np.int64)

    capped = plan_sample_then_group(
        xcoords,
        ycoords,
        object_index=object_index,
        count=8,
        neighbor_count=4,
        group_size=1,
        seed=3,
    )
    np.testing.assert_array_equal(capped.neighbor_indices[:, 0], np.arange(5))

    with pytest.raises(ValueError, match="K choose C oversampling.*not enabled"):
        plan_sample_then_group(
            xcoords,
            ycoords,
            object_index=object_index,
            count=8,
            neighbor_count=4,
            group_size=2,
            seed=3,
        )

    oversampled = plan_sample_then_group(
        xcoords,
        ycoords,
        object_index=object_index,
        count=8,
        neighbor_count=4,
        group_size=2,
        seed=3,
        enable_oversampling=True,
    )
    np.testing.assert_array_equal(
        oversampled.neighbor_indices,
        [[3, 1], [2, 1], [2, 0], [1, 4], [1, 0], [1, 3], [2, 4], [3, 2]],
    )
    assert oversampled.policy == "raw_k_choose_c_oversampling"


def test_sample_then_group_partitions_only_by_object_identity():
    from ptycho.grouping import plan_sample_then_group

    xcoords = np.tile(np.arange(6, dtype=np.float64), 2)
    ycoords = np.zeros(12, dtype=np.float64)
    object_index = np.tile(np.arange(2, dtype=np.int64), 6)
    experiment_id = np.arange(12, dtype=np.int64) + 100

    plan = plan_sample_then_group(
        xcoords,
        ycoords,
        object_index=object_index,
        experiment_id=experiment_id,
        count=12,
        neighbor_count=4,
        group_size=4,
        seed=17,
    )

    assert all(np.unique(object_index[row]).size == 1 for row in plan.neighbor_indices)
    np.testing.assert_array_equal(plan.object_index, object_index[plan.center_indices])
    np.testing.assert_array_equal(
        plan.experiment_id, experiment_id[plan.center_indices]
    )
    assert any(
        np.unique(experiment_id[row]).size > 1 for row in plan.neighbor_indices
    )


@pytest.mark.parametrize("policy", ["Nearest", "Min_dist"])
def test_scan_centered_matches_current_rows_and_carries_identity(policy):
    from ptycho.grouping import plan_scan_centered

    xcoords, ycoords = _grid(3)
    eligible = np.array([1, 3, 4, 5, 7], dtype=np.int64)
    object_index = np.zeros(len(xcoords), dtype=np.int64)
    experiment_id = np.arange(len(xcoords), dtype=np.int64) + 40

    plan = plan_scan_centered(
        xcoords,
        ycoords,
        eligible_indices=eligible,
        object_index=object_index,
        experiment_id=experiment_id,
        policy=policy,
        group_size=2,
        neighbor_count=4,
        repeats=2,
        seed=23,
        min_neighbor_distance=0.5,
        max_neighbor_distance=1.5,
    )

    expected = {
        "Nearest": [
            [5, 1], [4, 3], [3, 1], [1, 7], [1, 3],
            [4, 1], [7, 1], [7, 4], [7, 4], [7, 4],
        ],
        "Min_dist": [
            [3, 4], [5, 4], [4, 4], [4, 1], [1, 3],
            [1, 3], [1, 4], [1, 4], [4, 4], [4, 4],
        ],
    }
    np.testing.assert_array_equal(plan.neighbor_indices, expected[policy])
    np.testing.assert_array_equal(plan.center_indices, np.repeat(eligible, 2))
    np.testing.assert_array_equal(plan.center_available, np.ones(10, dtype=bool))
    np.testing.assert_array_equal(plan.object_index, np.zeros(10, dtype=np.int64))
    np.testing.assert_array_equal(
        plan.experiment_id, np.repeat(experiment_id[eligible], 2)
    )
    assert plan.policy == policy


def test_scan_centered_omits_min_dist_rows_with_invalid_candidates():
    from ptycho.grouping import plan_scan_centered

    plan = plan_scan_centered(
        np.array([0.0, 10.0, 20.0]),
        np.zeros(3),
        eligible_indices=np.arange(3),
        object_index=np.zeros(3, dtype=np.int64),
        policy="Min_dist",
        group_size=3,
        neighbor_count=3,
        repeats=1,
        seed=7,
        min_neighbor_distance=0.0,
        max_neighbor_distance=1.0,
    )

    assert plan.neighbor_indices.shape == (0, 3)
    assert plan.center_indices.shape == (0,)
    assert not plan.coverage_complete


def test_scan_centered_quadrant_partitioning_and_seeded_parity():
    from ptycho.grouping import plan_scan_centered

    base_x, base_y = _grid(3)
    xcoords = np.concatenate([base_x, base_x])
    ycoords = np.concatenate([base_y, base_y])
    object_index = np.repeat(np.arange(2, dtype=np.int64), 9)
    experiment_id = np.arange(18, dtype=np.int64) % 3

    plan = plan_scan_centered(
        xcoords,
        ycoords,
        eligible_indices=np.arange(18),
        object_index=object_index,
        experiment_id=experiment_id,
        policy="4_quadrant",
        group_size=4,
        neighbor_count=8,
        quadrant_neighbor_count=20,
        repeats=2,
        seed=909,
        max_neighbor_distance=3.0,
        scan_pattern="Isotropic",
    )

    np.testing.assert_array_equal(plan.center_indices, [4, 4, 13, 13])
    np.testing.assert_array_equal(
        plan.neighbor_indices,
        [[6, 8, 4, 2], [6, 8, 0, 2], [15, 13, 9, 11], [15, 17, 9, 13]],
    )
    assert all(np.unique(object_index[row]).size == 1 for row in plan.neighbor_indices)
    np.testing.assert_array_equal(plan.object_index, [0, 0, 1, 1])
    np.testing.assert_array_equal(plan.experiment_id, experiment_id[plan.center_indices])


def test_scan_centered_quadrant_allows_same_object_candidates_outside_bounds():
    from ptycho.grouping import plan_scan_centered

    plan = plan_scan_centered(
        np.array([0.0, -1.0, 1.0, -1.0, 1.0]),
        np.array([0.0, 1.0, 1.0, -1.0, -1.0]),
        eligible_indices=np.array([0], dtype=np.int64),
        object_index=np.zeros(5, dtype=np.int64),
        policy="4_quadrant",
        group_size=4,
        neighbor_count=4,
        quadrant_neighbor_count=5,
        repeats=1,
        seed=7,
        max_neighbor_distance=3.0,
        scan_pattern="Isotropic",
    )

    assert plan.neighbor_indices.shape == (1, 4)
    np.testing.assert_array_equal(plan.center_indices, [0])
    assert set(plan.neighbor_indices[0]).issubset({0, 1, 2, 3, 4})
    assert any(index != 0 for index in plan.neighbor_indices[0])


def test_scan_centered_nearest_partitions_interleaved_object_banks():
    from ptycho.grouping import plan_scan_centered

    xcoords = np.tile(np.arange(5, dtype=np.float64), 2)
    ycoords = np.zeros(10, dtype=np.float64)
    object_index = np.tile(np.arange(2, dtype=np.int64), 5)

    plan = plan_scan_centered(
        xcoords,
        ycoords,
        eligible_indices=np.arange(10),
        object_index=object_index,
        experiment_id=np.arange(10, dtype=np.int64),
        policy="Nearest",
        group_size=4,
        neighbor_count=4,
        repeats=1,
        seed=5,
    )

    assert all(np.unique(object_index[row]).size == 1 for row in plan.neighbor_indices)
    assert any(
        np.unique(plan.experiment_id[index : index + 1]).size == 1
        and np.unique(plan.neighbor_indices[index]).size == 4
        for index in range(len(plan.neighbor_indices))
    )


def test_scan_centered_nearest_repairs_complete_participant_coverage():
    from ptycho.grouping import plan_scan_centered

    coordinate_rng = np.random.default_rng(118549108)
    xcoords = coordinate_rng.uniform(64.0, 328.0, size=1024)
    ycoords = coordinate_rng.uniform(64.0, 328.0, size=1024)
    eligible = np.arange(len(xcoords), dtype=np.int64)
    common = dict(
        eligible_indices=eligible,
        object_index=np.zeros(len(xcoords), dtype=np.int64),
        policy="Nearest",
        group_size=4,
        neighbor_count=4,
        repeats=1,
        seed=523213049,
    )

    original = plan_scan_centered(xcoords, ycoords, **common)
    repaired = plan_scan_centered(
        xcoords, ycoords, **common, ensure_complete_coverage=True
    )

    assert not original.coverage_complete
    assert repaired.coverage_complete
    assert set(repaired.neighbor_indices.reshape(-1)) == set(eligible)
    np.testing.assert_array_equal(repaired.center_indices, eligible)


def test_grouping_plan_arrays_are_immutable_and_global_rng_is_untouched():
    from ptycho.grouping import plan_scan_centered

    xcoords, ycoords = _grid(3)
    np.random.seed(20260813)
    state_before = np.random.get_state()

    plan = plan_scan_centered(
        xcoords,
        ycoords,
        eligible_indices=np.arange(len(xcoords)),
        object_index=np.zeros(len(xcoords), dtype=np.int64),
        policy="Nearest",
        group_size=2,
        neighbor_count=4,
        repeats=1,
        seed=12,
    )

    _assert_ambient_rng_equal(state_before, np.random.get_state())
    for value in (
        plan.neighbor_indices,
        plan.center_indices,
        plan.center_available,
        plan.eligible_indices,
        plan.source_indices,
        plan.object_index,
        plan.experiment_id,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        plan.neighbor_indices[0, 0] = 0


def test_grouping_plans_preserve_optional_source_index_mapping():
    from ptycho.grouping import plan_sample_then_group, plan_scan_centered

    xcoords = np.arange(4, dtype=np.float64)
    ycoords = np.zeros(4, dtype=np.float64)
    source_indices = np.array([3, 8, 13, 21], dtype=np.int64)

    sample_plan = plan_sample_then_group(
        xcoords,
        ycoords,
        source_indices=source_indices,
        count=4,
        neighbor_count=1,
        group_size=1,
        sequential=True,
    )
    scan_plan = plan_scan_centered(
        xcoords,
        ycoords,
        source_indices=source_indices,
        eligible_indices=np.arange(4),
        policy="Nearest",
        group_size=1,
        neighbor_count=1,
    )
    fallback = plan_sample_then_group(
        xcoords,
        ycoords,
        count=4,
        neighbor_count=1,
        group_size=1,
        sequential=True,
    )

    np.testing.assert_array_equal(sample_plan.source_indices, source_indices)
    np.testing.assert_array_equal(scan_plan.source_indices, source_indices)
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
def test_grouping_owner_rejects_invalid_source_index_mapping(source_indices):
    from ptycho.grouping import plan_sample_then_group

    with pytest.raises(ValueError, match="source_indices"):
        plan_sample_then_group(
            np.arange(4, dtype=np.float64),
            np.zeros(4, dtype=np.float64),
            source_indices=source_indices,
            count=4,
            neighbor_count=1,
            group_size=1,
            sequential=True,
        )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"neighbor_indices": np.array([[-1]])}, "neighbor_indices"),
        ({"neighbor_indices": np.array([[2]])}, "neighbor_indices"),
        ({"center_indices": np.array([2])}, "center_indices"),
        ({"eligible_indices": np.array([0, 2])}, "eligible_indices"),
        ({"source_indices": np.array([10, -20])}, "source_indices"),
        ({"source_indices": np.array([10, 10])}, "source_indices"),
        ({"object_index": np.array([-1])}, "object_index"),
        ({"experiment_id": np.array([-1])}, "experiment_id"),
    ],
)
def test_grouping_plan_rejects_invalid_public_indices(override, match):
    from ptycho.grouping import GroupingPlan

    values = {
        "neighbor_indices": np.array([[0]]),
        "center_indices": np.array([0]),
        "center_available": np.array([True]),
        "eligible_indices": np.array([0, 1]),
        "source_indices": np.array([10, 20]),
        "object_index": np.array([0]),
        "experiment_id": np.array([7]),
        "policy": "test",
        "coverage_complete": False,
    }
    values.update(override)

    with pytest.raises(ValueError, match=match):
        GroupingPlan(**values)


def test_grouping_owner_rejects_cross_object_rows(monkeypatch):
    from ptycho import grouping

    monkeypatch.setattr(
        grouping,
        "_sample_then_group",
        lambda *_args, **_kwargs: (
            np.array([[0, 1]], dtype=np.int64),
            np.array([0], dtype=np.int64),
        ),
    )

    with pytest.raises(ValueError, match="object_index partition"):
        grouping.plan_sample_then_group(
            np.array([0.0, 1.0]),
            np.zeros(2),
            object_index=np.array([0, 1]),
            count=1,
            neighbor_count=2,
            group_size=2,
            seed=3,
        )
