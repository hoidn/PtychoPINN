"""Task specifications for the PDEBench 128x128 image-suite plan."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    """Static identity and expected contract for one PDEBench image task."""

    task_id: str
    pde_name: str
    title: str
    expected_filename: str
    task_type: str
    expected_spatial_shape: tuple[int, int] = (128, 128)
    expected_darus_id: str | None = None
    listed_size_gb: float | None = None
    expected_md5: str | None = None


_TASK_SPECS: dict[str, TaskSpec] = {
    "swe": TaskSpec(
        task_id="swe",
        pde_name="swe",
        title="2D Shallow Water Equations",
        expected_filename="2D_rdb_NA_NA.h5",
        task_type="dynamic_one_step",
        expected_darus_id="133021",
        listed_size_gb=6.2,
    ),
    "darcy": TaskSpec(
        task_id="darcy",
        pde_name="darcy",
        title="2D Darcy Flow",
        expected_filename="2D_DarcyFlow_beta1.0_Train.hdf5",
        task_type="static_operator",
        expected_darus_id="133219",
        listed_size_gb=6.2,
    ),
    "2d_cfd_cns": TaskSpec(
        task_id="2d_cfd_cns",
        pde_name="2d_cfd",
        title="2D Compressible Navier-Stokes",
        expected_filename="2D_CFD_Rand_M1.0_Eta0.01_Zeta0.01_periodic_128_Train.hdf5",
        task_type="dynamic_one_step",
        expected_darus_id="164690",
        listed_size_gb=55.050245208,
        expected_md5="21969082d0e9524bcc4708e216148e60",
    ),
}

TASK_IDS = ("swe", "darcy", "2d_cfd_cns")
TASK_SPECS = {task_id: _TASK_SPECS[task_id] for task_id in TASK_IDS}


def get_task_spec(task_id: str) -> TaskSpec:
    """Return a task spec by stable task id."""
    try:
        return TASK_SPECS[task_id]
    except KeyError as exc:
        supported = ", ".join(TASK_IDS)
        raise ValueError(f"unknown PDEBench image128 task_id {task_id!r}; supported: {supported}") from exc
