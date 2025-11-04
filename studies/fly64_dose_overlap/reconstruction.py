"""
Phase F: PtyChi LSQML reconstruction job builder.

This module constructs a manifest of reconstruction jobs for pty-chi LSQML
baseline comparisons against PtychoPINN trained models. Each job targets
Phase C baseline datasets (gs1) or Phase D overlap views (gs2 dense/sparse).

Job manifest includes CLI arguments for scripts/reconstruction/ptychi_reconstruct_tike.py
with algorithm='LSQML', num_epochs=100 (baseline; parameterizable), and
artifact paths derived from dose/view/split combinations.

CONFIG-001 compliance: This builder remains pure (no params.cfg mutation).
The CONFIG-001 bridge is deferred to the actual LSQML runner invocation in Phase F2.

DATA-001 compliance: Builder validates NPZ paths against Phase C/D outputs
and assumes canonical contract (amplitude diffraction, complex64 Y patches).

POLICY-001 note: Pty-chi uses PyTorch internally, which is acceptable per
study design. No PtychoPINN backend switch is required.

OVERSAMPLING-001 note: Reconstruction jobs inherit neighbor_count=7 from
Phase D/E artifacts. No additional K≥C validation is needed in the builder.
"""

from pathlib import Path
from typing import List
from dataclasses import dataclass, field
import subprocess
from enum import Enum

from studies.fly64_dose_overlap.design import get_study_design


class ViewType(str, Enum):
    """Overlap view types for reconstruction jobs."""
    BASELINE = "baseline"
    DENSE = "dense"
    SPARSE = "sparse"


@dataclass
class ReconstructionJob:
    """
    Single pty-chi LSQML reconstruction job specification.

    Attributes:
        dose: Photon dose (photons per exposure)
        view: View type ('baseline', 'dense', or 'sparse')
        split: Data split ('train' or 'test')
        input_npz: Path to input NPZ (Phase C baseline or Phase D overlap)
        output_dir: Artifact directory for LSQML outputs
        algorithm: Reconstruction algorithm ('LSQML')
        num_epochs: Number of LSQML iterations
        cli_args: Assembled arguments for scripts/reconstruction/ptychi_reconstruct_tike.py
    """
    dose: float
    view: str
    split: str
    input_npz: Path
    output_dir: Path
    algorithm: str = "LSQML"
    num_epochs: int = 100
    cli_args: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Validate and assemble CLI arguments."""
        # Ensure paths are Path objects
        self.input_npz = Path(self.input_npz)
        self.output_dir = Path(self.output_dir)

        # Assemble CLI arguments if not provided
        if not self.cli_args:
            self.cli_args = [
                "python",
                "scripts/reconstruction/ptychi_reconstruct_tike.py",
                "--algorithm", self.algorithm,
                "--num-epochs", str(self.num_epochs),
                "--input-npz", str(self.input_npz),
                "--output-dir", str(self.output_dir),
            ]


def build_ptychi_jobs(
    phase_c_root: Path,
    phase_d_root: Path,
    artifact_root: Path,
    allow_missing: bool = False,
) -> List[ReconstructionJob]:
    """
    Enumerate pty-chi LSQML reconstruction jobs for study doses and views.

    Expected manifest structure:
    - 3 doses × 2 views (dense, sparse) + 1 baseline per dose = 7 jobs per dose
    - Total: 21 jobs for the full study (3 doses × 7 jobs)

    Each job contains:
    - dose (float): photon dose from StudyDesign.dose_list
    - view (str): 'baseline', 'dense', or 'sparse'
    - split (str): 'train' or 'test'
    - input_npz (Path): Phase C baseline or Phase D overlap NPZ
    - output_dir (Path): artifact directory for LSQML outputs
    - algorithm (str): 'LSQML'
    - num_epochs (int): 100 (baseline; parameterizable in future)
    - cli_args (list): arguments for scripts/reconstruction/ptychi_reconstruct_tike.py

    Args:
        phase_c_root: Root directory for Phase C baseline datasets
                      (dose_{dose}/patched_{split}.npz)
        phase_d_root: Root directory for Phase D overlap views
                      (dose_{dose}/{view}/{view}_{split}.npz)
        artifact_root: Root directory for reconstruction outputs
                       (phase_f_reconstruction/dose_{dose}/{view}/{split}/)
        allow_missing: If False, raise FileNotFoundError for missing NPZ files

    Returns:
        List of ReconstructionJob dataclasses

    Raises:
        FileNotFoundError: If allow_missing=False and input NPZ does not exist
    """
    design = get_study_design()
    jobs = []

    # Deterministic ordering: doses ascending, views (baseline→dense→sparse), splits (train→test)
    for dose in sorted(design.dose_list):
        dose_int = int(dose)

        # Process each split
        for split in ["train", "test"]:
            # Baseline jobs (Phase C patched datasets, gs=1)
            phase_c_dose_dir = phase_c_root / f"dose_{dose_int}"
            baseline_npz = phase_c_dose_dir / f"patched_{split}.npz"

            if not allow_missing and not baseline_npz.exists():
                raise FileNotFoundError(
                    f"Phase C baseline NPZ not found: {baseline_npz}. "
                    f"Expected from Phase C patching."
                )

            baseline_output = artifact_root / f"dose_{dose_int}" / "baseline" / split
            jobs.append(ReconstructionJob(
                dose=dose,
                view=ViewType.BASELINE.value,
                split=split,
                input_npz=baseline_npz,
                output_dir=baseline_output,
            ))

            # Overlap view jobs (Phase D datasets, gs=2)
            phase_d_dose_dir = phase_d_root / f"dose_{dose_int}"
            for view in ["dense", "sparse"]:
                view_dir = phase_d_dose_dir / view
                overlap_npz = view_dir / f"{view}_{split}.npz"

                if not allow_missing and not overlap_npz.exists():
                    raise FileNotFoundError(
                        f"Phase D overlap NPZ not found: {overlap_npz}. "
                        f"Expected from Phase D overlap filtering."
                    )

                overlap_output = artifact_root / f"dose_{dose_int}" / view / split
                jobs.append(ReconstructionJob(
                    dose=dose,
                    view=view,
                    split=split,
                    input_npz=overlap_npz,
                    output_dir=overlap_output,
                ))

    return jobs


def run_ptychi_job(job: ReconstructionJob, dry_run: bool = True) -> subprocess.CompletedProcess:
    """
    Execute a single pty-chi LSQML reconstruction job.

    Args:
        job: ReconstructionJob to execute
        dry_run: If True, skip actual execution and return mock result

    Returns:
        subprocess.CompletedProcess with execution results

    Note:
        CONFIG-001 bridge (update_legacy_dict) is handled by the reconstruction
        script itself. This runner simply dispatches the subprocess.
    """
    if dry_run:
        # Return mock result without executing
        return subprocess.CompletedProcess(
            args=job.cli_args,
            returncode=0,
            stdout=f"[DRY RUN] Would execute: {' '.join(job.cli_args)}",
            stderr="",
        )

    # Ensure output directory exists
    job.output_dir.mkdir(parents=True, exist_ok=True)

    # Execute reconstruction script
    result = subprocess.run(
        job.cli_args,
        capture_output=True,
        text=True,
        check=False,
    )

    return result
