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


def build_ptychi_jobs(
    phase_c_root: Path,
    phase_d_root: Path,
    artifact_root: Path,
) -> List:
    """
    Enumerate pty-chi LSQML reconstruction jobs for study doses and views.

    Expected manifest structure:
    - 3 doses × 2 views (dense, sparse) + 1 baseline per dose = 7 jobs per dose
    - Total: 21 jobs for the full study

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

    Returns:
        List of ReconstructionJob dataclasses (to be defined in GREEN phase)

    Raises:
        NotImplementedError: Until GREEN implementation (F1.1)
    """
    raise NotImplementedError(
        "build_ptychi_jobs() is a Phase F0 RED stub. "
        "GREEN implementation tracked as F1.1 in "
        "plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/"
        "2025-11-04T094500Z/phase_f_ptychi_baseline_plan/plan.md"
    )
