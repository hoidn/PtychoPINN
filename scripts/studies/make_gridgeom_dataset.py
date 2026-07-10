"""Build a flat count-convention LINES dataset using the frozen gs2 gate's
dense-4px-grid scan geometry (Task 3 Step 1, RESURRECTED 2026-07-08).

Reuses the lines-family object/probe/count-convention lineage from
make_lines_datasets.py / make_synthetic_truth_datasets.py (provenance_lines.json
names that lineage) rather than duplicating the physics. The only new piece is
the scan geometry: instead of make_synthetic_truth_datasets.scan_positions'
jittered grid, xcoords/ycoords are extracted read-only from the frozen grouped
gs2 gate dataset at .artifacts/integration/grid_lines_gs2/datasets/N128/gs2/
{train,test}.npz.

That frozen file stores gridsize=2 grouped coordinates: coords_nominal is the
per-member offset relative to its group center (local_offset_sign=-1 convention,
see ptycho.raw_data.get_relative_coords), and coords_offsets is the group center
in absolute frame. Absolute position recovers as

    abs = coords_offsets - coords_nominal

(inverting coords_relative = -(coords_nn - coords_offsets)). Flattening every
group's 4 members and taking the unique set recovers the underlying scan grid.

Output: flat RawData-format npz mirroring lines_N128_{train,test}.npz's
key/dtype/shape contract (xcoords/ycoords/xcoords_start/ycoords_start/
diff3d(uint16)/calibrated probeGuess/objectGuess/scan_index/
ground_truth_patches/_metadata), plus
provenance_gridgeom.json recording the measured photons/image dose (must be
>=1e6 per user mandate) and the coords source citation.
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))

import make_lines_datasets as L
import make_synthetic_truth_datasets as M

N = 128
GATE_TRAIN_NPZ = REPO / ".artifacts/integration/grid_lines_gs2/datasets/N128/gs2/train.npz"
GATE_TEST_NPZ = REPO / ".artifacts/integration/grid_lines_gs2/datasets/N128/gs2/test.npz"
EXPECTED_PITCH = 4.0
POISSON_SEEDS = {"train": 50_017, "test": 50_018}


def extract_unique_scan_positions(npz_path: Path) -> tuple:
    """Recover flat, unique absolute scan positions from a frozen gs2 grouped
    npz (coords_nominal/coords_offsets, shape (n_groups, 1, 2, gridsize**2)).
    Read-only: does not mutate or resave the source file."""
    with np.load(npz_path, allow_pickle=True) as d:
        coords_nominal = d["coords_nominal"]
        coords_offsets = d["coords_offsets"]
    x_abs = coords_offsets[:, 0, 0, :] - coords_nominal[:, 0, 0, :]
    y_abs = coords_offsets[:, 0, 1, :] - coords_nominal[:, 0, 1, :]
    pts = np.stack([x_abs.ravel(), y_abs.ravel()], axis=1)
    uniq = np.unique(pts, axis=0)
    return uniq[:, 0], uniq[:, 1]


def verify_grid_pitch(xc: np.ndarray, expected_pitch: float) -> float:
    """Verify the unique x-coordinate values form a regular grid at
    expected_pitch (px), returning the measured (modal) pitch. Raises if the
    dominant spacing does not match."""
    ux = np.unique(xc)
    diffs = np.round(np.diff(ux), 4)
    values, counts = np.unique(diffs, return_counts=True)
    modal_pitch = float(values[np.argmax(counts)])
    assert modal_pitch == expected_pitch, (
        f"grid pitch {modal_pitch} != expected {expected_pitch} px"
    )
    return modal_pitch


def measure_photons_per_image(diff3d: np.ndarray) -> dict:
    """diff3d is the count-convention array (n, N, N) uint16. Returns
    mean/min/max summed photons per diffraction pattern."""
    photons = diff3d.astype(np.float64).sum(axis=(1, 2))
    return {
        "mean": float(photons.mean()),
        "min": float(photons.min()),
        "max": float(photons.max()),
    }


def build_split(split: str, gate_npz: Path, obj: np.ndarray, probe: np.ndarray) -> dict:
    xc, yc = extract_unique_scan_positions(gate_npz)
    measured_pitch = verify_grid_pitch(xc, EXPECTED_PITCH)

    generated = M.generate_ci_count_dataset(
        obj,
        probe,
        xc,
        yc,
        N=N,
        target_mean_count=M.TARGET_MEAN_COUNT,
        poisson_seed=POISSON_SEEDS[split],
    )
    counts = generated.payload["diff3d"]
    dev = M.cross_pattern_deviation(counts)
    assert dev > 0.2, f"degenerate diffraction (dev={dev:.4f}) for gridgeom {split}"

    dose = measure_photons_per_image(counts)

    path = M.DS_DIR / f"gridgeom_N{N}_{split}.npz"
    np.savez(path, **generated.payload)

    return {
        "path": str(path), "n": int(len(xc)), "N": N,
        "probe_gauge": generated.metadata["probe_gauge"],
        "dose_amplitude_scale": generated.dose_amplitude_scale,
        "grid_pitch_px": measured_pitch,
        "cross_pattern_deviation": round(dev, 4),
        "counts_mean": round(float(counts.mean()), 2), "counts_max": int(counts.max()),
        "photons_per_image": {
            "mean": round(dose["mean"], 1), "min": round(dose["min"], 1), "max": round(dose["max"], 1),
        },
        "coords_source": str(gate_npz),
    }


def main() -> int:
    probe = M.load_probe(M.SPECS[N]["probe_src"], N=N)
    obj = L.frozen_lines_object(N, M.SPECS[N]["obj_res"])

    provenance = {
        "task": "Task 3 Step 1 RESURRECTED: gridgeom flat dataset, gate's dense-4px-grid geometry",
        "object": "ptycho.diffsim.sim_object_image(data_source='lines') -> amp-norm + phase (reused from make_lines_datasets.frozen_lines_object)",
        "amp_range": [L.AMP_LO, 1.0], "phase_max_rad": L.PHASE_MAX,
        "target_mean_count": M.TARGET_MEAN_COUNT,
        "scale_contract_version": M.CI_SCALE_CONTRACT,
        "measurement_domain": M.COUNT_INTENSITY,
        "probe_gauge": M.PHYSICAL_CALIBRATED_PROBE_GAUGE,
        "convention": "fresh Poisson count intensity from the calibrated physical probe",
        "coords_source": {
            "train": str(GATE_TRAIN_NPZ), "test": str(GATE_TEST_NPZ),
            "extraction": "abs = coords_offsets - coords_nominal (grouped gs2 gate npz, read-only), unique() over flattened group members",
            "expected_pitch_px": EXPECTED_PITCH,
        },
        "dose_requirement": "mean photons/image >= 1e6 (user-mandated 2026-07-08)",
        "outputs": {},
    }
    for split, gate_npz in (("train", GATE_TRAIN_NPZ), ("test", GATE_TEST_NPZ)):
        info = build_split(split, gate_npz, obj, probe)
        assert info["photons_per_image"]["mean"] >= 1.0e6, (
            f"gridgeom {split} mean photons/image {info['photons_per_image']['mean']:.3e} "
            f"below 1e6 floor (raise M.TARGET_MEAN_COUNT)"
        )
        provenance["outputs"][f"N{N}_{split}"] = info
        print(f"N={N} {split}: {info}")

    with open(M.DS_DIR / "provenance_gridgeom.json", "w") as fh:
        json.dump(provenance, fh, indent=1)
    print("wrote", M.DS_DIR / "provenance_gridgeom.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
