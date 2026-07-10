"""Build dose-only twins of the frozen lines_N128 datasets for the dose-ladder
etiology experiment (Task 3, cnn N=128 collapse RCA).

Reuses the lines-family object/probe/simulation machinery from
make_lines_datasets.py / make_synthetic_truth_datasets.py rather than
re-deriving the physics: the frozen lines object (L.frozen_lines_object,
cached to disk) and probe (M.load_probe(M.SPECS[128]["probe_src"])) are
identical to the ones used to build lines_N128_{train,test}.npz. Scan
coordinates are extracted read-only from the frozen source npz (already
stored flat, no grouping to invert). Each rung independently evaluates the
same raw object/probe forward, calibrates the stored physical probe amplitude
to target_mean_count, and draws fresh Poisson counts.

Output: flat RawData-format npz mirroring lines_N128_{train,test}.npz's
key/dtype/shape contract, at .artifacts/varpro_ablation/datasets/
lines_N128_tmc{T}_{train,test}.npz (T = int(target_mean_count)), plus a
shared provenance_dose_ladder.json accumulating one record per rung with the
MEASURED counts_mean/counts_max and per-split photons/image (mean/min/max);
nphotons is recorded as inert legacy metadata, not the dose parameter.

The frozen source datasets are opened read-only and never modified.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))

import make_gridgeom_dataset as G
import make_lines_datasets as L
import make_synthetic_truth_datasets as M

N = 128
SOURCE_TRAIN_NPZ = M.DS_DIR / "lines_N128_train.npz"
SOURCE_TEST_NPZ = M.DS_DIR / "lines_N128_test.npz"
POISSON_SEEDS = {"train": 40_017, "test": 40_018}


def extract_coords(npz_path: Path) -> tuple:
    """Read-only extraction of xcoords/ycoords from a flat RawData-format npz
    (lines_N128_{train,test}.npz already stores unpacked scan coordinates)."""
    with np.load(npz_path, allow_pickle=True) as d:
        return d["xcoords"], d["ycoords"]


def build_split(split: str, source_npz: Path, target_mean_count: float,
                 obj: np.ndarray, probe: np.ndarray) -> dict:
    xc, yc = extract_coords(source_npz)
    generated = M.generate_ci_count_dataset(
        obj,
        probe,
        xc,
        yc,
        N=N,
        target_mean_count=target_mean_count,
        poisson_seed=POISSON_SEEDS[split] + int(round(target_mean_count * 10)),
    )
    counts = generated.payload["diff3d"]
    dev = M.cross_pattern_deviation(counts)
    assert dev > 0.2, (
        f"degenerate diffraction (dev={dev:.4f}) for dose ladder {split} "
        f"tmc={target_mean_count}"
    )
    dose = G.measure_photons_per_image(counts)

    T = int(round(target_mean_count))
    path = M.DS_DIR / f"lines_N{N}_tmc{T}_{split}.npz"
    np.savez(path, **generated.payload)

    return {
        "path": str(path), "n": int(len(xc)), "N": N,
        "target_mean_count": target_mean_count,
        "probe_gauge": generated.metadata["probe_gauge"],
        "dose_amplitude_scale": generated.dose_amplitude_scale,
        "cross_pattern_deviation": round(dev, 4),
        "counts_mean": round(float(counts.mean()), 2), "counts_max": int(counts.max()),
        "photons_per_image": {
            "mean": round(dose["mean"], 1), "min": round(dose["min"], 1), "max": round(dose["max"], 1),
        },
        "source": str(source_npz),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="Build a dose-only lines_N128 twin (train+test) at a given target_mean_count."
    )
    ap.add_argument("--target-mean-count", type=float, required=True)
    args = ap.parse_args(argv)
    target_mean_count = args.target_mean_count

    probe = M.load_probe(M.SPECS[N]["probe_src"], N=N)
    obj = L.frozen_lines_object(N, M.SPECS[N]["obj_res"])

    prov_path = M.DS_DIR / "provenance_dose_ladder.json"
    if prov_path.exists():
        with open(prov_path) as fh:
            provenance = json.load(fh)
    else:
        provenance = {
            "task": "Task 3 (E1) calibrated dose ladder for lines_N128",
            "source": {"train": str(SOURCE_TRAIN_NPZ), "test": str(SOURCE_TEST_NPZ)},
            "object": "make_lines_datasets.frozen_lines_object (identical cached object to lines_N128)",
            "probe": "raw source probe calibrated to count dose independently per rung/split",
            "scale_contract_version": M.CI_SCALE_CONTRACT,
            "measurement_domain": M.COUNT_INTENSITY,
            "probe_gauge": M.PHYSICAL_CALIBRATED_PROBE_GAUGE,
            "convention": "fresh Poisson count intensity from the calibrated physical probe",
            "rungs": {},
        }

    rung_key = f"tmc{int(round(target_mean_count))}"
    rung = {"target_mean_count": target_mean_count, "outputs": {}}
    for split, source_npz in (("train", SOURCE_TRAIN_NPZ), ("test", SOURCE_TEST_NPZ)):
        info = build_split(split, source_npz, target_mean_count, obj, probe)
        rung["outputs"][split] = info
        print(f"tmc{target_mean_count} {split}: {info}")
    provenance["rungs"][rung_key] = rung

    with open(prov_path, "w") as fh:
        json.dump(provenance, fh, indent=1)
    print("wrote", prov_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
