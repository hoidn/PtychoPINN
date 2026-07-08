"""Build dose-only twins of the frozen lines_N128 datasets for the dose-ladder
etiology experiment (Task 3, cnn N=128 collapse RCA).

Reuses the lines-family object/probe/simulation machinery from
make_lines_datasets.py / make_synthetic_truth_datasets.py rather than
re-deriving the physics: the frozen lines object (L.frozen_lines_object,
cached to disk) and probe (M.load_probe(M.SPECS[128]["probe_src"])) are
identical to the ones used to build lines_N128_{train,test}.npz. Scan
coordinates are extracted read-only from the frozen source npz (already
stored flat, no grouping to invert). Only the count scale (target_mean_count,
via M.to_counts) varies between rungs -- geometry, object, and probe are held
fixed, isolating dose as the sole differential from the collapsing baseline.

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

REPO = Path("/home/ollie/Documents/PtychoPINN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))

import make_gridgeom_dataset as G
import make_lines_datasets as L
import make_synthetic_truth_datasets as M

N = 128
SOURCE_TRAIN_NPZ = M.DS_DIR / "lines_N128_train.npz"
SOURCE_TEST_NPZ = M.DS_DIR / "lines_N128_test.npz"


def extract_coords(npz_path: Path) -> tuple:
    """Read-only extraction of xcoords/ycoords from a flat RawData-format npz
    (lines_N128_{train,test}.npz already stores unpacked scan coordinates)."""
    with np.load(npz_path, allow_pickle=True) as d:
        return d["xcoords"], d["ycoords"]


def build_split(split: str, source_npz: Path, target_mean_count: float,
                 obj: np.ndarray, probe: np.ndarray) -> dict:
    xc, yc = extract_coords(source_npz)
    rd = M.simulate(obj, probe, xc, yc, N)
    counts = M.to_counts(np.asarray(rd.diff3d), target_mean_count=target_mean_count)
    dev = M.cross_pattern_deviation(counts)
    assert dev > 0.2, (
        f"degenerate diffraction (dev={dev:.4f}) for dose ladder {split} "
        f"tmc={target_mean_count}"
    )
    dose = G.measure_photons_per_image(counts)

    out = {
        "xcoords": rd.xcoords, "ycoords": rd.ycoords,
        "xcoords_start": rd.xcoords_start, "ycoords_start": rd.ycoords_start,
        "diff3d": counts, "probeGuess": probe, "objectGuess": obj,
        "scan_index": np.asarray(rd.scan_index),
    }
    if getattr(rd, "Y", None) is not None:
        out["ground_truth_patches"] = np.asarray(rd.Y)

    T = int(round(target_mean_count))
    path = M.DS_DIR / f"lines_N{N}_tmc{T}_{split}.npz"
    np.savez(path, **out)

    return {
        "path": str(path), "n": int(len(xc)), "N": N,
        "target_mean_count": target_mean_count,
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

    probe = M.load_probe(M.SPECS[N]["probe_src"])
    assert probe.shape[0] == N, f"probe {probe.shape} != N={N}"
    obj = L.frozen_lines_object(N, M.SPECS[N]["obj_res"])

    prov_path = M.DS_DIR / "provenance_dose_ladder.json"
    if prov_path.exists():
        with open(prov_path) as fh:
            provenance = json.load(fh)
    else:
        provenance = {
            "task": "Task 3 (E1) dose ladder: dose-only twins of lines_N128, geometry/object/probe held fixed",
            "source": {"train": str(SOURCE_TRAIN_NPZ), "test": str(SOURCE_TEST_NPZ)},
            "object": "make_lines_datasets.frozen_lines_object (identical cached object to lines_N128)",
            "probe": "make_synthetic_truth_datasets.load_probe(SPECS[128].probe_src) (identical to lines_N128)",
            "nphotons_legacy_inert": M.NPHOTONS,
            "nphotons_note": "M.NPHOTONS is inert legacy metadata, not the binding dose parameter; target_mean_count (per rung, below) sets the counts scale via to_counts(), and photons/image is MEASURED per split, not assumed.",
            "convention": "diff3d = round(amp^2 * S) uint16 counts, S set per rung by target_mean_count",
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
