"""Build synthetic LINES datasets with real ground truth for the ablation.

Task 1.6 pivot (amendment #13b, user standing instruction "consider switching to
lines if you reach an impasse with this dataset"; "grid lines torch is confirmed
working on fno-stable"). dead_leaves reconstructions plateaued at per-patch |amp|
NCC ~0.31 (reproducibly) because that object is weakly amplitude-scattering
(amp_std ~0.15). Lines are the proven, easy, high-contrast alternative.

Object: ptycho.diffsim.mk_lines_img (random straight lines blurred at Gaussian
scales 1/5/10) via sim_object_image(data_source='lines') -- the exact recipe the
codebase ships and the user confirmed reconstructs. Raw lines are pure-amplitude
(phase 0) with amp in ~[1.2,5.4], which is OUT of the real_imag decoder box
(real ceiling 1.15). We therefore:
  1. normalize amplitude linearly into [AMP_LO, 1.0] (strong contrast, in-box),
  2. add spatially-varying phase in +-PHASE_MAX rad correlated with the line
     structure. The phase is REQUIRED: a constant-phase (pure-amplitude) object
     re-triggers the VarPro X3=0 s1<->s2 degeneracy (progress.md). With amp<=1 and
     |phase|<=0.5, imag=amp*sin(phase)<=0.48 < 0.65 and real in [0.26,1.0] -> 0%
     out-of-box.
The object is FROZEN to disk on first generation (mk_lines_img ignores
np.random.seed) so the whole demo is reproducible.

Reuses count-convention / scan / non-degeneracy helpers from
make_synthetic_truth_datasets. Outputs lines_N{64,128}_{train,test}.npz in the
same schema (diff3d uint16 counts, objectGuess = complex truth,
ground_truth_patches).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ollie/Documents/PtychoPINN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))

from ptycho import params as p
from ptycho.diffsim import sim_object_image
import make_synthetic_truth_datasets as M

AMP_LO = 0.3      # normalized-amplitude floor (background); ceiling is 1.0
PHASE_MAX = 0.5   # +-rad, in-box; nonzero to keep VarPro non-degenerate


def _raw_lines(obj_res: int) -> np.ndarray:
    """sim_object_image('lines') amplitude map at obj_res (its shipped recipe)."""
    orig = {k: p.get(k) for k in ("data_source", "size", "N")}
    try:
        p.set("data_source", "lines")
        p.set("size", obj_res)
        img = sim_object_image(size=obj_res).squeeze().astype(np.float64)
    finally:
        for k, v in orig.items():
            p.set(k, v)
    return np.abs(img)


def frozen_lines_object(N: int, obj_res: int) -> np.ndarray:
    """Reproducible complex lines object: amp in [AMP_LO,1], phase in +-PHASE_MAX
    correlated with the lines. Frozen to disk on first generation."""
    cache = M.DS_DIR / f"linesobj_N{N}.npy"
    if cache.exists():
        return np.load(cache)
    raw = _raw_lines(obj_res)
    t = (raw - raw.min()) / (raw.max() - raw.min() + 1e-12)   # [0,1]
    amp = AMP_LO + (1.0 - AMP_LO) * t                          # [AMP_LO, 1.0]
    phase = PHASE_MAX * (2.0 * t - 1.0)                        # [-PHASE_MAX, PHASE_MAX]
    obj = (amp * np.exp(1j * phase)).astype(np.complex64)
    np.save(cache, obj)
    return obj


def build_one(N: int, split: str, spec: dict, obj: np.ndarray, probe: np.ndarray) -> dict:
    cfg = spec[split]
    xc, yc = M.scan_positions(spec["obj_res"], N, cfg["n"], cfg["seed"], cfg["jitter"])
    rd = M.simulate(obj, probe, xc, yc, N)
    counts = M.to_counts(np.asarray(rd.diff3d))
    dev = M.cross_pattern_deviation(counts)
    assert dev > 0.2, f"degenerate diffraction (dev={dev:.4f}) for N={N} {split}"
    out = {
        "xcoords": rd.xcoords, "ycoords": rd.ycoords,
        "xcoords_start": rd.xcoords_start, "ycoords_start": rd.ycoords_start,
        "diff3d": counts, "probeGuess": probe, "objectGuess": obj,
        "scan_index": np.asarray(rd.scan_index),
    }
    if getattr(rd, "Y", None) is not None:
        out["ground_truth_patches"] = np.asarray(rd.Y)
    path = M.DS_DIR / f"lines_N{N}_{split}.npz"
    np.savez(path, **out)
    re, im = obj.real, obj.imag
    return {
        "path": str(path), "n": int(cfg["n"]), "N": N,
        "cross_pattern_deviation": round(dev, 4),
        "counts_mean": round(float(counts.mean()), 2), "counts_max": int(counts.max()),
        "obj_amp_std": round(float(np.abs(obj).std()), 4),
        "obj_phase_std": round(float(np.angle(obj).std()), 4),
        "out_of_box": round(float(np.mean((re < -0.8) | (re > 1.15) | (im < -1.19) | (im > 0.65))), 4),
    }


def main() -> int:
    provenance = {
        "task": "Task 1.6 pivot to synthetic lines (amendment #13b)",
        "object": "ptycho.diffsim.sim_object_image(data_source='lines') -> amp-norm + phase",
        "amp_range": [AMP_LO, 1.0], "phase_max_rad": PHASE_MAX,
        "nphotons": M.NPHOTONS, "target_mean_count": M.TARGET_MEAN_COUNT,
        "convention": "diff3d = round(amp^2 * S) uint16 counts (matches real fly001)",
        "outputs": {},
    }
    for N, spec in M.SPECS.items():
        probe = M.load_probe(spec["probe_src"])
        assert probe.shape[0] == N, f"probe {probe.shape} != N={N}"
        obj = frozen_lines_object(N, spec["obj_res"])
        for split in ("train", "test"):
            info = build_one(N, split, spec, obj, probe)
            provenance["outputs"][f"N{N}_{split}"] = info
            print(f"N={N} {split}: {info}")
    with open(M.DS_DIR / "provenance_lines.json", "w") as fh:
        json.dump(provenance, fh, indent=1)
    print("wrote", M.DS_DIR / "provenance_lines.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
