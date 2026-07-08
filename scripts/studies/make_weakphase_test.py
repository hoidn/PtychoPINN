"""Weak-phase dead_leaves test datasets (root-cause fix experiment).

Diagnosis (2026-07-01): per-patch object recon at the correct coordinate is only
NCC ~0.31; reassembly/gauge are fine (all averaging modes agree). The object's
full +-pi phase swing is not representable by main's real/imag decoder box
(real in (-0.8,1.15), imag in (-1.19,0.65)) at unit amplitude, so the network
cannot fit it. This regenerates the SAME dead_leaves objects with phase
compressed to +-PHASE_MAX rad (representable) to test whether object phase
range is the reconstruction bottleneck.
"""
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ollie/Documents/PtychoPINN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))

from ptycho_torch.datagen.objects import create_dead_leaves
import make_synthetic_truth_datasets as M

PHASE_MAX = 0.8  # target |phase| ceiling (rad), within the decoder's reachable box


def compress_phase(obj: np.ndarray, pmax: float) -> np.ndarray:
    amp = np.abs(obj); ph = np.angle(obj)
    ph = ph / (np.pi) * pmax          # +-pi -> +-pmax
    return (amp * np.exp(1j * ph)).astype(np.complex64)


def main() -> int:
    N = 64
    spec = M.SPECS[64]
    probe = M.load_probe(spec["probe_src"])
    rng = np.random.default_rng(1000 + N)
    np.random.seed(int(rng.integers(0, 2**31 - 1)))
    obj = create_dead_leaves((spec["obj_res"], spec["obj_res"]), M.DEAD_LEAVES_ARG).astype(np.complex64)
    obj = compress_phase(obj, PHASE_MAX)
    print(f"weak-phase object: amp std {np.abs(obj).std():.3f}, phase std {np.angle(obj).std():.3f}, "
          f"phase range [{np.angle(obj).min():.2f},{np.angle(obj).max():.2f}]")
    for split in ("train", "test"):
        cfg = spec[split]
        xc, yc = M.scan_positions(spec["obj_res"], N, cfg["n"], cfg["seed"], cfg["jitter"])
        rd = M.simulate(obj, probe, xc, yc, N)
        counts = M.to_counts(np.asarray(rd.diff3d))
        dev = M.cross_pattern_deviation(counts)
        out = {
            "xcoords": rd.xcoords, "ycoords": rd.ycoords,
            "xcoords_start": rd.xcoords_start, "ycoords_start": rd.ycoords_start,
            "diff3d": counts, "probeGuess": probe, "objectGuess": obj,
            "scan_index": np.asarray(rd.scan_index),
        }
        if getattr(rd, "Y", None) is not None:
            out["ground_truth_patches"] = np.asarray(rd.Y)
        path = M.DS_DIR / f"deadleaves_weak_N{N}_{split}.npz"
        np.savez(path, **out)
        print(f"  {split}: dev {dev:.3f}, counts mean {counts.mean():.1f}, -> {path.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
