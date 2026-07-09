"""Photon-flux sweep datasets for the s1/s2 scaling demonstration (amendment #14).

Goal: show that the dynamic-scaling constants s1/s2 (PtychoPINN-CI "dynamic scaling
factorization", manuscript Eq 1/3/5) absorb the ABSOLUTE incident-photon scale --
end-to-end through the real pipeline, not just the isolated solver.

Design (single object + probe, only flux varies):
  * ONE frozen lines object (strong amplitude contrast, reuses make_lines_datasets)
    and ONE probe. Identical scan positions across all fluxes.
  * Simulate the diffraction AMPLITUDE once per split, then quantize to counts at
    several target mean-count levels {1, 100, 10000} -- ~4 orders of magnitude of
    incident flux. Only the multiplicative count scale (+ integer quantization)
    differs between fluxes; the object, probe, coords, and noise-free intensity are
    bit-identical.
  * Train ONCE at the mid flux (mean 100); flux_sweep_eval.py then evaluates that
    single checkpoint at all three test fluxes x inference varpro {ON, OFF}.

Prediction (amendment #14): solved s1/s2 ∝ sqrt(mean-count); varpro-ON keeps |O|~1
across fluxes while varpro-OFF drifts off the training flux.

Outputs (N=64, gridsize 1) in .artifacts/varpro_ablation/datasets/:
  fluxsweep_N64_train.npz              (mean count 100)
  fluxsweep_N64_mean{1,100,10000}_test.npz
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ollie/Documents/PtychoPINN")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))

import make_synthetic_truth_datasets as M
from make_lines_datasets import frozen_lines_object

N = 64
TRAIN_MEAN = 100.0
TEST_MEANS = [1.0, 100.0, 10000.0]


def to_counts_at_mean(amp: np.ndarray, target_mean: float) -> np.ndarray:
    """Amplitude diff -> integer counts with the requested mean pixel count.

    Same convention as make_synthetic_truth_datasets.to_counts but with an explicit
    target mean, so the SAME noise-free intensity maps to different flux levels.
    """
    intensity = amp.astype(np.float64) ** 2
    S = target_mean / (intensity.mean() + 1e-30)
    counts = np.round(intensity * S)
    assert counts.max() < 65535, f"count overflow at mean={target_mean}: {counts.max()}"
    return counts.astype(np.uint16)


def simulate_split(obj, probe, spec, split):
    cfg = spec[split]
    xc, yc = M.scan_positions(spec["obj_res"], N, cfg["n"], cfg["seed"], cfg["jitter"])
    rd = M.simulate(obj, probe, xc, yc, N)
    return rd


def pack(rd, counts, obj, probe):
    out = {
        "xcoords": rd.xcoords, "ycoords": rd.ycoords,
        "xcoords_start": rd.xcoords_start, "ycoords_start": rd.ycoords_start,
        "diff3d": counts, "probeGuess": probe, "objectGuess": obj,
        "scan_index": np.asarray(rd.scan_index),
    }
    if getattr(rd, "Y", None) is not None:
        out["ground_truth_patches"] = np.asarray(rd.Y)
    return out


def main() -> int:
    spec = M.SPECS[N]
    probe = M.load_probe(spec["probe_src"])
    assert probe.shape[0] == N, f"probe {probe.shape} != N={N}"
    obj = frozen_lines_object(N, spec["obj_res"])

    prov = {"task": "amendment #14 photon-flux sweep (s1/s2 scale absorption)",
            "object": "frozen lines (make_lines_datasets.frozen_lines_object)",
            "N": N, "train_mean": TRAIN_MEAN, "test_means": TEST_MEANS,
            "convention": "diff3d = round(amp^2 * S) uint16; S set per target mean count",
            "outputs": {}}

    # --- train (mid flux) ---
    rd_tr = simulate_split(obj, probe, spec, "train")
    amp_tr = np.asarray(rd_tr.diff3d)
    counts_tr = to_counts_at_mean(amp_tr, TRAIN_MEAN)
    p_tr = M.DS_DIR / f"fluxsweep_N{N}_train.npz"
    np.savez(p_tr, **pack(rd_tr, counts_tr, obj, probe))
    prov["outputs"]["train"] = {"path": str(p_tr), "mean": round(float(counts_tr.mean()), 3),
                                "max": int(counts_tr.max()), "n": int(spec["train"]["n"])}
    print(f"train mean={counts_tr.mean():.3f} max={counts_tr.max()}")

    # --- test sweep (same amplitude, 3 flux levels) ---
    rd_te = simulate_split(obj, probe, spec, "test")
    amp_te = np.asarray(rd_te.diff3d)
    for mean in TEST_MEANS:
        counts = to_counts_at_mean(amp_te, mean)
        dev = M.cross_pattern_deviation(counts)
        tag = int(mean)
        p = M.DS_DIR / f"fluxsweep_N{N}_mean{tag}_test.npz"
        np.savez(p, **pack(rd_te, counts, obj, probe))
        prov["outputs"][f"test_mean{tag}"] = {
            "path": str(p), "mean": round(float(counts.mean()), 4),
            "max": int(counts.max()), "cross_pattern_deviation": round(dev, 4)}
        print(f"test mean~{mean:>8.1f}: counts mean={counts.mean():.4f} max={counts.max()} dev={dev:.3f}")

    with open(M.DS_DIR / "provenance_fluxsweep.json", "w") as fh:
        json.dump(prov, fh, indent=1)
    print("wrote", M.DS_DIR / "provenance_fluxsweep.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
