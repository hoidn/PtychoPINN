"""Build synthetic dead_leaves datasets WITH real ground truth for the ablation.

Task 1.6 corrective (user decision 2026-07-01, plan amendment #12): fly001 is
experimental data whose stored ``objectGuess`` is an all-ones placeholder, so
error-vs-truth metrics are invalid, and simulating N=128 from that placeholder
(as Task 0.1 did) yields degenerate near-identical diffraction. Replace with
synthetic ``dead_leaves`` objects: scale-invariant multi-scale structure
(features across the full spatial-frequency band -- exactly what makes
varpro/probe effects visible), and a KNOWN object we store as ground truth.

Pipeline (per N in {64, 128}, per split train/test):
1. create_dead_leaves -> complex object, sized so all scan positions + probe
   fit inside with a buffer (obj_res = span + 2*N).
2. Probe: real experimental probe reused from the fly datasets (fly001's 64x64
   for N=64; fly128_p1e9's 128x128 for N=128) -- realistic illumination.
3. Scan positions: Sobol-like jittered grid giving dense uniform coverage
   (overlap-rich for train, sparser for test), seeded per split.
4. RawData.from_simulation -> normalized-amplitude diff3d with Poisson noise at
   nphotons SNR (main's tested physics), plus ground_truth_patches (Y).
5. Convert to COUNT convention to match real fly001 (the form that empirically
   trains under Poisson loss): counts = round(amp^2 * S), S chosen so mean
   per-pixel count ~= real fly001 (~108). Store as uint16 diff3d.
6. Save with the dead_leaves object as ``objectGuess`` (REAL truth).

Non-degeneracy is asserted: cross-pattern relative deviation must exceed 0.2
(real fly001 is 0.57; the broken fly128 sim was 0.002).
"""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ollie/Documents/PtychoPINN")
sys.path.insert(0, str(REPO))

from ptycho import params as p
from ptycho.raw_data import RawData
from ptycho_torch.datagen.objects import create_dead_leaves

DS_DIR = REPO / ".artifacts/varpro_ablation/datasets"
FLY64_PROBE_SRC = REPO / "datasets/fly/fly001.npz"
FLY128_PROBE_SRC = DS_DIR / "fly128_p1e9_train.npz"

TARGET_MEAN_COUNT = 108.0  # real fly001 mean pixel count
NPHOTONS = 1.0e9
DEAD_LEAVES_ARG = {"max_iters": 700, "r_min_frac": 0.02, "r_max_frac": 0.18, "r_sigma": 3}

# Object phase is compressed into the real_imag decoder's representable box
# (real in (-0.8,1.15), imag in (-1.19,0.65); model.py:430-449). The binding
# constraint is the ASYMMETRIC positive-imag ceiling +0.65: imag = amp*sin(phase),
# so the worst-case pixel is amp_max*sin(PHASE_MAX). With dead_leaves amp in
# [0.6,1.1], PHASE_MAX=0.5 gives max imag 1.1*sin(0.5)=0.53 < 0.65 -> 0% out-of-box.
# (PHASE_MAX=0.8 railed 16.5% of pixels on the imag ceiling and capped recon at
# canvas |amp| NCC ~0.57; full +-pi caps per-patch at 0.31.) See amendment #13.
# VarPro/probe-scaling effects are amplitude/intensity-gauge phenomena, unaffected
# by this phase range.
PHASE_MAX = 0.5


def compress_phase(obj: np.ndarray, pmax: float) -> np.ndarray:
    """Rescale object phase +-pi -> +-pmax, amplitude untouched."""
    amp = np.abs(obj)
    ph = np.angle(obj) / np.pi * pmax
    return (amp * np.exp(1j * ph)).astype(np.complex64)


def frozen_raw_object(N: int, obj_res: int) -> np.ndarray:
    """Reproducible raw dead_leaves object. create_dead_leaves ignores
    np.random.seed (the perlin/noise backend keeps its own RNG), so freeze the
    object to disk on first generation and reuse it thereafter -- this is what
    makes the whole demo reproducible across regenerations."""
    cache = DS_DIR / f"rawobj_N{N}.npy"
    if cache.exists():
        return np.load(cache)
    obj = create_dead_leaves((obj_res, obj_res), DEAD_LEAVES_ARG).astype(np.complex64)
    np.save(cache, obj)
    return obj

# (N, probe_src, obj_res, train {n,seed,jitter}, test {n,seed,jitter})
SPECS = {
    64: dict(probe_src=FLY64_PROBE_SRC, obj_res=320,
             train=dict(n=512, seed=7, jitter=1.5), test=dict(n=128, seed=8, jitter=1.5)),
    128: dict(probe_src=FLY128_PROBE_SRC, obj_res=480,
              train=dict(n=512, seed=17, jitter=2.0), test=dict(n=128, seed=18, jitter=2.0)),
}


def load_probe(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=True) as d:
        return d["probeGuess"].astype(np.complex64)


def scan_positions(obj_res: int, N: int, n: int, seed: int, jitter: float):
    """Jittered square grid of scan positions inside [N, obj_res-N]."""
    rng = np.random.default_rng(seed)
    lo, hi = float(N), float(obj_res - N)
    side = int(np.ceil(np.sqrt(n)))
    lin = np.linspace(lo, hi, side)
    gx, gy = np.meshgrid(lin, lin)
    pts = np.column_stack([gx.ravel(), gy.ravel()])
    idx = rng.choice(len(pts), size=n, replace=False)
    pts = pts[idx]
    pts += rng.normal(0.0, jitter, size=pts.shape)
    pts = np.clip(pts, lo, hi)
    return pts[:, 0], pts[:, 1]


def simulate(obj: np.ndarray, probe: np.ndarray, xc: np.ndarray, yc: np.ndarray, N: int):
    """RawData.from_simulation with global params set for N/gridsize/nphotons."""
    orig = {k: p.get(k) for k in ("N", "gridsize", "nphotons", "batch_size")}
    try:
        p.set("N", N)
        p.set("gridsize", 1)
        p.set("nphotons", NPHOTONS)
        if p.get("batch_size") is None:
            p.set("batch_size", 64)
        scan_index = np.zeros(len(xc), dtype=int)
        rd = RawData.from_simulation(xc, yc, probe, obj, scan_index)
    finally:
        for k, v in orig.items():
            p.set(k, v)
    return rd


def to_counts(amp: np.ndarray, target_mean_count: float = TARGET_MEAN_COUNT) -> np.ndarray:
    """Normalized-amplitude diff3d -> integer counts at target_mean_count dose
    (default 108.0 matches real fly001 mean; existing callers are unaffected).
    Raises ValueError before the uint16 cast if the scaled max would exceed
    65535 (casting directly would silently wrap instead of failing)."""
    intensity = amp.astype(np.float64) ** 2
    S = target_mean_count / intensity.mean()
    counts = np.round(intensity * S)
    max_count = float(counts.max())
    if max_count > 65535:
        raise ValueError(
            f"to_counts: scaled max count {max_count:.1f} exceeds uint16 range "
            f"(65535) at target_mean_count={target_mean_count} "
            f"(lower target_mean_count or check source dynamic range)"
        )
    return counts.astype(np.uint16)


def cross_pattern_deviation(counts: np.ndarray) -> float:
    x = counts.astype(np.float64)
    m = x.mean(axis=0)
    return float(np.sqrt(((x - m) ** 2).mean()) / (m.mean() + 1e-12))


def build_one(N: int, split: str, spec: dict, obj: np.ndarray, probe: np.ndarray) -> dict:
    cfg = spec[split]
    xc, yc = scan_positions(spec["obj_res"], N, cfg["n"], cfg["seed"], cfg["jitter"])
    rd = simulate(obj, probe, xc, yc, N)
    amp = np.asarray(rd.diff3d)
    counts = to_counts(amp)
    dev = cross_pattern_deviation(counts)
    assert dev > 0.2, f"degenerate diffraction (dev={dev:.4f}) for N={N} {split}"

    out = {
        "xcoords": rd.xcoords, "ycoords": rd.ycoords,
        "xcoords_start": rd.xcoords_start, "ycoords_start": rd.ycoords_start,
        "diff3d": counts,
        "probeGuess": probe,
        "objectGuess": obj,
        "scan_index": np.asarray(rd.scan_index),
    }
    if getattr(rd, "Y", None) is not None:
        out["ground_truth_patches"] = np.asarray(rd.Y)
    path = DS_DIR / f"deadleaves_N{N}_{split}.npz"
    np.savez(path, **out)
    return {
        "path": str(path), "n": int(cfg["n"]), "N": N,
        "cross_pattern_deviation": round(dev, 4),
        "counts_mean": round(float(counts.mean()), 2),
        "counts_max": int(counts.max()),
        "obj_amp_std": round(float(np.abs(obj).std()), 4),
        "obj_phase_std": round(float(np.angle(obj).std()), 4),
        "scan_span": [round(float(xc.max() - xc.min()), 1), round(float(yc.max() - yc.min()), 1)],
    }


def main() -> int:
    provenance = {
        "task": "Task 1.6 corrective #3 (representable-phase dead_leaves w/ truth, amendment #13)",
        "object": "ptycho_torch.datagen.objects.create_dead_leaves",
        "dead_leaves_arg": DEAD_LEAVES_ARG,
        "phase_max_rad": PHASE_MAX,
        "phase_note": "phase compressed +-pi -> +-PHASE_MAX for real_imag decoder box representability",
        "nphotons": NPHOTONS,
        "target_mean_count": TARGET_MEAN_COUNT,
        "convention": "diff3d = round(amp^2 * S) uint16 counts (matches real fly001)",
        "outputs": {},
    }
    for N, spec in SPECS.items():
        probe = load_probe(spec["probe_src"])
        assert probe.shape[0] == N, f"probe {probe.shape} != N={N}"
        # one frozen object per N, shared by train/test (same sample, disjoint scans)
        obj = compress_phase(frozen_raw_object(N, spec["obj_res"]), PHASE_MAX)
        for split in ("train", "test"):
            info = build_one(N, split, spec, obj, probe)
            provenance["outputs"][f"N{N}_{split}"] = info
            print(f"N={N} {split}: {info}")
    with open(DS_DIR / "provenance_deadleaves.json", "w") as fh:
        json.dump(provenance, fh, indent=1)
    print("wrote", DS_DIR / "provenance_deadleaves.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
