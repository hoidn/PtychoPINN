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
4. Extract noiseless object patches and evaluate the object/raw-probe forward.
5. Calibrate the physical probe amplitude to the requested mean count dose,
   then draw one fresh Poisson count measurement. Store uint16 count intensity.
6. Save with the dead_leaves object as ``objectGuess`` (REAL truth).

Non-degeneracy is asserted: cross-pattern relative deviation must exceed 0.2
(real fly001 is 0.57; the broken fly128 sim was 0.002).
"""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from ptycho import params as p
from ptycho.raw_data import RawData, get_image_patches, get_relative_coords
from ptycho_torch.datagen.objects import create_dead_leaves
from ptycho_torch.scaling_contract import CI_SCALE_CONTRACT, COUNT_INTENSITY

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

PHYSICAL_CALIBRATED_PROBE_GAUGE = "physical_calibrated"
ABSOLUTE_OBJECT_UNITS = "absolute"


@dataclass(frozen=True)
class CICountMeasurement:
    counts: np.ndarray
    expected_count_intensity: np.ndarray
    probe_physical: np.ndarray
    dose_intensity_scale: float
    dose_amplitude_scale: float


@dataclass(frozen=True)
class CICountDataset:
    payload: dict
    metadata: dict
    expected_count_intensity: np.ndarray
    dose_intensity_scale: float
    dose_amplitude_scale: float


def compress_phase(obj: np.ndarray, pmax: float) -> np.ndarray:
    """Rescale object phase +-pi -> +-pmax, amplitude untouched."""
    amp = np.abs(obj)
    ph = np.angle(obj) / np.pi * pmax
    return (amp * np.exp(1j * ph)).astype(np.complex64)


def frozen_raw_object(N: int, obj_res: int, *, seed: int) -> np.ndarray:
    """Generate the reproducible raw dead-leaves object for one study size."""
    rng = np.random.default_rng(seed)
    return create_dead_leaves(
        (obj_res, obj_res),
        DEAD_LEAVES_ARG,
        rng=rng,
    ).astype(np.complex64)

# (N, probe_src, obj_res, train {n,seed,jitter}, test {n,seed,jitter})
SPECS = {
    64: dict(probe_src=FLY64_PROBE_SRC, obj_res=320,
             train=dict(n=512, seed=7, jitter=1.5), test=dict(n=128, seed=8, jitter=1.5)),
    128: dict(probe_src=FLY128_PROBE_SRC, obj_res=480,
              train=dict(n=512, seed=17, jitter=2.0), test=dict(n=128, seed=18, jitter=2.0)),
}


def canonicalize_probe_modes(probe: np.ndarray, *, N: int) -> np.ndarray:
    """Validate builder probe layouts and move a trailing singleton mode first."""
    probe = np.asarray(probe)
    expected_spatial_shape = (N, N)
    if probe.ndim == 2 and tuple(probe.shape) == expected_spatial_shape:
        return probe
    if probe.ndim == 3:
        if tuple(probe.shape) == (N, N, 1):
            return np.ascontiguousarray(np.moveaxis(probe, -1, 0))
        if probe.shape[0] > 0 and tuple(probe.shape[-2:]) == expected_spatial_shape:
            return probe
    raise ValueError(
        "probe must have shape (N,N), (N,N,1), or (P,N,N); "
        f"got {tuple(probe.shape)} for N={N}"
    )


def load_probe(path: Path, *, N: int) -> np.ndarray:
    with np.load(path, allow_pickle=True) as d:
        probe = d["probeGuess"].astype(np.complex64)
    return canonicalize_probe_modes(probe, N=N)


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


def legacy_simulate_normalized_amplitude(
    obj: np.ndarray,
    probe: np.ndarray,
    xc: np.ndarray,
    yc: np.ndarray,
    N: int,
):
    """Reproduce the historical normalized-amplitude simulation path."""
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


def legacy_rescale_normalized_amplitude_to_counts(
    amp: np.ndarray,
    target_mean_count: float = TARGET_MEAN_COUNT,
) -> np.ndarray:
    """Deterministically rescale an existing amplitude for legacy artifacts.

    This is not a fresh Poisson measurement and does not calibrate the stored
    probe. New CI builders must use :func:`generate_ci_count_dataset`.
    """
    intensity = amp.astype(np.float64) ** 2
    S = target_mean_count / intensity.mean()
    counts = np.round(intensity * S)
    max_count = float(counts.max())
    if max_count > 65535:
        raise ValueError(
            "legacy_rescale_normalized_amplitude_to_counts: scaled max count "
            f"{max_count:.1f} exceeds uint16 range "
            f"(65535) at target_mean_count={target_mean_count} "
            f"(lower target_mean_count or check source dynamic range)"
        )
    return counts.astype(np.uint16)


# Compatibility names for historical artifact scripts. New builders below do
# not call these aliases; their implementations remain explicitly legacy-named.
simulate = legacy_simulate_normalized_amplitude
to_counts = legacy_rescale_normalized_amplitude_to_counts


def noiseless_detector_intensity(
    object_patches: np.ndarray,
    probe_physical: np.ndarray,
) -> np.ndarray:
    """Evaluate incoherent detector intensity with an orthonormal 2-D FFT."""
    patches = np.asarray(object_patches)
    if patches.ndim != 3 or not np.iscomplexobj(patches):
        raise ValueError("object_patches must be complex with shape (B, H, W)")
    probe = canonicalize_probe_modes(
        np.asarray(probe_physical),
        N=patches.shape[-1],
    )
    if not np.iscomplexobj(probe):
        raise ValueError("probe_physical must be complex with shape (H, W) or (P, H, W)")
    if not np.isfinite(patches).all() or not np.isfinite(probe).all():
        raise ValueError("object patches and probe must contain only finite values")

    modes = probe[None] if probe.ndim == 2 else probe
    exit_waves = patches[:, None] * modes[None]
    detector_fields = np.fft.fft2(exit_waves, axes=(-2, -1), norm="ortho")
    intensity = np.abs(detector_fields) ** 2
    intensity = np.sum(intensity, axis=1)
    return np.fft.fftshift(intensity, axes=(-2, -1)).astype(np.float64)


def generate_ci_count_measurement(
    object_patches: np.ndarray,
    raw_probe: np.ndarray,
    *,
    target_mean_count: float,
    rng: np.random.Generator,
) -> CICountMeasurement:
    """Calibrate a raw probe to dose, then draw fresh Poisson count intensity."""
    if not np.isfinite(target_mean_count) or target_mean_count <= 0:
        raise ValueError("target_mean_count must be positive and finite")
    if not isinstance(rng, np.random.Generator):
        raise TypeError("rng must be a numpy.random.Generator")

    patches = np.asarray(object_patches)
    if patches.ndim != 3:
        raise ValueError("object_patches must have shape (B, H, W)")
    raw_probe = canonicalize_probe_modes(raw_probe, N=patches.shape[-1])
    raw_expected = noiseless_detector_intensity(patches, raw_probe)
    raw_mean = float(raw_expected.mean())
    if not np.isfinite(raw_mean) or raw_mean <= 0:
        raise ValueError("raw object/probe forward must have positive finite mean intensity")

    dose_intensity_scale = float(target_mean_count / raw_mean)
    dose_amplitude_scale = float(np.sqrt(dose_intensity_scale))
    raw_probe_array = np.asarray(raw_probe)
    probe_physical = (raw_probe_array * dose_amplitude_scale).astype(
        raw_probe_array.dtype,
        copy=False,
    )
    expected = noiseless_detector_intensity(object_patches, probe_physical)
    counts_wide = rng.poisson(expected)
    max_count = int(counts_wide.max(initial=0))
    if max_count > np.iinfo(np.uint16).max:
        raise ValueError(
            f"fresh Poisson count {max_count} exceeds uint16 range at "
            f"target_mean_count={target_mean_count}"
        )
    return CICountMeasurement(
        counts=counts_wide.astype(np.uint16),
        expected_count_intensity=expected,
        probe_physical=probe_physical,
        dose_intensity_scale=dose_intensity_scale,
        dose_amplitude_scale=dose_amplitude_scale,
    )


def extract_object_patches(
    obj: np.ndarray,
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    N: int,
) -> np.ndarray:
    """Extract the noiseless object patches used by the historical geometry."""
    xcoords = np.asarray(xcoords)
    ycoords = np.asarray(ycoords)
    if xcoords.ndim != 1 or ycoords.ndim != 1 or xcoords.shape != ycoords.shape:
        raise ValueError("xcoords and ycoords must be matching one-dimensional arrays")
    coords_nn = np.zeros((len(xcoords), 1, 2, 1), dtype=np.float64)
    coords_nn[:, 0, 0, 0] = xcoords
    coords_nn[:, 0, 1, 0] = ycoords
    global_offsets, local_offsets = get_relative_coords(coords_nn)
    patches = get_image_patches(
        obj,
        global_offsets,
        local_offsets,
        N=N,
        gridsize=1,
    )
    return np.asarray(patches)[..., 0].astype(np.complex64)


def generate_ci_count_dataset(
    obj: np.ndarray,
    raw_probe: np.ndarray,
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    *,
    N: int,
    target_mean_count: float,
    poisson_seed: int,
    scan_index: np.ndarray | None = None,
) -> CICountDataset:
    """Build one count-calibrated CI NPZ payload from object and raw probe."""
    xcoords = np.asarray(xcoords)
    ycoords = np.asarray(ycoords)
    patches = extract_object_patches(obj, xcoords, ycoords, N)
    measurement = generate_ci_count_measurement(
        patches,
        raw_probe,
        target_mean_count=target_mean_count,
        rng=np.random.default_rng(poisson_seed),
    )
    if scan_index is None:
        scan_index = np.zeros(len(xcoords), dtype=np.int64)
    else:
        scan_index = np.asarray(scan_index)
        if scan_index.shape != xcoords.shape:
            raise ValueError("scan_index must match xcoords and ycoords")

    metadata = {
        "schema_version": "1.0.0",
        "scale_contract_version": CI_SCALE_CONTRACT,
        "measurement_domain": COUNT_INTENSITY,
        "probe_gauge": PHYSICAL_CALIBRATED_PROBE_GAUGE,
        "object_units": ABSOLUTE_OBJECT_UNITS,
        "probe_calibration": {
            "status": "calibrated",
            "method": "raw_object_probe_forward_to_requested_mean_count",
            "target_mean_count": float(target_mean_count),
            "dose_intensity_scale": measurement.dose_intensity_scale,
            "dose_amplitude_scale": measurement.dose_amplitude_scale,
        },
        "poisson_sampling": {
            "status": "fresh",
            "seed": int(poisson_seed),
        },
    }
    payload = {
        "xcoords": xcoords,
        "ycoords": ycoords,
        "xcoords_start": xcoords.copy(),
        "ycoords_start": ycoords.copy(),
        "diff3d": measurement.counts,
        "probeGuess": measurement.probe_physical,
        "objectGuess": np.asarray(obj),
        "scan_index": scan_index,
        "ground_truth_patches": patches[..., None],
        "_metadata": np.array(json.dumps(metadata, sort_keys=True), dtype=object),
    }
    return CICountDataset(
        payload=payload,
        metadata=metadata,
        expected_count_intensity=measurement.expected_count_intensity,
        dose_intensity_scale=measurement.dose_intensity_scale,
        dose_amplitude_scale=measurement.dose_amplitude_scale,
    )


def generate_seeded_dead_leaves_count_dataset(
    raw_probe: np.ndarray,
    xcoords: np.ndarray,
    ycoords: np.ndarray,
    *,
    N: int,
    obj_res: int,
    target_mean_count: float,
    seed: int,
    dead_leaves_arg: dict | None = None,
    phase_max: float = PHASE_MAX,
) -> CICountDataset:
    """Generate object, calibrated probe, noiseless mean, and counts from one seed."""
    object_sequence, poisson_sequence = np.random.SeedSequence(seed).spawn(2)
    raw_object = create_dead_leaves(
        (obj_res, obj_res),
        DEAD_LEAVES_ARG if dead_leaves_arg is None else dead_leaves_arg,
        rng=np.random.default_rng(object_sequence),
    ).astype(np.complex64)
    obj = compress_phase(raw_object, phase_max)
    poisson_seed = int(poisson_sequence.generate_state(1, dtype=np.uint32)[0])
    return generate_ci_count_dataset(
        obj,
        raw_probe,
        xcoords,
        ycoords,
        N=N,
        target_mean_count=target_mean_count,
        poisson_seed=poisson_seed,
    )


def cross_pattern_deviation(counts: np.ndarray) -> float:
    x = counts.astype(np.float64)
    m = x.mean(axis=0)
    return float(np.sqrt(((x - m) ** 2).mean()) / (m.mean() + 1e-12))


def build_one(N: int, split: str, spec: dict, obj: np.ndarray, probe: np.ndarray) -> dict:
    cfg = spec[split]
    xc, yc = scan_positions(spec["obj_res"], N, cfg["n"], cfg["seed"], cfg["jitter"])
    generated = generate_ci_count_dataset(
        obj,
        probe,
        xc,
        yc,
        N=N,
        target_mean_count=TARGET_MEAN_COUNT,
        poisson_seed=10_000 + cfg["seed"],
    )
    counts = generated.payload["diff3d"]
    dev = cross_pattern_deviation(counts)
    assert dev > 0.2, f"degenerate diffraction (dev={dev:.4f}) for N={N} {split}"

    path = DS_DIR / f"deadleaves_N{N}_{split}.npz"
    np.savez(path, **generated.payload)
    return {
        "path": str(path), "n": int(cfg["n"]), "N": N,
        "probe_gauge": generated.metadata["probe_gauge"],
        "dose_amplitude_scale": generated.dose_amplitude_scale,
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
        "target_mean_count": TARGET_MEAN_COUNT,
        "scale_contract_version": CI_SCALE_CONTRACT,
        "measurement_domain": COUNT_INTENSITY,
        "probe_gauge": PHYSICAL_CALIBRATED_PROBE_GAUGE,
        "convention": "fresh Poisson count intensity from the calibrated physical probe",
        "outputs": {},
    }
    for N, spec in SPECS.items():
        probe = load_probe(spec["probe_src"], N=N)
        # One seeded object per N, shared by train/test (same sample, disjoint scans).
        obj = compress_phase(
            frozen_raw_object(N, spec["obj_res"], seed=1_000 + N),
            PHASE_MAX,
        )
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
