"""Post-hoc metric recomputation with the ptychographic gauge quotiented out.

The ablation harness's compute_metrics removes only the global PHASE before
scoring recon vs truth. But a ptychographic object reconstruction has a full
global complex-scalar gauge freedom (object * c, probe / c is unobservable),
so a variant that rescales amplitude -- exactly what VarPro does -- is
penalized for a gauge choice, not a reconstruction error. This reads the saved
per-variant canvas.npz files (no retraining) and rescores each with the
least-squares-optimal global complex scalar removed:

    alpha = <recon, truth> / <recon, recon> = sum(conj(recon)*truth)/sum(|recon|^2)
    aligned = alpha * recon         # cancels global magnitude AND phase

This quotients the ISOTROPIC gauge; VarPro's ANISOTROPIC part (s1 != s2, sign
flips) survives, so a genuinely structure-changing rescale still shows. Also
reports a unit-RMS-normalized amplitude MAE as a second scale-invariant view.

Prints, per arm/variant: phase-only (stored) vs gauge-quotiented MAE, plus the
recovered alpha magnitude -- so the findings doc can separate "VarPro changed
the absolute scale" (large |alpha| correction) from "VarPro changed structure"
(residual MAE after the correction).
"""
import glob
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/home/ollie/Documents/PtychoPINN")
DS_DIR = REPO / ".artifacts/varpro_ablation/datasets"


def center_crop(a, shape):
    h, w = a.shape[-2:]
    th, tw = shape
    return a[..., (h - th) // 2:(h - th) // 2 + th, (w - tw) // 2:(w - tw) // 2 + tw]


def overlap_crop(a, b):
    th, tw = min(a.shape[-2], b.shape[-2]), min(a.shape[-1], b.shape[-1])
    return center_crop(a, (th, tw)), center_crop(b, (th, tw))


def gauge_align(recon, truth):
    """Optimal global complex scalar: alpha = <recon,truth>/<recon,recon>."""
    num = np.sum(np.conj(recon) * truth)
    den = np.sum(np.abs(recon) ** 2) + 1e-30
    alpha = num / den
    return alpha * recon, alpha


def mae_set(aligned, truth):
    cmae = float(np.mean(np.abs(aligned - truth)))
    amae = float(np.mean(np.abs(np.abs(aligned) - np.abs(truth))))
    pdiff = np.angle(np.exp(1j * (np.angle(aligned) - np.angle(truth))))
    pmae = float(np.mean(np.abs(pdiff)))
    return cmae, amae, pmae


def rms_norm_amp_mae(recon, truth):
    r = recon / (np.sqrt(np.mean(np.abs(recon) ** 2)) + 1e-30)
    t = truth / (np.sqrt(np.mean(np.abs(truth) ** 2)) + 1e-30)
    # phase-align then amplitude MAE
    inner = np.sum(r * np.conj(t))
    r = r * np.exp(-1j * np.angle(inner))
    return float(np.mean(np.abs(np.abs(r) - np.abs(t))))


def truth_for_arm(arm):
    split = "N128" if arm.endswith("_n128") else "N64"
    with np.load(DS_DIR / f"deadleaves_{split}_test.npz", allow_pickle=True) as d:
        return d["objectGuess"]


def main(matrix_dir):
    out = {}
    header = f"{'arm':26s} {'variant':18s} | {'stored_cmae':>11s} {'gauge_cmae':>10s} {'gauge_amae':>10s} {'rmsN_amae':>9s} {'|alpha|':>8s}"
    print(header)
    print("-" * len(header))
    for mf in sorted(glob.glob(f"{matrix_dir}/*/*/canvas.npz")):
        p = Path(mf).parts
        arm, variant = p[-3], p[-2]
        recon = np.load(mf)["canvas"]
        truth = truth_for_arm(arm)
        rc, tc = overlap_crop(recon, truth)
        aligned, alpha = gauge_align(rc, tc)
        cmae, amae, pmae = mae_set(aligned, tc)
        rmsn = rms_norm_amp_mae(rc, tc)
        stored = json.load(open(Path(mf).parent / "metrics.json"))
        out.setdefault(arm, {})[variant] = {
            "gauge_complex_mae": cmae, "gauge_amp_mae": amae, "gauge_phase_mae": pmae,
            "rms_norm_amp_mae": rmsn, "alpha_abs": float(np.abs(alpha)),
            "stored_complex_mae": stored["complex_mae"], "s1": stored["s1"], "s2": stored["s2"],
        }
        print(f"{arm:26s} {variant:18s} | {stored['complex_mae']:11.3f} {cmae:10.3f} {amae:10.3f} {rmsn:9.3f} {np.abs(alpha):8.3f}")
    dest = Path(matrix_dir) / "gauge_metrics.json"
    json.dump(out, open(dest, "w"), indent=1)
    print("\nwrote", dest)
    return 0


if __name__ == "__main__":
    md = sys.argv[1] if len(sys.argv) > 1 else str(REPO / ".artifacts/varpro_ablation/matrix_dl")
    sys.exit(main(md))
