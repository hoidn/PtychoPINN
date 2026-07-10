"""Unified placement diagnostic: resolve per-patch (0.707) vs canvas (0.14) gap.

Both quantities must be measured against the SAME true object using ONE
coordinate source (coords_global) and DIRECT overlay (no template-match), so
they are strictly comparable. Template-match is banned here: it inflated an
earlier per-patch number to 0.72 via best-of-window matching in self-similar
texture.

For each reconstructed (trimmed) patch we extract the true object crop at the
same coords_global used for canvas placement, gauge-align, and report |amp| NCC.
We try {patch, patch.T} x {(x,y),(y,x)} placement conventions to rule out a
transpose/axis-swap, then build a direct-overlay canvas in object-pixel frame
and report canvas |amp| NCC by cropping the object at the identical frame.

  If per-patch-at-coord ~= canvas ~= 0.7 : reassembly fine; earlier 0.14 was a
      template-match/frame artifact.
  If per-patch-at-coord ~= 0.7 but canvas ~= 0.14 : genuine overlap/averaging
      loss (but all modes agreed, so not phase) -> positional jitter.
  If per-patch-at-coord ~= 0.14 : the 0.707 was measurement optimism; recon
      patches do not match truth at coords_global -> coordinate/gauge issue.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))
from varpro_probe_ablation_runner import build_configs, build_test_dataset, ARM_TABLE
from ptycho_torch.dataloader import Collate_Lightning
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.lightning_utils import load_checkpoint_with_configs

MIDDLE = 32
SC = Path("/tmp/claude-1000/-home-ollie-Documents-PtychoPINN/9ff76e13-9c97-4cfe-a574-8d5fff1cd235/scratchpad/place")


def trim(p):
    N = p.shape[-1]; s = N // 2 - MIDDLE // 2
    return p[..., s:s + MIDDLE, s:s + MIDDLE]


def gauge(r, t):
    a = np.sum(np.conj(r) * t) / (np.sum(np.abs(r) ** 2) + 1e-30)
    return a * r


def ncc(a, b):
    a = a.ravel().astype(np.float64) - a.mean()
    b = b.ravel().astype(np.float64) - b.mean()
    return float(np.sum(a * b) / (np.sqrt(np.sum(a ** 2) * np.sum(b ** 2)) + 1e-30))


def main():
    test_npz = Path(sys.argv[1])
    ckpt = sys.argv[2]
    arm = ARM_TABLE["gs1_frozen"]
    dc, mc, tc, ic, gc = build_configs(arm, 16, 25)
    ds = build_test_dataset(test_npz, mc, dc, tc, SC)
    n = len(ds)
    raw = ds[list(range(n))]
    b = Collate_Lightning(False)(raw); td = b[0]
    model, _ = load_checkpoint_with_configs(ckpt, PtychoPINN_Lightning)
    model = model.cpu().eval()
    with torch.no_grad():
        patches = model.forward_predict(td["images"], td["coords_relative"], b[1], td["rms_scaling_constant"])
    patches = trim(patches).numpy()[:, 0]                     # (n, M, M) complex
    gc2 = td["coords_global"].squeeze(2).numpy().reshape(-1, 2)  # (n,2)

    obj = np.load(test_npz, allow_pickle=True)["objectGuess"]  # (R,R) complex
    R = obj.shape[0]
    h = MIDDLE // 2

    # coords_global columns: try both as (x,y) and (y,x)
    for axis_name, (cxa, cya) in [("gc=(col0=x,col1=y)", (gc2[:, 0], gc2[:, 1])),
                                  ("gc=(col0=y,col1=x)", (gc2[:, 1], gc2[:, 0]))]:
        for pname, pfun in [("patch", lambda p: p), ("patch.T", lambda p: p.T)]:
            per = []
            for i in range(n):
                cx = int(round(cxa[i])); cy = int(round(cya[i]))
                if cy - h < 0 or cy + h > R or cx - h < 0 or cx + h > R:
                    continue
                truth = obj[cy - h:cy + h, cx - h:cx + h]
                rp = pfun(patches[i])
                rp = gauge(rp, truth)
                per.append(ncc(np.abs(rp), np.abs(truth)))
            if per:
                print(f"  {axis_name:22s} {pname:8s} per-patch@coord |amp| NCC "
                      f"mean {np.mean(per):.3f} (n={len(per)})")

    # Direct-overlay canvas in object-pixel frame, best convention (col0=x,col1=y, patch)
    cx = np.round(gc2[:, 0]).astype(int); cy = np.round(gc2[:, 1]).astype(int)
    canvas = np.zeros((R, R), np.complex128); wt = np.zeros((R, R), np.float64)
    for i in range(n):
        y0, x0 = cy[i] - h, cx[i] - h
        if y0 < 0 or x0 < 0 or y0 + MIDDLE > R or x0 + MIDDLE > R:
            continue
        # gauge each patch to its own truth crop before overlay (removes per-patch scalar)
        truth = obj[y0:y0 + MIDDLE, x0:x0 + MIDDLE]
        p = gauge(patches[i], truth)
        canvas[y0:y0 + MIDDLE, x0:x0 + MIDDLE] += p
        wt[y0:y0 + MIDDLE, x0:x0 + MIDDLE] += 1.0
    m = wt > 0
    canvas[m] /= wt[m]
    ov = ncc(np.abs(canvas[m]), np.abs(obj[m]))
    print(f"  DIRECT-OVERLAY canvas |amp| NCC vs object (same frame, per-patch gauged): {ov:.3f} "
          f"(covered px {int(m.sum())})")

    # And a single global gauge (one scalar for all patches) instead of per-patch:
    canvas2 = np.zeros((R, R), np.complex128); wt2 = np.zeros((R, R), np.float64)
    for i in range(n):
        y0, x0 = cy[i] - h, cx[i] - h
        if y0 < 0 or x0 < 0 or y0 + MIDDLE > R or x0 + MIDDLE > R:
            continue
        canvas2[y0:y0 + MIDDLE, x0:x0 + MIDDLE] += patches[i]
        wt2[y0:y0 + MIDDLE, x0:x0 + MIDDLE] += 1.0
    m2 = wt2 > 0
    canvas2[m2] /= wt2[m2]
    g = gauge(canvas2[m2], obj[m2])
    ov2 = ncc(np.abs(g), np.abs(obj[m2]))
    print(f"  DIRECT-OVERLAY canvas |amp| NCC vs object (single global gauge):        {ov2:.3f}")


if __name__ == "__main__":
    main()
