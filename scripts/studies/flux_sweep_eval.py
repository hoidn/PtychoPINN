"""Evaluate ONE trained checkpoint across the photon-flux sweep (amendment #14).

Loads the mid-flux gs1_frozen checkpoint and reconstructs each flux test set
(mean count {1,100,10000}) with inference varpro {ON, OFF}, recording the solved
s1/s2 and the reconstructed |O| level. Reuses the harness's own inference path
(run_inference_variant -> reconstruct_image_barycentric), so the solve sees exactly
what the ablation sees. Fresh per-flux scratch per amendment #13c.

Prediction (amendment #14): with the object/probe fixed and only the measured
counts scaled, s1,s2 (amplitude-form) should track sqrt(mean-count); c_A grows with
flux. Whether varpro-ON |O| stays ~1 or scales tells us whether the pipeline's
un-normalized-probe convention carries the flux (paper Sec 2.2) or leaves it to s1/s2.

FIDELITY: the RAW amp/phase MAE below come from the harness's phase-aligned
barycentric canvas (validated, kept as-is). The gauge-quotiented |amp| NCC used
to come from center-cropping that same barycentric canvas against the padded
objectGuess -- an unvalidated framing that returned ~0.25-0.31 and contradicted
recon_quality_gate.py's 0.97 for the same checkpoint/object family (lines gs1).
This module now measures fidelity via recon_quality_gate.py's validated
methodology instead: direct placement of raw forward_predict patches at their
true coords_global in the truth object's own pixel frame (direct_placement_canvas),
with the varpro real/imag scale (s1,s2) folded into that placed canvas before a
single global complex gauge is quotiented out (gauged_fidelity). run_anchor_check
reproduces recon_quality_gate.py's ~0.97 on the lines gs1 checkpoint before any
flux-sweep fidelity number is trusted.
"""
import argparse
import dataclasses
import glob
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts/studies"))

from varpro_probe_ablation_runner import (
    build_configs, build_test_dataset, run_inference_variant, compute_metrics, ARM_TABLE,
)
from diagnose_placement import gauge, ncc, trim, MIDDLE
from ptycho_torch.dataloader import Collate_Lightning
from ptycho_torch.model import PtychoPINN_Lightning
from ptycho_torch.lightning_utils import load_checkpoint_with_configs

SC = Path("/tmp/claude-1000/-home-ollie-Documents-PtychoPINN/"
          "9ff76e13-9c97-4cfe-a574-8d5fff1cd235/scratchpad/fluxsweep_eval")
DS = REPO / ".artifacts/varpro_ablation/datasets"
OUT = REPO / ".artifacts/varpro_ablation/matrix_fluxsweep"
MEANS = [1, 100, 10000]

# Hard validation anchor (amendment #14 requirement): the lines gs1_frozen
# checkpoint + its held-out test set is recon_quality_gate.py's own reference
# case, documented (amendment #13c) as canvas |amp| NCC 0.972.
ANCHOR_CKPT_GLOB = ".artifacts/varpro_ablation/matrix_lines/gs1_frozen/**/best-checkpoint.ckpt"
ANCHOR_NPZ = REPO / ".artifacts/varpro_ablation/datasets/lines_N64_test.npz"
ANCHOR_TARGET = 0.97
ANCHOR_TOL = 0.03


def support_median_abs(canvas: np.ndarray) -> float:
    m = np.abs(canvas)
    nz = m[m > 1e-9]
    return float(np.median(nz)) if nz.size else 0.0


def direct_placement_canvas(model, ds, npz_path: Path):
    """Object-frame canvas via recon_quality_gate.py's validated methodology:
    run raw model.forward_predict on every item, trim to MIDDLE, and place
    each patch at its true coords_global directly into the truth object's
    pixel frame (simple overlap-count average, NO barycentric/probe weighting,
    NO gauge yet -- gauging and any varpro s1/s2 scale are applied downstream
    by gauged_fidelity so the same raw canvas serves both varpro on/off).

    Also returns the collated batch (``td``, ``probe``) and the raw
    (untrimmed) ``forward_predict`` output computed along the way, so callers
    needing the measurement-domain error (``predicted_diffraction_amplitude``)
    can reuse this same collate/forward-predict pass instead of recomputing
    it on the same ``ds``."""
    n = len(ds)
    b = Collate_Lightning(False)(ds[list(range(n))]); td = b[0]
    with torch.no_grad():
        patches_raw = model.forward_predict(
            td["images"], td["coords_relative"], b[1], td["rms_scaling_constant"]
        )
    patches = trim(patches_raw).numpy().reshape(-1, MIDDLE, MIDDLE)
    gc2 = td["coords_global"].squeeze(2).numpy().reshape(-1, 2)  # col0=x, col1=y

    obj = np.load(npz_path, allow_pickle=True)["objectGuess"]
    R = obj.shape[0]; h = MIDDLE // 2
    cx = np.round(gc2[:, 0]).astype(int); cy = np.round(gc2[:, 1]).astype(int)
    canvas = np.zeros((R, R), np.complex128); wt = np.zeros((R, R), np.float64)
    for i in range(patches.shape[0]):
        y0, x0 = cy[i] - h, cx[i] - h
        if y0 < 0 or x0 < 0 or y0 + MIDDLE > R or x0 + MIDDLE > R:
            continue
        canvas[y0:y0 + MIDDLE, x0:x0 + MIDDLE] += patches[i]
        wt[y0:y0 + MIDDLE, x0:x0 + MIDDLE] += 1.0
    mask = wt > 0
    canvas[mask] /= wt[mask]
    return canvas, mask, obj, td, b[1], patches_raw


def gauged_fidelity(canvas: np.ndarray, mask: np.ndarray, obj: np.ndarray, s1: float, s2: float):
    """Fold the varpro real/imag scale (s1,s2) into the direct-placement
    canvas -- O_scaled = s1*real + j*s2*imag, matching
    apply_varpro_canvas_scaling's anisotropic rescale (reassembly.py) -- then
    quotient ONE global complex gauge (removes residual isotropic scale +
    rotation only, NOT the s1!=s2 real/imag balance) and score |amp| NCC plus
    gauge-quotiented phase MAE against truth on the covered pixels."""
    r = s1 * canvas[mask].real + 1j * s2 * canvas[mask].imag
    t = obj[mask]
    g = gauge(r, t)
    amp_ncc = ncc(np.abs(g), np.abs(t))
    phase_diff = np.angle(np.exp(1j * (np.angle(g) - np.angle(t))))
    phase_mae = float(np.mean(np.abs(phase_diff)))
    return amp_ncc, phase_mae


def measurement_domain_error(pred_amplitude: np.ndarray, measured_amplitude: np.ndarray) -> float:
    """Fourier/measurement-domain error (plan L52/L89): relative L2 between a
    forward-simulated predicted diffraction amplitude and the dataset's
    measured diffraction amplitude, ``||pred - measured||_2 /
    ||measured||_2``. Identical arrays give exactly 0; a uniform amplitude
    scale factor ``k`` gives exactly ``|k - 1|``."""
    pred = np.asarray(pred_amplitude, dtype=np.float64)
    measured = np.asarray(measured_amplitude, dtype=np.float64)
    denom = np.linalg.norm(measured)
    if denom < 1e-30:
        return float("nan")
    return float(np.linalg.norm(pred - measured) / denom)


def physics_forward_mode_warning(mode: str) -> Optional[str]:
    """Return a prominent warning string if the checkpoint's
    ``physics_forward_mode`` is not ``'amplitude'``, else ``None``.
    ``predicted_diffraction_amplitude``'s fresh-``ForwardModel``-equivalence
    assumption (no trained ``alpha``/``beta``/``rect_scaler`` weights) only
    holds on the ``'amplitude'`` path; on any other mode the measurement-
    domain error numbers may not reflect the checkpoint's own forward
    physics."""
    if mode == "amplitude":
        return None
    return (
        f"WARNING: checkpoint physics_forward_mode={mode!r} (not 'amplitude') -- "
        "predicted_diffraction_amplitude's fresh-ForwardModel equivalence "
        "assumption may not hold; measurement-domain error numbers below may "
        "not reflect the checkpoint's own forward physics."
    )


def predicted_diffraction_amplitude(
    model_config, data_config, patches: torch.Tensor, positions: torch.Tensor,
    probe: torch.Tensor, output_scale_factor: torch.Tensor, experiment_ids: torch.Tensor,
    s1: float, s2: float,
) -> np.ndarray:
    """Forward-simulate diffraction amplitude from the reconstruction: fold
    the varpro real/imag scale into ``patches`` (``O_scaled = s1*real +
    j*s2*imag``, the same convention as ``gauged_fidelity``), then reuse
    ptycho_torch's own forward physics -- ``ForwardModel.forward`` (probe
    illumination + FFT via ``pad_and_diffract`` + the inverse output scale),
    the exact path ``PtychoPINN_Lightning.compute_loss`` uses to produce the
    prediction its Poisson/MAE loss compares against ``observed_images`` --
    to get predicted diffraction amplitude at each scan position, comparable
    to the dataset's measured amplitude (``td["images"]``). Read-only reuse:
    a fresh ``ForwardModel`` is constructed from the checkpoint's own configs
    (mirrors ``scripts/studies/dump_forward_parity_fixtures.py``'s
    ``_run_forward``); no physics code is modified. On the default
    ``physics_forward_mode='amplitude'`` path this evaluates, ``ForwardModel``
    holds no trained weights (``alpha``/``beta``/``rect_scaler`` are unused),
    so this is equivalent to the checkpoint's own forward physics."""
    from ptycho_torch.model import ForwardModel

    scaled = torch.complex(s1 * patches.real, s2 * patches.imag)
    forward_model = ForwardModel(model_config, data_config).eval()
    with torch.no_grad():
        pred = forward_model.forward(
            scaled, None, positions, probe, output_scale_factor, experiment_ids=experiment_ids,
        )
    return pred.detach().cpu().numpy()


def build_scale_payload(label: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Assemble the full-precision, machine-readable per-flux dump (I4):
    solved s1, s2, derived c_A/c_phi_deg, reconstructed |O| on/off levels,
    and the measurement-domain error (I1) on/off -- everything the printed
    4-dp tables round away. ``rows`` entries carry the same keys `main()`
    already accumulates (``mean, s1, s2, cA, on, off``) plus the optional
    ``meas_err_on``/``meas_err_off`` (``None`` if not computed)."""
    return {
        "label": label,
        "rows": [
            {
                "mean_ct": r["mean"],
                "s1": r["s1"],
                "s2": r["s2"],
                "c_A": r["cA"],
                "c_phi_deg": float(np.degrees(np.arctan2(r["s2"], r["s1"]))),
                "O_on": r["on"],
                "O_off": r["off"],
                "measurement_error_on": r.get("meas_err_on"),
                "measurement_error_off": r.get("meas_err_off"),
            }
            for r in rows
        ],
    }


def write_scale_json(path: Path, payload: Dict[str, Any]) -> None:
    """Persist `build_scale_payload`'s dict as full-precision JSON (I4) --
    `json.dumps`'s float `repr` round-trips exactly, unlike the printed
    4-dp tables."""
    path.write_text(json.dumps(payload, indent=2))


def scratch_for(label: str) -> Path:
    """Per-generator scratch root under SC so two generators evaluated with
    different --label values don't collide on stale memory-maps (amendment
    #13c)."""
    return SC / label


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate one trained checkpoint across the photon-flux sweep."
    )
    parser.add_argument(
        "--out", type=Path, default=OUT,
        help="checkpoint root under which gs1_frozen/**/best-checkpoint.ckpt is globbed "
             f"(default: {OUT})",
    )
    parser.add_argument(
        "--label", type=str, default="cnn",
        help="tag printed in the table header and used to derive a unique scratch dir",
    )
    parser.add_argument(
        "--skip-anchor", action="store_true", default=False,
        help="skip run_anchor_check() (escape hatch; the anchor runs by default)",
    )
    parser.add_argument(
        "--scale-json-out", type=Path, default=None,
        help="full-precision machine-readable scale-scalar dump path (I4); "
             "defaults to '<out>/<label>_scale.json'",
    )
    return parser


def run_anchor_check() -> float:
    """HARD VALIDATION ANCHOR: direct_placement_canvas + gauged_fidelity at
    s1=s2=1 (no-varpro) on the lines gs1_frozen checkpoint must reproduce
    recon_quality_gate.py's canvas |amp| NCC (~0.97) within ANCHOR_TOL. This
    proves the object-frame alignment used below is the gate's validated
    placement, not the barycentric-canvas center-crop that amendment #14
    flagged as a framing artifact (~0.25-0.31). Raises if the anchor fails --
    no flux-sweep fidelity number is reported in that case."""
    ckpts = glob.glob(str(REPO / ANCHOR_CKPT_GLOB), recursive=True)
    assert ckpts, f"no anchor checkpoint under {ANCHOR_CKPT_GLOB}"
    dc, mc, tc, _ic, _gc = build_configs(ARM_TABLE["gs1_frozen"], 16, 25)
    scr = SC / "anchor"
    shutil.rmtree(scr, ignore_errors=True)
    ds = build_test_dataset(ANCHOR_NPZ, mc, dc, tc, scr)
    model, _cfgs = load_checkpoint_with_configs(ckpts[0], PtychoPINN_Lightning, device="cpu")
    model = model.cpu().eval()
    canvas, mask, obj, _td, _probe, _patches_raw = direct_placement_canvas(model, ds, ANCHOR_NPZ)
    amp_ncc, _phase_mae = gauged_fidelity(canvas, mask, obj, 1.0, 1.0)
    print(f"ANCHOR (lines gs1_frozen, no-varpro, direct-placement) |amp| NCC = "
          f"{amp_ncc:.4f} (gate target ~{ANCHOR_TARGET}, tol {ANCHOR_TOL})")
    if abs(amp_ncc - ANCHOR_TARGET) > ANCHOR_TOL:
        raise RuntimeError(
            f"Anchor validation FAILED: direct-placement |amp| NCC {amp_ncc:.4f} "
            f"does not reproduce recon_quality_gate.py's ~{ANCHOR_TARGET} "
            f"(tol {ANCHOR_TOL}). The flux-sweep fidelity table below would not "
            f"be trustworthy -- refusing to compute it."
        )
    return amp_ncc


def main() -> int:
    args = build_parser().parse_args()
    out = args.out
    scratch_root = scratch_for(args.label)

    anchor_ncc = None
    if not args.skip_anchor:
        anchor_ncc = run_anchor_check()
        print()

    ckpts = glob.glob(str(out / "gs1_frozen/**/best-checkpoint.ckpt"), recursive=True)
    assert ckpts, f"no checkpoint under {out}/gs1_frozen"
    ckpt = ckpts[0]
    print("label:", args.label)
    print("checkpoint:", ckpt)
    model, cfgs = load_checkpoint_with_configs(ckpt, PtychoPINN_Lightning, device="cpu")
    model.eval()
    dc, mc, tc, _ic, _dg = cfgs
    tc = dataclasses.replace(tc, device="cuda")

    # truth object amplitude level for reference
    obj = np.load(DS / f"fluxsweep_N{mc_N(mc)}_mean100_test.npz", allow_pickle=True)["objectGuess"]
    print(f"truth |O|: median(support)={np.median(np.abs(obj)[np.abs(obj)>0]):.4f} "
          f"mean={np.abs(obj).mean():.4f}\n")

    truth = np.asarray(obj)
    # --- SCALE TABLE (s1/s2 absorb the flux) ---
    print("== SCALE (varpro solve) ==")
    hdr = (f"{'mean_ct':>8} {'sqrt(ct)':>9} {'s1':>9} {'s2':>9} {'c_A=|s|':>9} "
           f"{'cphi_deg':>9} {'|O|_on':>9} {'|O|_off':>9}")
    print(hdr); print("-" * len(hdr))
    rows = []
    for mean in MEANS:
        npz = DS / f"fluxsweep_N{mc_N(mc)}_mean{mean}_test.npz"
        scr = scratch_root / f"m{mean}"
        shutil.rmtree(scr, ignore_errors=True)
        ds = build_test_dataset(npz, mc, dc, tc, scr)
        recon_on, _pre, s1, s2 = run_inference_variant(model, ds, tc, dc, mc, "uniform", True)
        recon_off, _pre2, _s1o, _s2o = run_inference_variant(model, ds, tc, dc, mc, "uniform", False)
        cA = float(np.hypot(s1, s2))
        cphi = float(np.degrees(np.arctan2(s2, s1)))
        on, off = support_median_abs(recon_on), support_median_abs(recon_off)
        print(f"{mean:>8} {mean**0.5:>9.3f} {s1:>9.4f} {s2:>9.4f} {cA:>9.4f} "
              f"{cphi:>9.3f} {on:>9.4f} {off:>9.4f}")
        # raw (phase-aligned, NOT amp-gauged) MAE vs truth, per the harness convention
        m_on = compute_metrics(recon_on, truth)
        m_off = compute_metrics(recon_off, truth)

        # Trustworthy gauge-quotiented fidelity: direct-placement canvas built
        # from this SAME fresh-scratch ds (already satisfies amendment #13c),
        # varpro s1/s2 folded in for ON, identity for OFF (run_anchor_check
        # validated this methodology above before any of these are trusted).
        model = model.cpu().eval()
        canvas_raw, mask, obj_full, td, probe, patches_raw = direct_placement_canvas(
            model, ds, npz
        )
        fid_on = gauged_fidelity(canvas_raw, mask, obj_full, s1, s2)
        fid_off = gauged_fidelity(canvas_raw, mask, obj_full, 1.0, 1.0)

        # I1: measurement-domain (Fourier) error -- forward-simulate predicted
        # diffraction amplitude from the SAME collated batch/raw predicted
        # patches direct_placement_canvas already computed above (varpro
        # s1/s2 folded in for ON, identity for OFF, matching gauged_fidelity's
        # real-space convention) and compare against the dataset's own
        # measured diffraction amplitude.
        measured_amp = td["images"].detach().cpu().numpy()
        meas_err_on = measurement_domain_error(
            predicted_diffraction_amplitude(
                mc, dc, patches_raw, td["coords_relative"], probe,
                td["rms_scaling_constant"], td["experiment_id"], s1, s2,
            ),
            measured_amp,
        )
        meas_err_off = measurement_domain_error(
            predicted_diffraction_amplitude(
                mc, dc, patches_raw, td["coords_relative"], probe,
                td["rms_scaling_constant"], td["experiment_id"], 1.0, 1.0,
            ),
            measured_amp,
        )

        rows.append(dict(mean=mean, s1=s1, s2=s2, cA=cA, on=on, off=off,
                         m_on=m_on, m_off=m_off, fid_on=fid_on, fid_off=fid_off,
                         meas_err_on=meas_err_on, meas_err_off=meas_err_off,
                         n_cov=int(mask.sum())))

    # --- RAW MAE TABLE (phase-aligned barycentric canvas; validated, unchanged) ---
    print("\n== RAW MAE vs truth (phase-aligned barycentric canvas only; NOT gauge-quotiented) ==")
    h2 = (f"{'mean_ct':>8} | {'amp_mae ON':>10} {'OFF':>8} | {'phase_mae ON':>13} {'OFF':>8}")
    print(h2); print("-" * len(h2))
    for r in rows:
        print(f"{r['mean']:>8} | {r['m_on']['amp_mae']:>10.4f} {r['m_off']['amp_mae']:>8.4f} | "
              f"{r['m_on']['phase_mae']:>13.4f} {r['m_off']['phase_mae']:>8.4f}")

    print(f"\ncheckpoint physics_forward_mode: {mc.physics_forward_mode}")
    mode_warning = physics_forward_mode_warning(mc.physics_forward_mode)
    if mode_warning is not None:
        print(f"\n{'!' * 70}\n{mode_warning}\n{'!' * 70}")

    # --- MEASUREMENT-DOMAIN (FOURIER) ERROR TABLE (I1, plan L52/L89) ---
    print("\n== MEASUREMENT-DOMAIN (FOURIER) ERROR (relative L2, forward-simulated "
          "vs measured diffraction amplitude) ==")
    h_fourier = f"{'mean_ct':>8} | {'meas_err ON':>11} {'OFF':>8}"
    print(h_fourier); print("-" * len(h_fourier))
    for r in rows:
        print(f"{r['mean']:>8} | {r['meas_err_on']:>11.4f} {r['meas_err_off']:>8.4f}")

    # --- TRUSTWORTHY FIDELITY TABLE (direct-placement, gauge-quotiented) ---
    anchor_str = f"{anchor_ncc:.4f}" if anchor_ncc is not None else "skipped"
    print(f"\n== FIDELITY vs truth (label={args.label}, direct-placement, "
          f"gauge-quotiented; anchor |amp| NCC={anchor_str}) ==")
    print("does varpro change fidelity? ampNCC/phaseMAE removes the isotropic c_A "
          "scale+rotation only -- NOT the s1!=s2 real/imag anisotropy, per the physics note.")
    h3 = (f"{'mean_ct':>8} | {'ampNCC_gauged ON':>17} {'OFF':>8} | "
          f"{'phaseMAE_gauged ON':>19} {'OFF':>8} | {'n_cov':>6}")
    print(h3); print("-" * len(h3))
    for r in rows:
        print(f"{r['mean']:>8} | {r['fid_on'][0]:>17.4f} {r['fid_off'][0]:>8.4f} | "
              f"{r['fid_on'][1]:>19.4f} {r['fid_off'][1]:>8.4f} | {r['n_cov']:>6}")

    # --- SCALING CHECK ---
    ref = next(r for r in rows if r['mean'] == 100)
    print("\nscaling vs mean=100 (predict s ~ sqrt(count) => ratio ~ sqrt(count/100)):")
    for r in rows:
        exp = (r['mean'] / 100.0) ** 0.5
        print(f"  mean={r['mean']:>7}: c_A ratio={r['cA']/ref['cA']:>8.3f}  expected sqrt={exp:>8.3f}  "
              f"|O|_on ratio={r['on']/ref['on']:>7.3f}  |O|_off ratio={r['off']/ref['off']:>7.3f}")

    # --- I4: full-precision machine-readable dump of the solved scalars ---
    scale_json_path = args.scale_json_out or (out / f"{args.label}_scale.json")
    write_scale_json(scale_json_path, build_scale_payload(args.label, rows))
    print(f"\nwrote full-precision scale scalars to {scale_json_path}")
    return 0


def mc_N(mc) -> int:
    return int(getattr(mc, "N", 64))


if __name__ == "__main__":
    sys.exit(main())
