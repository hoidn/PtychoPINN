"""Publication-style figure for the photon-flux-sweep result (amendment #14).

Renders a 2x2 composite figure demonstrating that the varpro dynamic-scaling
constants (s1, s2) absorb the incident-photon-count scale, while the
gauge-quotiented reconstruction fidelity is (slightly) better with varpro
inference OFF at matched scale.

SOURCE OF TRUTH FOR THE NUMBERS BELOW: ``scripts/studies/flux_sweep_eval.py``,
run against the ``gs1_frozen`` checkpoint over the mean-count {1, 100, 10000}
test sets (see ``.artifacts/varpro_ablation/matrix_fluxsweep``). These values
are hardcoded here (rather than re-run) because ``flux_sweep_eval.py`` loads a
checkpoint and performs six full reconstructions; the numbers are transcribed
verbatim from its validated output. Provenance and physical interpretation are
recorded in amendment #14 of ``.superpowers/sdd/plan-amendments-pending.md``
("rect_scaler / s1,s2 PHYSICAL SEMANTICS -- DEFINITIVE").

Definitions (amendment #14, manuscript Eq 5):
  c_A   = sqrt(s1^2 + s2^2)   -- amplitude/photon-scale contrast, predicted
                                 proportional to sqrt(mean_count)
  c_phi = arctan(s2 / s1)     -- phase contrast, predicted flux-invariant
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / ".artifacts/varpro_ablation/composite"
OUT_PNG = OUT_DIR / "flux_sweep.png"

# --- VALIDATED NUMBERS (scripts/studies/flux_sweep_eval.py, amendment #14) ---
MEAN_COUNT = np.array([1, 100, 10000], dtype=float)
C_A = np.array([0.913, 9.279, 92.96])
C_PHI_DEG = np.array([-64.9, -65.8, -65.4])
ABS_O_ON = np.array([0.724, 7.625, 76.25])
ABS_O_OFF = np.array([0.964, 1.025, 1.024])
AMP_NCC_ON = np.array([0.835, 0.952, 0.953])
AMP_NCC_OFF = np.array([0.818, 0.975, 0.975])
PHASE_MAE_ON = np.array([0.171, 0.167, 0.168])
PHASE_MAE_OFF = np.array([0.081, 0.045, 0.045])
TRUTH_ABS_O_MEDIAN = 0.6415

REF_IDX = 1  # mean_count == 100, the sqrt(flux) normalization anchor

# Sanity checks mirroring amendment #14's reported findings ("EXACT sqrt(flux)"
# scaling of c_A; flux-invariant c_phi), so this figure cannot silently drift
# from the source-of-truth numbers above without failing loudly.
_ca_ratio = C_A / C_A[REF_IDX]
_ca_expected = np.sqrt(MEAN_COUNT / MEAN_COUNT[REF_IDX])
assert np.allclose(_ca_ratio, _ca_expected, atol=0.02), (
    "c_A no longer tracks sqrt(flux) within tolerance -- rerun flux_sweep_eval.py "
    "and update the hardcoded numbers."
)
assert np.ptp(C_PHI_DEG) < 2.0, (
    "c_phi is no longer flux-invariant within tolerance -- rerun flux_sweep_eval.py "
    "and update the hardcoded numbers."
)


def make_figure():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 9.5))
    ax1, ax2, ax3, ax4 = axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]

    # --- Panel 1: c_A vs mean_count (log-log), sqrt(flux) reference ---
    ax1.loglog(MEAN_COUNT, C_A, "o-", color="tab:blue", label=r"$c_A=\sqrt{s_1^2+s_2^2}$ (solved)")
    ref = np.sqrt(MEAN_COUNT) * (C_A[REF_IDX] / np.sqrt(MEAN_COUNT[REF_IDX]))
    ax1.loglog(MEAN_COUNT, ref, "k--", label=r"$\sqrt{\mathrm{mean\_count}}$ (normalized at mean=100)")
    for x, y, ratio, exp in zip(MEAN_COUNT, C_A, _ca_ratio, _ca_expected):
        ax1.annotate(f"ratio {ratio:.3f}\n(vs √ {exp:.3f})", (x, y),
                     textcoords="offset points", xytext=(-45, 14), fontsize=8)
    ax1.set_xlabel("mean photon count")
    ax1.set_ylabel(r"$c_A$ (amplitude contrast)")
    ax1.set_title(r"$c_A$ scales as $\sqrt{\mathrm{flux}}$")
    ax1.set_ylim(0.5, 300)
    ax1.legend(fontsize=8, loc="lower right")
    ax1.grid(True, which="both", alpha=0.3)

    # --- Panel 2: c_phi vs mean_count (semilog-x), flux-invariant ---
    ax2.semilogx(MEAN_COUNT, C_PHI_DEG, "o-", color="tab:purple", label=r"$c_\phi=\arctan(s_2/s_1)$ (solved)")
    ax2.axhline(np.mean(C_PHI_DEG), color="k", linestyle="--", alpha=0.5,
                label=f"mean = {np.mean(C_PHI_DEG):.1f}°")
    ax2.set_xlabel("mean photon count")
    ax2.set_ylabel(r"$c_\phi$ (phase contrast, deg)")
    ax2.set_title(r"$c_\phi$ is flux-invariant ($\approx -65^\circ$)")
    ax2.legend(fontsize=8, loc="best")
    ax2.grid(True, which="both", alpha=0.3)

    # --- Panel 3: |O| level vs mean_count (semilog-x), ON vs OFF ---
    ax3.semilogx(MEAN_COUNT, ABS_O_ON, "o-", color="tab:red", label="varpro ON (∝flux)")
    ax3.semilogx(MEAN_COUNT, ABS_O_OFF, "s-", color="tab:green", label="varpro OFF (flat)")
    ax3.axhline(TRUTH_ABS_O_MEDIAN, color="k", linestyle=":", label=f"truth |O| median = {TRUTH_ABS_O_MEDIAN}")
    ax3.set_xlabel("mean photon count")
    ax3.set_ylabel(r"reconstructed $|O|$ (median)")
    ax3.set_title(r"$|O|$ level: varpro ON vs OFF")
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(True, which="both", alpha=0.3)
    ax3.text(0.30, 0.55,
              "Neither variant hits truth without a gauge\n"
              "(probe-normalization convention, amendment #14).",
              transform=ax3.transAxes, fontsize=7.5, va="center", ha="left", style="italic")

    # --- Panel 4: gauge-quotiented fidelity vs mean_count (semilog-x) ---
    l1, = ax4.semilogx(MEAN_COUNT, AMP_NCC_ON, "o-", color="tab:blue", label="|amp| NCC, varpro ON")
    l2, = ax4.semilogx(MEAN_COUNT, AMP_NCC_OFF, "s-", color="tab:cyan", label="|amp| NCC, varpro OFF")
    ax4.set_xlabel("mean photon count")
    ax4.set_ylabel("gauge-quotiented |amp| NCC")
    ax4.set_title("Fidelity vs truth: varpro ON is slightly worse")
    ax4.grid(True, which="both", alpha=0.3)

    ax4b = ax4.twinx()
    l3, = ax4b.semilogx(MEAN_COUNT, PHASE_MAE_ON, "o--", color="tab:orange", label="phase MAE, varpro ON")
    l4, = ax4b.semilogx(MEAN_COUNT, PHASE_MAE_OFF, "s--", color="tab:red", label="phase MAE, varpro OFF")
    ax4b.set_ylabel("gauge-quotiented phase MAE (rad)")

    lines = [l1, l2, l3, l4]
    ax4.legend(lines, [l.get_label() for l in lines], fontsize=7.5, loc="center right")

    fig.suptitle(
        "Dynamic scaling (s1, s2) absorbs incident photon flux\n"
        "(lines, gs1_frozen)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig = make_figure()
    fig.savefig(OUT_PNG, dpi=200)
    import matplotlib.pyplot as plt
    plt.close(fig)
    print(f"wrote {OUT_PNG}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
