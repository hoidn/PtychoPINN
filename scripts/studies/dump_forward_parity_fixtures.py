"""Freeze deterministic forward-model + loss parity fixtures (Task 1.4).

These NPZs are the cross-branch physics oracle Task 2.6 (on ``fno-stable``)
will check its port of ``RectangularScaledDiffraction`` against. See
``docs/plans/2026-07-01-varpro-ablation-phase1-findings.md`` for why loss
scalars are frozen alongside forward tensors: this branch's ``PoissonLoss``
consumes the forward output as an unsquared intensity while ``MAELoss``
re-squares it internally, so forward-output parity alone cannot catch a
loss-units mistake made while porting the physics.

Case matrix (see ``CASES`` below) and documented collapses:

* ``object_big=False`` (case ``c1_bigF``) never reaches
  ``training_patch_weighting`` at all -- ``ForwardModel.forward`` only reads
  that knob inside its ``if self.object_big:`` branch -- so probe vs uniform
  weighting is provably identical there. Verified at generation time (not
  merely assumed) and only one representative ("probe") is stored.
* ``rect_s1s2_trainable`` only toggles ``requires_grad`` on
  ``RectangularScaledDiffraction.s1``/``.s2``; both initialize to
  ``torch.ones(num_datasets)`` regardless of the flag, so forward output (and
  therefore both loss scalars) are bitwise identical between
  frozen/trainable-at-init for every case. Verified at generation time; only
  the main-default ("trainable=True") representative is stored per case, and
  ``model_config["rect_s1s2_trainable"]`` in the fixture's own metadata
  records which ``requires_grad`` state was exercised.

This collapses the nominal 3 (C-shape) x 2 (weighting) x 2 (s1s2) = 12 cases
down to 5 physical fixture files:
``c1_bigF``, ``c1_bigT_probe``, ``c1_bigT_uniform``, ``c4_bigT_probe``,
``c4_bigT_uniform``.

Determinism: every case resets ``torch.manual_seed(0)`` before drawing any
random tensor (x, I_raw); ``ForwardModel`` itself has no randomly-initialized
parameters (s1/s2 are deterministic ones; ``IntensityScalerModule`` is unused
inside ``ForwardModel.forward`` and holds no parameters), so re-running this
script produces byte-identical NPZs *within a single torch build*.

NOT RNG-reproducible across torch builds: fresh generation seeds only the
GLOBAL torch RNG (bare ``torch.manual_seed(0)`` + ``torch.randn``/``torch.randint``,
no isolated ``torch.Generator``). torch's CPU sampling algorithms are not
guaranteed bit-identical across versions/builds. The fixtures currently
committed were frozen on the ``varpro-ablation`` branch (see each fixture's own
``metadata_json["commit"]``); a 2026-07-06 pristine-code control regeneration on
a different torch build drifted the sign-fix-incapable ``c1_bigF``
(``object_big=False``) case by 5 orders of magnitude on ``expected_forward``,
proving the drift was pure RNG/environment noise, not a physics change. Fresh
generation (no ``--refreeze``) must therefore never be used to re-freeze these
committed fixtures.

The sanctioned re-freeze path is ``--refreeze``: it loads each existing
``c*.npz``, keeps every input array (``x_real``/``x_imag``, ``positions``,
``probe_real``/``probe_imag``, ``scale``, ``experiment_ids``, ``I_raw``,
``metadata_json``) byte-identical, and recomputes ONLY the three
``expected_*`` outputs by replaying them through the (possibly-fixed) forward
model and loss -- no RNG involved anywhere in that path, so any resulting
delta is isolated to an actual physics/code change.

Canvas-padding parity: for ``object_big=True`` cases, replay restores
``ptycho_torch/helper.py``'s ORIGINAL (varpro-ablation-era) canvas convention
(``get_padded_size`` buffer = ``max_position_jitter``) for the duration of the
replay, exactly mirroring
``tests/torch/test_rectangular_scaled_forward.py::test_rectangular_bigT_parity_under_main_padding``'s
monkeypatch. fno-stable's current default (buffer=0, commit ``ba3f705d``) is a
separate, already out-of-scope canvas-size difference belonging to the
physics-reconciliation backlog (see that test's own ``_OBJECT_BIG_XFAIL``
comment) -- without restoring the original convention here, a replay would
silently re-anchor these fixtures onto that unrelated difference too, on top
of whatever physics/code change is actually being validated.

Loss-units discriminating power: the ``scale`` input is multiplied by
``INTENSITY_SCALE_FACTOR`` (see below) so forward outputs land ``O(1-10)``,
the same order as the synthetic ``I_raw`` counts -- otherwise a "forgot to
square pred before MAE" convention mistake shifts the frozen loss by less
than the test's own assert_close tolerance and would silently pass. See
``task-1.4-report.md`` for the per-case convention-mistake-delta table.
"""
import argparse
import contextlib
import dataclasses
import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from ptycho_torch.config_params import DataConfig, ModelConfig
from ptycho_torch.model import ForwardModel, PoissonLoss, MAELoss

REPO_ROOT = Path(__file__).resolve().parents[2]
FLY001_NPZ = REPO_ROOT / "datasets" / "fly" / "fly001.npz"

PROBE_SEMANTICS = (
    "bare probe inside RectangularScaledDiffraction; main probe_mask=None "
    "semantics -- no mask"
)

N = 64  # fly001 probeGuess is (64, 64); matches DataConfig default N.

# Loss-units discriminating power: at scale=1.0 (main's raw "ones" scale),
# ForwardModel.forward's I_pred lands ~1e-4-2.4e-4 -- negligible next to the
# synthetic I_raw counts (0-49), so a "forgot to square pred before MAE"
# convention mistake (MAELoss.forward: `self.mae(pred**2, raw)`, dropping the
# `**2`) shifts the frozen MAE scalar by only ~1e-4-2.4e-4, *below* the
# test's effective assert_close tolerance (rtol=1e-5, atol=1e-6 against
# expected~24.4 -> ~2.45e-4) -- the fixture would silently pass a units bug.
# Multiplying the `scale` input (quadratic in the FFT intensity, since
# I_pred = |FFT(scale * exit_wave)|**2) by this constant lifts forward
# outputs to O(1-10), the same order as I_raw, so unsquared-vs-squared
# convention mistakes produce deltas an order of magnitude past tolerance.
# See task-1.4-report.md ("Fix: portability + loss-oracle strengthening").
INTENSITY_SCALE_FACTOR = 200.0


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True,
        ).strip()
    except Exception:
        return "unknown"


def _load_probe() -> torch.Tensor:
    """Load fly001's single-mode probeGuess, cast to complex64. Shape (N, N)."""
    data = np.load(FLY001_NPZ)
    probe = data["probeGuess"]
    assert probe.ndim == 2, f"expected a single-mode 2D probe, got shape {probe.shape}"
    assert probe.shape == (N, N), f"expected ({N}, {N}) probe, got {probe.shape}"
    return torch.from_numpy(probe).to(torch.complex64)


def _expand_probe(probe_2d: torch.Tensor, B: int, C: int) -> torch.Tensor:
    """Broadcast the bare (N, N) probe to production shape (B, C, P=1, N, N).

    Mirrors ``ptycho_torch/dataloader.py``'s own
    ``.unsqueeze(1).expand(-1, channels, ...)`` broadcast of a single stored
    probe over channels; no memory is duplicated (stride-0 view).
    """
    return probe_2d.view(1, 1, 1, N, N).expand(B, C, 1, N, N)


def _run_forward(data_cfg: DataConfig, model_cfg: ModelConfig, probe_2d: torch.Tensor,
                  x: torch.Tensor, positions: torch.Tensor, scale: torch.Tensor,
                  experiment_ids: torch.Tensor) -> torch.Tensor:
    model = ForwardModel(model_cfg, data_cfg)
    model.eval()
    probe = _expand_probe(probe_2d, x.shape[0], x.shape[1]) if model_cfg.object_big else probe_2d
    with torch.no_grad():
        # I_measured=None: ForwardModel.forward always calls
        # RectangularScaledDiffraction with autograd=True, and I_raw is only
        # read on the autograd=False (variable-projection) branch -- it is a
        # dead argument on this call path.
        return model.forward(x, None, positions, probe, scale, experiment_ids=experiment_ids)


def _c1_positions(B: int, C: int) -> torch.Tensor:
    # A single-patch group (grid_size=(1,1)) has no relative neighbor offset.
    return torch.zeros(B, C, 1, 2, dtype=torch.float32)


def _c4_positions(B: int, C: int) -> torch.Tensor:
    # Small overlapping 2x2-grid offsets (pixels, (x, y)); N=64 patches
    # shifted by only +/-3px overlap heavily, well within the padded canvas
    # margin (M - N = 14 for the default grid_size=(2,2)/max_neighbor_distance=3.0).
    assert (B, C) == (1, 4)
    return torch.tensor(
        [[[-3.0, -3.0]], [[3.0, -3.0]], [[-3.0, 3.0]], [[3.0, 3.0]]],
        dtype=torch.float32,
    ).view(1, 4, 1, 2)


CASES = [
    dict(
        name="c1_bigF",
        B=1, C=1,
        data_config_kwargs=dict(N=N, C=1, grid_size=(1, 1)),
        model_config_kwargs=dict(
            object_big=False, C_forward=1, C_model=1, num_datasets=1,
            training_patch_weighting="probe", rect_s1s2_trainable=True,
        ),
        positions_fn=_c1_positions,
        collapse=("weighting", "rect_s1s2_trainable"),
    ),
    dict(
        name="c1_bigT_probe",
        B=1, C=1,
        data_config_kwargs=dict(N=N, C=1, grid_size=(1, 1)),
        model_config_kwargs=dict(
            object_big=True, C_forward=1, C_model=1, num_datasets=1,
            training_patch_weighting="probe", rect_s1s2_trainable=True,
        ),
        positions_fn=_c1_positions,
        collapse=("rect_s1s2_trainable",),
    ),
    dict(
        name="c1_bigT_uniform",
        B=1, C=1,
        data_config_kwargs=dict(N=N, C=1, grid_size=(1, 1)),
        model_config_kwargs=dict(
            object_big=True, C_forward=1, C_model=1, num_datasets=1,
            training_patch_weighting="uniform", rect_s1s2_trainable=True,
        ),
        positions_fn=_c1_positions,
        collapse=("rect_s1s2_trainable",),
    ),
    dict(
        name="c4_bigT_probe",
        B=1, C=4,
        data_config_kwargs=dict(N=N, C=4, grid_size=(2, 2)),
        model_config_kwargs=dict(
            object_big=True, C_forward=4, C_model=4, num_datasets=1,
            training_patch_weighting="probe", rect_s1s2_trainable=True,
        ),
        positions_fn=_c4_positions,
        collapse=("rect_s1s2_trainable",),
    ),
    dict(
        name="c4_bigT_uniform",
        B=1, C=4,
        data_config_kwargs=dict(N=N, C=4, grid_size=(2, 2)),
        model_config_kwargs=dict(
            object_big=True, C_forward=4, C_model=4, num_datasets=1,
            training_patch_weighting="uniform", rect_s1s2_trainable=True,
        ),
        positions_fn=_c4_positions,
        collapse=("rect_s1s2_trainable",),
    ),
]

WEIGHTING_COLLAPSE_NOTE = (
    "object_big=False skips reassembly (training_patch_weighting is only "
    "read inside ForwardModel.forward's `if self.object_big:` branch); "
    "probe vs uniform forward outputs verified torch.testing.assert_close "
    "at generation time; only 'probe' is stored."
)

S1S2_COLLAPSE_NOTE_TMPL = (
    "s1/s2 both initialize to torch.ones(num_datasets) regardless of "
    "requires_grad; forward output verified torch.testing.assert_close "
    "between rect_s1s2_trainable=True/False at initialization (before any "
    "optimizer step) at generation time. Only the main-default "
    "(trainable=True) representative is stored; requires_grad at "
    "initialization for the stored fixture is {trainable}."
)


def _generate_case(case: dict, probe_2d: torch.Tensor, commit: str) -> dict:
    B, C = case["B"], case["C"]
    data_cfg = DataConfig(**case["data_config_kwargs"])
    model_cfg = ModelConfig(**case["model_config_kwargs"])

    torch.manual_seed(0)
    x = torch.randn(B, C, N, N, dtype=torch.complex64)
    positions = case["positions_fn"](B, C)
    # PoissonLoss validates its `raw` argument against Poisson's support
    # (non-negative integers), so the synthetic measurement must be
    # integer-valued counts, not a continuous [0, 1) draw.
    I_raw = torch.randint(0, 50, (B, C, N, N)).to(torch.float32)
    # See INTENSITY_SCALE_FACTOR docstring above: this is not a physical
    # scale guess, it's chosen so the loss-units oracle actually discriminates.
    scale = torch.ones(B, 1, 1, 1, dtype=torch.float32) * INTENSITY_SCALE_FACTOR
    experiment_ids = torch.zeros(B, dtype=torch.long)

    forward_out = _run_forward(data_cfg, model_cfg, probe_2d, x, positions, scale, experiment_ids)

    poisson_loss = PoissonLoss()(forward_out, I_raw).mean()
    mae_loss = MAELoss()(forward_out, I_raw).mean()

    collapse_notes = {}
    if "weighting" in case["collapse"]:
        alt_weighting = "uniform" if model_cfg.training_patch_weighting == "probe" else "probe"
        alt_cfg = dataclasses.replace(model_cfg, training_patch_weighting=alt_weighting)
        alt_out = _run_forward(data_cfg, alt_cfg, probe_2d, x, positions, scale, experiment_ids)
        torch.testing.assert_close(forward_out, alt_out, rtol=1e-6, atol=1e-7)
        collapse_notes["weighting"] = WEIGHTING_COLLAPSE_NOTE
    if "rect_s1s2_trainable" in case["collapse"]:
        alt_cfg = dataclasses.replace(model_cfg, rect_s1s2_trainable=not model_cfg.rect_s1s2_trainable)
        alt_out = _run_forward(data_cfg, alt_cfg, probe_2d, x, positions, scale, experiment_ids)
        torch.testing.assert_close(forward_out, alt_out, rtol=1e-6, atol=1e-7)
        collapse_notes["rect_s1s2_trainable"] = S1S2_COLLAPSE_NOTE_TMPL.format(
            trainable=model_cfg.rect_s1s2_trainable,
        )

    metadata = {
        "case": case["name"],
        "data_config": dataclasses.asdict(data_cfg),
        "model_config": dataclasses.asdict(model_cfg),
        "shapes": {"B": B, "C": C, "N": N, "P": 1},
        "probe_source": "datasets/fly/fly001.npz:probeGuess (single mode, cast complex64)",
        "probe_semantics": PROBE_SEMANTICS,
        "branch": "varpro-ablation",
        "commit": commit,
        "seed": 0,
        "collapsed": collapse_notes,
    }

    return dict(
        x_real=x.real.numpy(),
        x_imag=x.imag.numpy(),
        positions=positions.numpy(),
        probe_real=probe_2d.real.numpy(),
        probe_imag=probe_2d.imag.numpy(),
        scale=scale.numpy(),
        experiment_ids=experiment_ids.numpy(),
        I_raw=I_raw.numpy(),
        expected_forward=forward_out.numpy().astype(np.float32),
        expected_poisson_loss=np.array(poisson_loss.item(), dtype=np.float32),
        expected_mae_loss=np.array(mae_loss.item(), dtype=np.float32),
        metadata_json=np.array(json.dumps(metadata)),
    )


def _init_field_names(cls) -> set:
    """Dataclass init-field names -- copied from
    tests/torch/test_rectangular_scaled_forward.py::_init_field_names so dataclass
    drift (fields added/removed since a fixture was frozen) can't break the rebuild.
    """
    return {f.name for f in dataclasses.fields(cls) if f.init}


def _filtered_kwargs(stored: dict, cls) -> dict:
    """Copied from tests/torch/test_rectangular_scaled_forward.py::_filtered_kwargs."""
    return {k: v for k, v in stored.items() if k in _init_field_names(cls)}


def _rebuild_configs_from_metadata(metadata: dict):
    """Rebuild DataConfig/ModelConfig from stored ``metadata_json``, forcing
    ``physics_forward_mode='rectangular_scaled'`` -- exactly as
    tests/torch/test_rectangular_scaled_forward.py::_rebuild_configs does. This is
    the mode/loss contract these fixtures are actually checked against (see
    ``_refreeze_one``'s use of ``RectangularPoissonLoss``/``RectangularMAELoss``
    below), not this script's own ``_generate_case`` default (which predates the
    amplitude/rectangular_scaled split and would silently reproduce the wrong
    physics if used to rebuild configs for a re-freeze).
    """
    data_cfg = DataConfig(**_filtered_kwargs(metadata["data_config"], DataConfig))
    model_kwargs = _filtered_kwargs(metadata["model_config"], ModelConfig)
    model_kwargs["physics_forward_mode"] = "rectangular_scaled"
    model_cfg = ModelConfig(**model_kwargs)
    return data_cfg, model_cfg


@contextlib.contextmanager
def _original_canvas_padding(model_cfg: ModelConfig):
    """Temporarily restore ``ptycho_torch/helper.py::get_padded_size``'s ORIGINAL
    (varpro-ablation-era) canvas convention (buffer=``max_position_jitter``) for
    ``object_big=True`` configs -- mirrors
    tests/torch/test_rectangular_scaled_forward.py::
    test_rectangular_bigT_parity_under_main_padding's monkeypatch exactly. No-op
    when ``object_big`` is False (``get_padded_size`` isn't on that call path).
    """
    if not model_cfg.object_big:
        yield
        return
    import ptycho_torch.helper as hh
    orig_get_padded_size = hh.get_padded_size
    orig_get_bigN = hh.get_bigN
    hh.get_padded_size = lambda dc, mc: orig_get_bigN(dc, mc) + mc.max_position_jitter
    try:
        yield
    finally:
        hh.get_padded_size = orig_get_padded_size


def _refreeze_one(path: Path) -> dict:
    """Replay re-freeze for one fixture: recompute ONLY ``expected_forward`` /
    ``expected_poisson_loss`` / ``expected_mae_loss`` from this fixture's OWN
    frozen input arrays -- no RNG redraw anywhere. Inputs and ``metadata_json``
    are carried through byte-identical (including ``branch``/``commit`` fields,
    left as originally recorded). Canvas padding is held at its original
    convention (see ``_original_canvas_padding``) so only the physics/code
    change under test shows up in the recomputed outputs.
    """
    with np.load(path) as npz:
        stored = {k: npz[k] for k in npz.files}

    metadata = json.loads(str(stored["metadata_json"]))
    data_cfg, model_cfg = _rebuild_configs_from_metadata(metadata)

    x = torch.from_numpy(stored["x_real"] + 1j * stored["x_imag"]).to(torch.complex64)
    positions = torch.from_numpy(stored["positions"]).to(torch.float32)
    scale = torch.from_numpy(stored["scale"]).to(torch.float32)
    experiment_ids = torch.from_numpy(stored["experiment_ids"]).to(torch.long)
    I_raw = torch.from_numpy(stored["I_raw"]).to(torch.float32)
    probe_2d = torch.from_numpy(stored["probe_real"] + 1j * stored["probe_imag"]).to(torch.complex64)

    with _original_canvas_padding(model_cfg):
        forward_out = _run_forward(data_cfg, model_cfg, probe_2d, x, positions, scale, experiment_ids)

        # RectangularPoissonLoss/RectangularMAELoss match physics_forward_mode=
        # 'rectangular_scaled' (see test_rectangular_scaled_forward.py::
        # test_rectangular_loss_parity); the base PoissonLoss/MAELoss this script
        # otherwise imports are the amplitude-domain variants and do NOT reproduce
        # these fixtures' frozen loss scalars.
        from ptycho_torch.model import RectangularPoissonLoss, RectangularMAELoss
        poisson_loss = RectangularPoissonLoss()(forward_out, I_raw).mean()
        mae_loss = RectangularMAELoss()(forward_out, I_raw).mean()

        # Extra validation: re-run the collapse cross-checks the metadata claims
        # were verified at original generation time (weighting / rect_s1s2_trainable).
        collapsed = metadata.get("collapsed", {})
        if "weighting" in collapsed:
            alt_weighting = "uniform" if model_cfg.training_patch_weighting == "probe" else "probe"
            alt_cfg = dataclasses.replace(model_cfg, training_patch_weighting=alt_weighting)
            alt_out = _run_forward(data_cfg, alt_cfg, probe_2d, x, positions, scale, experiment_ids)
            torch.testing.assert_close(forward_out, alt_out, rtol=1e-6, atol=1e-7)
        if "rect_s1s2_trainable" in collapsed:
            alt_cfg = dataclasses.replace(model_cfg, rect_s1s2_trainable=not model_cfg.rect_s1s2_trainable)
            alt_out = _run_forward(data_cfg, alt_cfg, probe_2d, x, positions, scale, experiment_ids)
            torch.testing.assert_close(forward_out, alt_out, rtol=1e-6, atol=1e-7)

    refrozen = dict(stored)
    refrozen["expected_forward"] = forward_out.numpy().astype(np.float32)
    refrozen["expected_poisson_loss"] = np.array(poisson_loss.item(), dtype=np.float32)
    refrozen["expected_mae_loss"] = np.array(mae_loss.item(), dtype=np.float32)
    return refrozen


def _refreeze_all(in_dir: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(in_dir.glob("c*.npz")):
        # Mirror test_rectangular_scaled_forward.py::_fixture_paths' own filter so
        # non-CASES fixtures sharing the "c*" glob (e.g.
        # compute_loss_c4_regression.npz) are skipped, not corrupted.
        with np.load(path) as probe_npz:
            if "metadata_json" not in probe_npz.files:
                continue
        arrays = _refreeze_one(path)
        out_path = out_dir / path.name
        np.savez_compressed(out_path, **arrays)
        print(f"refroze {path} -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="tests/fixtures/varpro_parity",
                         help="Output directory for the frozen NPZ fixtures.")
    parser.add_argument("--refreeze", action="store_true",
                         help="Replay re-freeze mode (see module docstring): "
                              "recompute expected_* outputs for every existing "
                              "c*.npz in --output from ITS OWN frozen inputs, no "
                              "RNG redraw. This is the only sanctioned way to "
                              "re-freeze these fixtures.")
    parser.add_argument("--refreeze-output", default=None,
                         help="Output directory for --refreeze (default: same as "
                              "--output, i.e. in-place). Point at a scratch "
                              "directory for dry-run validation.")
    args = parser.parse_args()

    out_dir = Path(args.output)

    if args.refreeze:
        refreeze_out = Path(args.refreeze_output) if args.refreeze_output else out_dir
        _refreeze_all(out_dir, refreeze_out)
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    probe_2d = _load_probe()
    commit = _git_commit()

    for case in CASES:
        arrays = _generate_case(case, probe_2d, commit)
        out_path = out_dir / f"{case['name']}.npz"
        np.savez_compressed(out_path, **arrays)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
