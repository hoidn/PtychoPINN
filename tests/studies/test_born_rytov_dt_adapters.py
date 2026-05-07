"""Adapter contract tests for the BRDT task-adapter scope.

Covers the bounded preflight surface added in
``2026-04-29-brdt-task-adapters``:

- dataset authority / loader / collator,
- row schema (model/training/input_mode/dataset_id/operator_version/row_status),
- separation between historical Born-image input and sinogram-input contracts,
- Born-init-image derivation (local adjoint backend),
- adapter construction + forward shape for U-Net, FNO vanilla, Hybrid family,
- unnormalize-before-physics guard via the dataset_contract helper,
- loss-term routing on a tiny synthetic batch,
- train/eval CLI argument validation and row-status emission paths.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch

from ptycho_torch.physics import BornRytovForward2D
from scripts.studies.born_rytov_dt import dataset_contract as dc
from scripts.studies.born_rytov_dt import (
    classical,
    data,
    evaluate,
    lightning_module,
    model_based_inverse,
    models,
    reporting,
    run_config,
    train,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SMOKE_MANIFEST_PATH = (
    REPO_ROOT
    / ".artifacts"
    / "NEURIPS-HYBRID-RESNET-2026"
    / "backlog"
    / "2026-04-29-brdt-dataset-preflight"
    / "dataset_manifest.json"
)


# ----------------------------------------------------------------------
# Row schema
# ----------------------------------------------------------------------
def test_required_row_fields_stable():
    fields = run_config.required_row_fields()
    expected = (
        "row_id",
        "model",
        "training",
        "input_mode",
        "dataset_id",
        "operator_version",
        "row_status",
    )
    assert fields == expected


def test_default_roster_uses_born_init_image_only():
    rows = run_config.default_row_roster(
        dataset_id="brdt128_sparse_fullview_preflight",
        operator_version="op-validation",
    )
    assert {r.row_id for r in rows} == {
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
    }
    for row in rows:
        assert row.input_mode == "born_init_image"
        payload = row.to_dict()
        for field in run_config.required_row_fields():
            assert field in payload, f"missing field {field}"


def test_default_roster_hybrid_label_sru_net():
    rows = run_config.default_row_roster(
        dataset_id="brdt128_sparse_fullview_preflight",
        operator_version="op-validation",
        hybrid_label="sru_net",
    )
    by_id = {r.row_id: r for r in rows}
    assert "sru_net" in by_id
    # Visible identity (row_id / paper_label) is "sru_net" / "SRU-Net";
    # internal architecture (model) stays as the actual adapter body
    # (``hybrid_resnet``). The execution-plan contract REQUIRES these
    # two fields to be distinct so the visible label cannot drift into
    # the internal architecture identifier.
    assert by_id["sru_net"].paper_label == "SRU-Net"
    assert by_id["sru_net"].model == run_config.HYBRID_FAMILY_MODEL == "hybrid_resnet"


def test_row_config_rejects_sru_net_as_internal_model():
    # ``sru_net`` is a visible paper label only; setting it as the
    # internal architecture id (``model``) must fail.
    with pytest.raises(ValueError, match="unsupported model"):
        run_config.RowConfig(
            row_id="sru_net",
            model="sru_net",
            training=run_config.DEFAULT_TRAINING_LABEL,
            input_mode="born_init_image",
            dataset_id="d",
            operator_version="o",
        )


def test_row_rejects_direct_sinogram_input():
    with pytest.raises(ValueError, match="direct-sinogram"):
        run_config.RowConfig(
            row_id="bad_sinogram_row",
            model="unet",
            training=run_config.DEFAULT_TRAINING_LABEL,
            input_mode="direct_sinogram",
            dataset_id="brdt128_sparse_fullview_preflight",
            operator_version="op",
        )


def test_row_rejects_pinn_only_label_for_supervised_plus_physics():
    with pytest.raises(ValueError, match="PINN-only"):
        run_config.RowConfig(
            row_id="bad_label",
            model="unet",
            training="PINN",
            input_mode="born_init_image",
            dataset_id="d",
            operator_version="o",
        )


def test_row_accepts_sinogram_input_contract():
    row = run_config.RowConfig(
        row_id="ffno",
        model="ffno",
        training=run_config.DEFAULT_TRAINING_LABEL,
        input_mode="sinogram",
        dataset_id="brdt128_sparse_fullview_preflight",
        operator_version="op",
    )
    assert row.input_mode == "sinogram"


def test_assert_input_mode_supported_accepts_sinogram():
    data.assert_input_mode_supported("sinogram")


def test_assert_input_mode_supported_rejects_legacy_direct_sinogram():
    with pytest.raises(ValueError, match="legacy direct-sinogram"):
        data.assert_input_mode_supported("direct_sinogram")


def test_sinogram_input_roster_uses_measured_sinogram():
    rows = run_config.sinogram_input_row_roster(
        dataset_id=dc.DECISION_SUPPORT_DATASET_NAME,
        operator_version="op",
    )
    assert {row.row_id for row in rows} == {
        "classical_born_backprop",
        "ffno",
        "sru_net",
    }
    assert all(row.input_mode == "sinogram" for row in rows)


def test_make_blocked_row_records_reason():
    row = run_config.make_blocked_row(
        row_id="fno_vanilla",
        model="fno_vanilla",
        training=run_config.DEFAULT_TRAINING_LABEL,
        dataset_id="d",
        operator_version="o",
        blocker_reason="neuralop_unavailable",
        blocker_message="neuralop is not installed",
    )
    payload = row.to_dict()
    assert payload["row_status"] == "blocked"
    assert payload["blocker_reason"] == "neuralop_unavailable"


# ----------------------------------------------------------------------
# Dataset authority + loader
# ----------------------------------------------------------------------
@pytest.fixture(scope="module")
def authority() -> data.DatasetAuthority:
    if not SMOKE_MANIFEST_PATH.exists():
        pytest.skip(f"BRDT smoke manifest not present at {SMOKE_MANIFEST_PATH}")
    return data.load_dataset_authority(SMOKE_MANIFEST_PATH)


def test_authority_consumes_locked_geometry(authority: data.DatasetAuthority):
    assert authority.dataset_id == dc.DATASET_NAME
    assert int(authority.operator_block["grid_size"]) == dc.LOCKED_GRID_SIZE
    assert int(authority.operator_block["detector_size"]) == dc.LOCKED_DETECTOR_SIZE
    assert int(authority.operator_block["angle_count"]) == dc.LOCKED_ANGLE_COUNT
    np.testing.assert_allclose(authority.angles_rad, dc.locked_angles())
    assert authority.normalization.std > 0.0


def test_split_loader_shapes_and_collation(authority: data.DatasetAuthority):
    split = data.BRDTSmokeSplit(
        authority.split_paths["train"], normalization=authority.normalization
    )
    assert len(split) == int(authority.raw_manifest["split"]["counts"]["train"])
    sample = split[0]
    assert sample["q_true_physical"].shape == (1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE)
    assert sample["q_true_norm"].shape == sample["q_true_physical"].shape
    assert sample["sinogram"].shape == (
        dc.LOCKED_ANGLE_COUNT,
        dc.LOCKED_DETECTOR_SIZE,
        2,
    )
    batch = data.brdt_collate([split[0], split[1]])
    assert batch["q_true_physical"].shape == (
        2,
        1,
        dc.LOCKED_GRID_SIZE,
        dc.LOCKED_GRID_SIZE,
    )
    assert batch["sinogram"].shape == (
        2,
        dc.LOCKED_ANGLE_COUNT,
        dc.LOCKED_DETECTOR_SIZE,
        2,
    )
    assert batch["sample_seed"].shape == (2,)
    assert isinstance(batch["phantom_family"], list) and len(batch["phantom_family"]) == 2


def test_sinogram_to_channels_first_from_dataset_layout():
    sino = torch.zeros(
        2,
        dc.LOCKED_ANGLE_COUNT,
        dc.LOCKED_DETECTOR_SIZE,
        2,
    )
    converted = data.sinogram_to_channels_first(sino)
    assert converted.shape == (
        2,
        2,
        dc.LOCKED_ANGLE_COUNT,
        dc.LOCKED_DETECTOR_SIZE,
    )


# ----------------------------------------------------------------------
# Classical Born-init-image derivation
# ----------------------------------------------------------------------
def test_classical_backend_detection_falls_back_to_local_adjoint():
    info = classical.detect_classical_backend()
    assert info.name in {"odtbrain", "local_adjoint"}
    if info.name == "local_adjoint":
        assert info.claim_boundary == "feasibility_only"


def test_local_adjoint_init_shape_and_dtype():
    angles = torch.from_numpy(dc.locked_angles())
    op = BornRytovForward2D(
        grid_size=dc.LOCKED_GRID_SIZE,
        detector_size=dc.LOCKED_DETECTOR_SIZE,
        angles=angles,
        wavelength_px=dc.LOCKED_WAVELENGTH_PX,
        medium_ri=dc.LOCKED_MEDIUM_RI,
        mode="born",
        normalize="unitary_fft",
    )
    sino = torch.zeros(
        2,
        dc.LOCKED_ANGLE_COUNT,
        dc.LOCKED_DETECTOR_SIZE,
        2,
        dtype=torch.float32,
    )
    sino[0, 5, 64, 0] = 1.0  # one nonzero entry to force a non-trivial adjoint
    init = classical.derive_born_init_image(
        sino,
        operator=op,
        backend=classical.ClassicalBackendInfo(
            name="local_adjoint",
            reason="forced",
            claim_boundary="feasibility_only",
        ),
    )
    assert init.shape == (2, 1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE)
    assert init.dtype == torch.float32
    assert torch.isfinite(init).all()
    # Non-trivial response on the populated sample, zero on the empty one.
    assert init[0].abs().sum().item() > 0.0
    assert init[1].abs().sum().item() == 0.0


def test_odtbrain_backprop_uses_vacuum_wavelength_and_object_function(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_module = types.ModuleType("odtbrain")
    calls = []

    def fake_backpropagate_2d(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return np.full(
            (dc.LOCKED_DETECTOR_SIZE, dc.LOCKED_DETECTOR_SIZE),
            0.25 + 0.0j,
            dtype=np.complex64,
        )

    fake_module.backpropagate_2d = fake_backpropagate_2d
    monkeypatch.setitem(sys.modules, "odtbrain", fake_module)

    sinogram = np.zeros(
        (dc.LOCKED_ANGLE_COUNT, dc.LOCKED_DETECTOR_SIZE, 2), dtype=np.float32
    )
    out = classical._odtbrain_backprop(sinogram, angles=dc.locked_angles())

    assert calls, "expected ODTbrain backpropagate_2d to be invoked"
    kwargs = calls[0]["kwargs"]
    assert kwargs["res"] == pytest.approx(
        dc.LOCKED_WAVELENGTH_PX * dc.LOCKED_MEDIUM_RI
    )
    assert kwargs["nm"] == pytest.approx(dc.LOCKED_MEDIUM_RI)
    assert "save_memory" not in kwargs
    assert out.shape == (dc.LOCKED_DETECTOR_SIZE, dc.LOCKED_DETECTOR_SIZE)
    assert np.allclose(out, 0.25)


def test_relative_physics_l2_is_zero_for_identical_sinograms():
    pred = torch.ones(2, 3, 4, 2)
    obs = pred.clone()

    assert model_based_inverse.relative_physics_l2(pred, obs).item() == pytest.approx(0.0)


def test_total_variation_l1_penalizes_spatial_jumps():
    q = torch.zeros(1, 1, 8, 8)
    q[:, :, 4:, :] = 1.0

    assert model_based_inverse.total_variation_l1(q).item() > 0.0


def test_model_based_inverse_reduces_physics_residual_on_tiny_clean_case():
    angles = torch.linspace(0.0, 2.0 * torch.pi, 9, dtype=torch.float64)[:-1]
    op = BornRytovForward2D(
        grid_size=32,
        detector_size=32,
        angles=angles,
        wavelength_px=8.0,
        medium_ri=1.333,
        mode="born",
        normalize="unitary_fft",
    )
    q_true = torch.zeros(1, 1, 32, 32)
    q_true[:, :, 12:20, 14:22] = 0.02
    with torch.no_grad():
        y = op(q_true)
    q0 = torch.zeros_like(q_true)
    initial = model_based_inverse.relative_physics_l2(op(q0), y).item()

    q_hat, info = model_based_inverse.optimize_born_inverse_batch(
        sinogram_obs=y,
        operator=op,
        config=model_based_inverse.ModelBasedInverseConfig(
            steps=40,
            learning_rate=0.2,
            tv_weight=0.0,
            l2_weight=0.0,
            clamp_min=-0.05,
            clamp_max=0.05,
        ),
        initial_q=q0,
    )

    final = model_based_inverse.relative_physics_l2(op(q_hat), y).item()
    assert final < 0.25 * initial
    assert info["final_relative_physics_l2"] < info["initial_relative_physics_l2"]


# ----------------------------------------------------------------------
# Adapter construction + forward shape
# ----------------------------------------------------------------------
@pytest.mark.parametrize("architecture", ["unet", "hybrid_resnet"])
def test_adapter_forward_shape_local_bodies(architecture: str):
    adapter = models.build_neural_adapter(
        architecture=architecture,
        in_channels=1,
        out_channels=1,
        grid_size=dc.LOCKED_GRID_SIZE,
    )
    x = torch.zeros(2, 1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE, dtype=torch.float32)
    y = adapter(x)
    assert y.shape == (2, 1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE)
    assert adapter.info().parameter_count > 0


def test_adapter_rejects_classical_label():
    with pytest.raises(ValueError, match="not a neural adapter"):
        models.build_neural_adapter(
            architecture="classical_born_backprop",
            in_channels=1,
        )


def test_adapter_rejects_wrong_input_shape():
    adapter = models.build_neural_adapter(architecture="unet", in_channels=1)
    with pytest.raises(ValueError):
        adapter(torch.zeros(2, 1, 64, 64))


def test_ffno_adapter_forward_shape_and_distinct_architecture():
    """FFNO must build under BRDT real-channel semantics with its own identity.

    The new row's adapter has to produce ``(B, 1, 128, 128)`` real
    output and report ``architecture='ffno'`` (NOT ``fno_vanilla``).
    """
    adapter = models.build_neural_adapter(
        architecture="ffno",
        in_channels=1,
        out_channels=1,
        grid_size=dc.LOCKED_GRID_SIZE,
    )
    x = torch.zeros(2, 1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE, dtype=torch.float32)
    y = adapter(x)
    assert y.shape == (2, 1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE)
    info = adapter.info()
    assert info.architecture == "ffno"
    assert info.parameter_count > 0
    # Architecture metadata must surface the FFNO-specific kwargs so
    # provenance does not silently collapse to the FNO-vanilla schema.
    for key in ("hidden_channels", "fno_modes", "fno_blocks"):
        assert key in info.arch_kwargs


def test_ffno_adapter_has_no_post_bottleneck_cnn_refiners():
    """BRDT FFNO stays close to the FFNO stack plus a minimal output adapter."""
    adapter = models.build_neural_adapter(
        architecture="ffno",
        in_channels=1,
        out_channels=1,
        grid_size=dc.LOCKED_GRID_SIZE,
    )
    ffno_body = adapter.body

    assert not hasattr(ffno_body, "refiners")
    assert "cnn_blocks" not in adapter.info().arch_kwargs
    assert not any(".refiners." in name for name, _ in adapter.named_modules())


def test_ffno_adapter_rejects_post_bottleneck_cnn_refiner_override():
    with pytest.raises(ValueError, match="cnn_blocks"):
        models.build_neural_adapter(
            architecture="ffno",
            in_channels=1,
            out_channels=1,
            grid_size=dc.LOCKED_GRID_SIZE,
            arch_kwargs={"cnn_blocks": 2},
        )


@pytest.mark.parametrize("architecture", ["ffno", "hybrid_resnet"])
def test_sinogram_input_adapter_forward_shape(architecture: str):
    adapter = models.build_sinogram_input_adapter(architecture=architecture)
    x = torch.zeros(
        2,
        2,
        dc.LOCKED_ANGLE_COUNT,
        dc.LOCKED_DETECTOR_SIZE,
        dtype=torch.float32,
    )
    y = adapter(x)
    assert y.shape == (2, 1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE)
    info = adapter.info()
    assert info.in_channels == 2
    assert info.arch_kwargs["input_mode"] == "sinogram"
    assert info.arch_kwargs["sinogram_to_grid"] == "bilinear_resize"


def test_sinogram_input_adapter_rejects_image_shape():
    adapter = models.build_sinogram_input_adapter(architecture="ffno")
    with pytest.raises(ValueError, match="spatial shape"):
        adapter(torch.zeros(2, 2, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE))


def test_ffno_adapter_parameter_count_distinct_from_fno_vanilla():
    """FFNO and FNO-vanilla must have separate parameter counts.

    Skipped when ``neuralop`` is unavailable since the FNO-vanilla path
    raises ``AdapterBuildError`` in that case; the FFNO path uses
    ptycho_torch's local FFNO components and does not need ``neuralop``.
    """
    ffno_adapter = models.build_neural_adapter(
        architecture="ffno",
        in_channels=1,
        grid_size=dc.LOCKED_GRID_SIZE,
    )
    try:
        fno_adapter = models.build_neural_adapter(
            architecture="fno_vanilla",
            in_channels=1,
            grid_size=dc.LOCKED_GRID_SIZE,
        )
    except models.AdapterBuildError:
        pytest.skip("neuralop unavailable; cannot compare parameter counts")
    assert ffno_adapter.info().architecture != fno_adapter.info().architecture
    # The two adapters share neither architecture id nor parameter count.
    assert ffno_adapter.info().parameter_count != fno_adapter.info().parameter_count


def test_run_config_row_schema_accepts_ffno_distinct_from_fno_vanilla():
    """``ffno`` is a supported internal architecture distinct from ``fno_vanilla``."""
    assert "ffno" in run_config.SUPPORTED_ARCHITECTURES
    assert "fno_vanilla" in run_config.SUPPORTED_ARCHITECTURES
    # Row schema must accept ``ffno`` as a model id.
    row = run_config.RowConfig(
        row_id="ffno",
        model="ffno",
        training=run_config.DEFAULT_TRAINING_LABEL,
        input_mode="born_init_image",
        dataset_id=dc.DECISION_SUPPORT_DATASET_NAME,
        operator_version="op",
        paper_label="FFNO",
    )
    payload = row.to_dict()
    assert payload["model"] == "ffno"
    assert payload["paper_label"] == "FFNO"


def test_fno_vanilla_blocker_when_neuralop_missing():
    # Build path raises AdapterBuildError if neuralop is not present; if it
    # *is* present, building must succeed and the forward must run.
    try:
        adapter = models.build_neural_adapter(
            architecture="fno_vanilla",
            in_channels=1,
            grid_size=dc.LOCKED_GRID_SIZE,
        )
    except models.AdapterBuildError as exc:
        payload = exc.to_payload()
        assert payload["model"] == "fno_vanilla"
        assert payload["reason"] == "neuralop_unavailable"
        return
    x = torch.zeros(1, 1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE)
    y = adapter(x)
    assert y.shape[-2:] == (dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE)


# ----------------------------------------------------------------------
# Loss + unnormalize-before-physics guard
# ----------------------------------------------------------------------
def test_normalized_q_routing_guard_rejects_anything_but_physical_q():
    with pytest.raises(ValueError, match="physical q"):
        dc.reject_normalized_q_to_operator("normalized_q")
    # Sanity: explicit "physical_q" routing does not raise.
    dc.reject_normalized_q_to_operator("physical_q")


def test_training_module_loss_routes_terms_with_physical_unnormalize():
    angles = torch.from_numpy(dc.locked_angles())
    op = BornRytovForward2D(
        grid_size=dc.LOCKED_GRID_SIZE,
        detector_size=dc.LOCKED_DETECTOR_SIZE,
        angles=angles,
        wavelength_px=dc.LOCKED_WAVELENGTH_PX,
        medium_ri=dc.LOCKED_MEDIUM_RI,
        mode="born",
        normalize="unitary_fft",
    )
    norm = dc.NormalizationStats(mean=1e-3, std=6e-3, qmin=-3e-2, qmax=3e-2)
    adapter = models.build_neural_adapter(architecture="unet", in_channels=1)
    module = lightning_module.BRDTTrainingModule(
        model=adapter,
        operator=op,
        normalization=norm,
        weights=run_config.LossWeights(),
        output_space="normalized_q",
    )
    q_pred_norm = torch.zeros(1, 1, dc.LOCKED_GRID_SIZE, dc.LOCKED_GRID_SIZE)
    q_phys = module.to_physical_q(q_pred_norm)
    np.testing.assert_allclose(q_phys.detach().numpy(), np.full_like(q_phys.numpy(), 1e-3))

    q_true_phys = torch.zeros_like(q_pred_norm)
    q_true_norm = torch.zeros_like(q_pred_norm)
    sino = torch.zeros(1, dc.LOCKED_ANGLE_COUNT, dc.LOCKED_DETECTOR_SIZE, 2)
    total, breakdown = module.compute_loss(
        q_pred=q_pred_norm,
        q_true_norm=q_true_norm,
        q_true_physical=q_true_phys,
        sinogram_obs=sino,
    )
    assert torch.isfinite(total)
    payload = breakdown.to_dict()
    for key in ("image", "physics", "relative_physics", "tv", "positivity", "total"):
        assert key in payload
    contract = module.loss_contract()
    assert contract["training_label"] == run_config.DEFAULT_TRAINING_LABEL
    assert contract["operator_input_routing_guard"].endswith(
        "reject_normalized_q_to_operator"
    )


# ----------------------------------------------------------------------
# CLI argument validation + dry-run / row-status emission
# ----------------------------------------------------------------------
def test_train_argparser_rejects_unknown_architecture():
    parser = train._build_argparser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--architecture", "rytov_kludge", "--manifest", "x", "--output-root", "y"])


def test_train_argparser_accepts_bounded_roster():
    parser = train._build_argparser()
    for arch in (
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
        "sru_net",
    ):
        ns = parser.parse_args(
            ["--architecture", arch, "--manifest", "m", "--output-root", "o"]
        )
        assert ns.architecture == arch


def test_evaluate_dry_run_emits_adapter_contract(tmp_path: Path):
    if not SMOKE_MANIFEST_PATH.exists():
        pytest.skip("BRDT smoke manifest missing")
    output_root = tmp_path / "eval_dry"
    result = evaluate.run_evaluation(
        architecture="unet",
        manifest_path=SMOKE_MANIFEST_PATH,
        output_root=output_root,
        split="val",
        batch_size=2,
        device_choice="cpu",
        in_channels=1,
        hybrid_label="hybrid_resnet",
        dry_run=True,
    )
    assert result["row_status"] == "feasibility_only"
    contract_path = Path(result["adapter_contract_path"])
    payload = json.loads(contract_path.read_text())
    assert payload["schema_version"] == reporting.ADAPTER_CONTRACT_SCHEMA_VERSION
    assert payload["dataset_id"] == dc.DATASET_NAME
    row_ids = {r["row_id"] for r in payload["rows"]}
    assert row_ids == {
        "classical_born_backprop",
        "unet",
        "fno_vanilla",
        "hybrid_resnet",
    }
    assert payload["row_schema"]["supported_input_modes"] == [
        "born_init_image",
        "sinogram",
    ]
    assert "direct_sinogram" in payload["row_schema"]["rejected_input_modes"]


def test_evaluate_classical_emits_feasibility_row(tmp_path: Path):
    if not SMOKE_MANIFEST_PATH.exists():
        pytest.skip("BRDT smoke manifest missing")
    output_root = tmp_path / "eval_classical"
    result = evaluate.run_evaluation(
        architecture="classical_born_backprop",
        manifest_path=SMOKE_MANIFEST_PATH,
        output_root=output_root,
        split="val",
        batch_size=2,
        device_choice="cpu",
        in_channels=1,
        hybrid_label="hybrid_resnet",
        dry_run=False,
    )
    assert result["row_status"] == "feasibility_only"
    summary = json.loads((output_root / "eval_summary.json").read_text())
    assert summary["row_id"] == "classical_born_backprop"
    # Implementation review HIGH-1: successful classical evaluation
    # MUST emit the durable adapter_contract.json alongside the
    # per-run summary so the four-row preflight can consume one
    # shared row schema regardless of which entrypoint produced it.
    contract_path = output_root / "adapter_contract.json"
    assert contract_path.exists(), "successful classical eval must emit adapter_contract.json"
    payload = json.loads(contract_path.read_text())
    assert payload["schema_version"] == reporting.ADAPTER_CONTRACT_SCHEMA_VERSION
    selected = next(r for r in payload["rows"] if r["row_id"] == "classical_born_backprop")
    assert selected["row_status"] == "feasibility_only"
    assert "sanity_summary" in selected


def test_evaluate_neural_success_emits_adapter_contract(tmp_path: Path):
    if not SMOKE_MANIFEST_PATH.exists():
        pytest.skip("BRDT smoke manifest missing")
    output_root = tmp_path / "eval_neural"
    result = evaluate.run_evaluation(
        architecture="unet",
        manifest_path=SMOKE_MANIFEST_PATH,
        output_root=output_root,
        split="val",
        batch_size=2,
        device_choice="cpu",
        in_channels=1,
        hybrid_label="hybrid_resnet",
        dry_run=False,
    )
    assert result["row_status"] == "feasibility_only"
    contract_path = output_root / "adapter_contract.json"
    assert contract_path.exists(), "successful neural eval must emit adapter_contract.json"
    payload = json.loads(contract_path.read_text())
    selected = next(r for r in payload["rows"] if r["row_id"] == "unet")
    assert selected["row_status"] == "feasibility_only"
    assert selected["sanity_summary"]["row_id"] == "unet"
    # Loss-contract snapshot from the training module must include the
    # operator-input routing guard reference (centralized invariant).
    assert "operator_input_routing_guard" in payload["loss_contract"]


def test_train_classical_only_emits_adapter_contract(tmp_path: Path):
    if not SMOKE_MANIFEST_PATH.exists():
        pytest.skip("BRDT smoke manifest missing")
    output_root = tmp_path / "train_classical"
    result = train.run_training(
        architecture="classical_born_backprop",
        manifest_path=SMOKE_MANIFEST_PATH,
        output_root=output_root,
        epochs=1,
        batch_size=2,
        learning_rate=2e-4,
        device_choice="cpu",
        fast_dev_run=True,
        in_channels=1,
        hybrid_label="hybrid_resnet",
    )
    assert result["row_status"] == "feasibility_only"
    contract_path = output_root / "adapter_contract.json"
    assert contract_path.exists(), "successful classical-only train must emit adapter_contract.json"
    payload = json.loads(contract_path.read_text())
    selected = next(r for r in payload["rows"] if r["row_id"] == "classical_born_backprop")
    assert selected["row_status"] == "feasibility_only"


def test_evaluate_sru_net_surfaces_distinct_row(tmp_path: Path):
    """``--architecture sru_net`` must not silently resolve to ``hybrid_resnet``.

    Implementation review HIGH-2: with the previous fallback, requesting
    ``sru_net`` while ``hybrid_label`` defaulted to ``hybrid_resnet``
    silently produced the ``hybrid_resnet`` row. The roster now follows
    the architecture choice so the requested visible row identity is
    preserved end-to-end, while ``model`` stays as the internal
    ``hybrid_resnet`` adapter body.
    """
    if not SMOKE_MANIFEST_PATH.exists():
        pytest.skip("BRDT smoke manifest missing")
    output_root = tmp_path / "eval_sru_dry"
    result = evaluate.run_evaluation(
        architecture="sru_net",
        manifest_path=SMOKE_MANIFEST_PATH,
        output_root=output_root,
        split="val",
        batch_size=2,
        device_choice="cpu",
        in_channels=1,
        hybrid_label="hybrid_resnet",  # default — must NOT cause silent resolution
        dry_run=True,
    )
    assert result["summary"]["row_id"] == "sru_net"
    contract_path = Path(result["adapter_contract_path"])
    payload = json.loads(contract_path.read_text())
    row_ids = {r["row_id"] for r in payload["rows"]}
    assert "sru_net" in row_ids and "hybrid_resnet" not in row_ids
    sru_row = next(r for r in payload["rows"] if r["row_id"] == "sru_net")
    assert sru_row["model"] == "hybrid_resnet"
    assert sru_row["paper_label"] == "SRU-Net"


def test_resolve_hybrid_label_architecture_is_authoritative():
    """When ``architecture`` names a Hybrid-family row, it takes precedence.

    The implementation-review HIGH-2 defect was that ``architecture="sru_net"``
    silently fell back to the ``hybrid_resnet`` row when ``hybrid_label``
    defaulted. The resolver now forces the roster row_id to match the
    requested architecture so the visible row identity cannot drift.
    For non-Hybrid-family architectures the ``hybrid_label`` flag still
    controls how the Hybrid-family row is presented in the roster.
    """
    # Architecture override: the requested visible row wins.
    assert train._resolve_hybrid_label("sru_net", "hybrid_resnet") == "sru_net"
    assert train._resolve_hybrid_label("hybrid_resnet", "sru_net") == "hybrid_resnet"
    assert evaluate._resolve_hybrid_label("sru_net", "hybrid_resnet") == "sru_net"
    assert evaluate._resolve_hybrid_label("hybrid_resnet", "sru_net") == "hybrid_resnet"
    # Non-Hybrid architecture: hybrid_label controls roster presentation.
    assert train._resolve_hybrid_label("unet", "sru_net") == "sru_net"
    assert train._resolve_hybrid_label("unet", "hybrid_resnet") == "hybrid_resnet"
