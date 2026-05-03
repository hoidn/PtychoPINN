"""Adapter contract tests for the BRDT task-adapter scope.

Covers the bounded preflight surface added in
``2026-04-29-brdt-task-adapters``:

- dataset authority / loader / collator,
- row schema (model/training/input_mode/dataset_id/operator_version/row_status),
- rejection of direct-sinogram input for the first bounded contract,
- Born-init-image derivation (local adjoint backend),
- adapter construction + forward shape for U-Net, FNO vanilla, Hybrid family,
- unnormalize-before-physics guard via the dataset_contract helper,
- loss-term routing on a tiny synthetic batch,
- train/eval CLI argument validation and row-status emission paths.
"""

from __future__ import annotations

import json
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
    assert by_id["sru_net"].paper_label == "SRU-Net"
    assert by_id["sru_net"].model == "sru_net"


def test_row_rejects_direct_sinogram_input():
    with pytest.raises(ValueError, match="Direct sinogram"):
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


def test_assert_input_mode_supported_rejects_sinogram():
    with pytest.raises(ValueError, match="rejects direct-sinogram"):
        data.assert_input_mode_supported("sinogram")


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
    assert payload["row_schema"]["supported_input_modes"] == ["born_init_image"]
    assert "sinogram" in payload["row_schema"]["rejected_input_modes"]


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
