"""Tests for the 2x2 checkpoint x evaluation-path cross-eval matrix.

The matrix diagnoses whether the mmap rungs' residual amp-SSIM regression is
post-training: sealed checkpoints are swapped across the two evaluation
paths (rung0's dictionary-container path vs the mmap rungs' generic-container
path), holding the historical grid-lines stitcher and the
eval_reconstruction metric convention fixed throughout.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.studies.ablation.runtime_errors import (
    RuntimeExecutionError,
    StudyRequestError,
)

def test_load_sealed_checkpoint_refuses_hash_mismatch(tmp_path: Path) -> None:
    """Identity check must fail closed BEFORE any model deserialization."""
    from scripts.studies.ablation.runtime_ladder_cross_eval import (
        load_sealed_checkpoint,
    )

    fake = tmp_path / "fake.ckpt"
    fake.write_bytes(b"not a checkpoint")
    with pytest.raises(StudyRequestError, match="sha256"):
        load_sealed_checkpoint(fake, expected_sha256="0" * 64, device="cpu")


def test_evaluate_cell_wires_historical_stitch_and_declared_metric(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cell evaluator must run the EXACT sealed-path chain by import:
    run_torch_inference -> complex patches ->
    _reassemble_predictions_for_metrics(reassembly_mode='grid_lines') ->
    ptycho.evaluation.eval_reconstruction on the historical canvas."""
    from ptycho import evaluation
    from scripts.studies import grid_lines_torch_runner as runner_mod
    from scripts.studies.ablation.runtime_ladder_cross_eval import evaluate_cell

    ground_truth = np.ones((6, 6), dtype=np.complex64)
    test_data = {"YY_ground_truth": ground_truth}
    patches = np.zeros((4, 3, 3, 2), dtype=np.float32)
    calls: dict[str, Any] = {}

    def fake_inference(model, data, cfg, metadata=None):
        calls["inference"] = {"model": model, "metadata": metadata}
        return patches

    def fake_reassemble(pred, gt, data, metadata, cfg):
        calls["reassembly_mode"] = cfg.reassembly_mode
        assert np.iscomplexobj(pred)
        return ground_truth.copy(), None, {}

    def fake_eval(canvas, gt, label, *, trim_offset):
        calls["metric_shapes"] = (canvas.shape, gt.shape)
        calls["metric_label"] = label
        calls["metric_trim_offset"] = trim_offset
        return {"mae": (0.1, 0.2), "ssim": (0.9, 0.95)}

    monkeypatch.setattr(runner_mod, "run_torch_inference", fake_inference)
    monkeypatch.setattr(
        runner_mod, "_reassemble_predictions_for_metrics", fake_reassemble
    )
    monkeypatch.setattr(evaluation, "eval_reconstruction_explicit", fake_eval)

    cfg = replace(
        runner_mod.TorchRunnerConfig(
            train_npz=Path("train.npz"),
            test_npz=Path("test.npz"),
            output_dir=Path("out"),
            architecture="hybrid_resnet",
        ),
        reassembly_mode="position",  # cell must OVERRIDE to grid_lines
    )
    payload = evaluate_cell(
        model=object(), runner_cfg=cfg, test_data=test_data,
        test_metadata={"m": 1}, label="cell_c", trim_offset=4,
    )

    assert calls["reassembly_mode"] == "grid_lines"
    assert calls["metric_shapes"] == ((6, 6, 1), (6, 6, 1))
    assert calls["metric_label"] == "cell_c"
    assert calls["metric_trim_offset"] == 4
    assert calls["inference"]["metadata"] == {"m": 1}
    assert payload["amp_ssim"] == 0.9
    assert payload["phase_ssim"] == 0.95
    assert payload["amp_mae"] == 0.1
    assert payload["phase_mae"] == 0.2
    assert isinstance(payload["historical_canvas_sha256"], str)
    assert isinstance(payload["pre_stitch_patch_sha256"], str)
    assert payload["canvas"].shape == ground_truth.shape


def test_evaluate_cell_refuses_resized_canvas(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from scripts.studies import grid_lines_torch_runner as runner_mod
    from scripts.studies.ablation.runtime_ladder_cross_eval import evaluate_cell

    ground_truth = np.ones((6, 6), dtype=np.complex64)
    monkeypatch.setattr(
        runner_mod, "run_torch_inference",
        lambda *a, **k: np.zeros((4, 3, 3, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        runner_mod, "_reassemble_predictions_for_metrics",
        lambda *a, **k: (np.ones((5, 5), dtype=np.complex64), None, {}),
    )
    cfg = runner_mod.TorchRunnerConfig(
        train_npz=Path("train.npz"), test_npz=Path("test.npz"),
        output_dir=Path("out"), architecture="hybrid_resnet",
    )
    with pytest.raises(RuntimeExecutionError, match="resiz"):
        evaluate_cell(
            model=object(), runner_cfg=cfg,
            test_data={"YY_ground_truth": ground_truth},
            test_metadata=None, label="cell", trim_offset=4,
        )


def test_load_sealed_checkpoint_refuses_hparams_absent_checkpoint(
    tmp_path: Path,
) -> None:
    """C-2: a checkpoint without persisted hyper_parameters cannot be
    restored configlessly — that must surface as a typed fail-closed error,
    not an uncaught TypeError from Lightning's constructor."""
    import hashlib

    torch = pytest.importorskip("torch")
    from scripts.studies.ablation.runtime_ladder_cross_eval import (
        load_sealed_checkpoint,
    )

    bare = tmp_path / "bare.ckpt"
    torch.save({"state_dict": {}, "pytorch-lightning_version": "2.0"}, bare)
    digest = hashlib.sha256(bare.read_bytes()).hexdigest()
    with pytest.raises(StudyRequestError, match="hyper_parameters"):
        load_sealed_checkpoint(bare, expected_sha256=digest, device="cpu")


def test_runner_config_delta_excludes_path_operands() -> None:
    """C-1: the eval contexts' runner-config delta is recorded in the matrix
    artifact; path-valued operands are excluded (always differ, never
    metric-relevant)."""
    from scripts.studies import grid_lines_torch_runner as runner_mod
    from scripts.studies.ablation.runtime_ladder_cross_eval import (
        runner_config_delta,
    )

    base = dict(
        train_npz=Path("a/train.npz"), test_npz=Path("a/test.npz"),
        output_dir=Path("a"), architecture="hybrid_resnet",
    )
    cfg_a = runner_mod.TorchRunnerConfig(**base)
    cfg_b = runner_mod.TorchRunnerConfig(
        **{**base, "train_npz": Path("b/train.npz"), "output_dir": Path("b")},
        scale_contract_version="legacy_v1",
        measurement_domain="normalized_amplitude",
    )
    delta = runner_config_delta(cfg_a, cfg_b)
    assert delta == {
        "scale_contract_version": ["ci_intensity_v2", "legacy_v1"],
        "measurement_domain": ["count_intensity", "normalized_amplitude"],
    }


def test_matrix_assembles_four_cells_with_provenance(tmp_path: Path) -> None:
    """The orchestrator emits a machine-readable matrix: four cells keyed
    A-D, each recording checkpoint identity, evaluation path, metrics, and
    the sealed expectation delta where one exists; convention provenance is
    recorded once at the top level. Canvases persist per cell."""
    from scripts.studies.ablation.runtime_ladder_cross_eval import (
        CellRequest,
        assemble_matrix,
    )

    canvas = np.ones((4, 4), dtype=np.complex64)

    def stub_runner(request: CellRequest) -> dict[str, Any]:
        return {
            "amp_ssim": 0.5, "phase_ssim": 0.8, "amp_mae": 0.2,
            "phase_mae": 0.1, "historical_canvas_sha256": "c" * 64,
            "pre_stitch_patch_sha256": "p" * 64, "canvas": canvas,
        }

    requests = [
        CellRequest(
            cell_id=cell_id,
            checkpoint_id=ckpt,
            checkpoint_path=Path(f"{ckpt}.ckpt"),
            checkpoint_sha256="a" * 64,
            eval_path=path,
            sealed_expected={"amp_ssim": 0.5} if cell_id in ("A", "B") else None,
        )
        for cell_id, ckpt, path in (
            ("A", "rung0_reference", "reference_dictionary"),
            ("B", "rung1e_sampler_plus_unit_norm", "mmap_generic"),
            ("C", "rung1e_sampler_plus_unit_norm", "reference_dictionary"),
            ("D", "rung0_reference", "mmap_generic"),
        )
    ]
    matrix = assemble_matrix(
        requests, stub_runner, output_dir=tmp_path,
        eval_context_delta={"scale_contract_version": ["a", "b"]},
    )

    assert sorted(matrix["cells"]) == ["A", "B", "C", "D"]
    assert matrix["eval_context_runner_delta"] == {
        "scale_contract_version": ["a", "b"]
    }
    cell_a = matrix["cells"]["A"]
    assert cell_a["checkpoint_id"] == "rung0_reference"
    assert cell_a["eval_path"] == "reference_dictionary"
    assert cell_a["amp_ssim"] == 0.5
    assert cell_a["sealed_expected"]["amp_ssim"] == 0.5
    assert cell_a["sealed_delta"]["amp_ssim"] == 0.0
    assert matrix["cells"]["C"]["sealed_expected"] is None
    # Convention provenance recorded once, top-level.
    assert matrix["stitcher"] == (
        "grid_lines_torch_runner._reassemble_predictions_for_metrics"
        "(reassembly_mode='grid_lines')"
    )
    assert matrix["metric"] == "ptycho.evaluation.eval_reconstruction"
    assert "gauge_handling" in matrix
    # JSON + per-cell canvases persisted; canvases never inline in JSON.
    written = json.loads((tmp_path / "cross_eval_matrix.json").read_text())
    assert sorted(written["cells"]) == ["A", "B", "C", "D"]
    assert "canvas" not in written["cells"]["A"]
    for cell_id in ("A", "B", "C", "D"):
        stored = np.load(tmp_path / cell_id / "canvas.npy")
        np.testing.assert_array_equal(stored, canvas)
