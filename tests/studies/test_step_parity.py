"""Tests for the build/step parity diagnostic helpers.

The parity diagnostic measures how the two training flows BUILD the model
(state_dict + config + RNG accounting) and how they STEP it (one
forward+loss+backward with identical weights and batch). Helpers are
generic over torch modules so they are testable on small stand-ins.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

torch = pytest.importorskip("torch")


def _pair() -> tuple["torch.nn.Module", "torch.nn.Module"]:
    torch.manual_seed(0)
    a = torch.nn.Linear(4, 3)
    torch.manual_seed(0)
    b = torch.nn.Linear(4, 3)
    return a, b


def test_compare_state_dicts_reports_equality_and_first_divergence() -> None:
    from scripts.studies.ablation.runtime_ladder_step_parity import (
        compare_state_dicts,
        state_dict_table,
    )

    a, b = _pair()
    table = state_dict_table(a)
    assert set(table) == {"weight", "bias"}
    assert table["weight"]["shape"] == (3, 4)
    assert table["weight"]["requires_grad"] is True

    report = compare_state_dicts(a, b)
    assert report["equal"] is True
    assert report["differing"] == []
    assert report["only_in_a"] == [] and report["only_in_b"] == []

    with torch.no_grad():
        b.bias[1] += 1e-3
    report = compare_state_dicts(a, b)
    assert report["equal"] is False
    (entry,) = report["differing"]
    assert entry["name"] == "bias"
    assert entry["max_abs_diff"] == pytest.approx(1e-3, rel=1e-4)
    assert report["first_divergent"] == "bias"


def test_compare_state_dicts_reports_param_set_mismatch() -> None:
    from scripts.studies.ablation.runtime_ladder_step_parity import (
        compare_state_dicts,
    )

    a, _ = _pair()
    b = torch.nn.Sequential(torch.nn.Linear(4, 3))
    report = compare_state_dicts(a, b)
    assert report["equal"] is False
    assert report["only_in_a"] == ["bias", "weight"]
    assert report["only_in_b"] == ["0.bias", "0.weight"]


def test_hparams_config_delta_diffs_dataclass_fields() -> None:
    from scripts.studies.ablation.runtime_ladder_step_parity import (
        hparams_config_delta,
    )

    @dataclass(frozen=True)
    class Cfg:
        n: int = 1
        mode: str = "amp"

    class Holder:
        def __init__(self, cfg: Cfg) -> None:
            self.hparams = {"model_config": cfg}

    delta = hparams_config_delta(Holder(Cfg()), Holder(Cfg(mode="rect")))
    assert delta == {"model_config": {"mode": ["amp", "rect"]}}
    assert hparams_config_delta(Holder(Cfg()), Holder(Cfg())) == {}


def test_record_rng_draws_counts_and_sites() -> None:
    from scripts.studies.ablation.runtime_ladder_step_parity import (
        record_rng_draws,
    )

    with record_rng_draws() as draws:
        torch.manual_seed(1)
        torch.rand(3)
        t = torch.empty(5)
        t.normal_()
    fns = [d["fn"] for d in draws]
    assert "rand" in fns and "normal_" in fns
    site = next(d for d in draws if d["fn"] == "rand")["site"]
    assert "test_step_parity.py" in site
    # Instrumentation must not perturb RNG determinism.
    with record_rng_draws():
        torch.manual_seed(1)
        inside = torch.rand(3)
    torch.manual_seed(1)
    outside = torch.rand(3)
    assert torch.equal(inside, outside)


def test_one_loss_and_grads_is_deterministic_and_comparable() -> None:
    from scripts.studies.ablation.runtime_ladder_step_parity import (
        compare_grads,
        one_loss_and_grads,
    )

    class Toy(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lin = torch.nn.Linear(4, 1)

        def compute_loss(self, batch):
            x, y = batch
            return torch.nn.functional.mse_loss(self.lin(x), y)

    torch.manual_seed(2)
    batch = (torch.rand(8, 4), torch.rand(8, 1))
    a, b = Toy(), Toy()
    b.load_state_dict(a.state_dict())

    ra = one_loss_and_grads(a, batch, seed=3)
    rb = one_loss_and_grads(b, batch, seed=3)
    assert ra["loss"] == rb["loss"]
    assert set(ra["grads"]) == {"lin.weight", "lin.bias"}
    cmp = compare_grads(ra, rb)
    assert cmp["identical"] is True
    assert cmp["max_rel_diff"] == 0.0

    # Perturbed weights -> differing grads, reported per parameter.
    with torch.no_grad():
        b.lin.weight += 0.5
    rb2 = one_loss_and_grads(b, batch, seed=3)
    cmp2 = compare_grads(ra, rb2)
    assert cmp2["identical"] is False
    assert cmp2["loss_delta"] == pytest.approx(
        abs(ra["loss"] - rb2["loss"]), rel=1e-6
    )
    assert cmp2["max_rel_diff"] > 0
    assert cmp2["first_divergent"] in {"lin.weight", "lin.bias"}


def test_capture_trainer_records_model() -> None:
    """Build parity needs the exact module each flow hands to Trainer.fit."""
    from scripts.studies.ablation.runtime_ladder_capture import (
        _CaptureTrainer,
        _LoaderCaptureRequested,
    )

    _CaptureTrainer.holder.clear()
    trainer = _CaptureTrainer(max_epochs=1)
    sentinel_model = object()
    with pytest.raises(_LoaderCaptureRequested):
        trainer.fit(sentinel_model, train_dataloaders=[1], val_dataloaders=[2])
    assert _CaptureTrainer.holder["model"] is sentinel_model
