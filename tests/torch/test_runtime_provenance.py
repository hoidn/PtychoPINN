"""Focused contracts for the shared Lightning runtime record."""

from types import SimpleNamespace

import torch


def test_strategy_runtime_records_backend_start_method_and_launcher():
    from ptycho_torch.runtime_provenance import strategy_runtime

    strategy = SimpleNamespace(
        root_device=torch.device("cpu"),
        parallel_devices=[torch.device("cpu"), torch.device("cpu")],
        _process_group_backend="gloo",
        _start_method="spawn",
        launcher=SimpleNamespace(),
    )

    runtime = strategy_runtime(strategy)

    assert runtime["process_group_backend"] == "gloo"
    assert runtime["start_method"] == "spawn"
    assert runtime["root_device"] == "cpu"
    assert runtime["launcher"]["class"] == "types.SimpleNamespace"
