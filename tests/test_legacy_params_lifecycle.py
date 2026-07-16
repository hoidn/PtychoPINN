"""Contract tests for process-local legacy params lifecycle containment."""

import asyncio
import dill
import multiprocessing
from pathlib import Path
import threading
from types import SimpleNamespace
import zipfile
from unittest.mock import MagicMock

import pytest

from ptycho import params


def _enter_scope_in_forked_child(result_queue):
    from ptycho.config.legacy_state import legacy_params_scope

    async def enter():
        with legacy_params_scope():
            return "entered"

    try:
        result_queue.put(asyncio.run(enter()))
    except Exception as error:  # pragma: no cover - parent asserts the payload
        result_queue.put(f"{type(error).__name__}: {error}")


def test_nested_scopes_restore_parent_without_rebinding_cfg():
    from ptycho.config.legacy_state import legacy_params_scope

    cfg_object = params.cfg
    original = dict(params.cfg)
    try:
        params.cfg["scope_marker"] = "root"

        with legacy_params_scope():
            params.cfg["scope_marker"] = "outer"
            params.cfg["outer_only"] = True

            with legacy_params_scope():
                params.cfg["scope_marker"] = "inner"
                params.cfg["inner_only"] = True

            assert params.cfg["scope_marker"] == "outer"
            assert params.cfg["outer_only"] is True
            assert "inner_only" not in params.cfg
            assert params.cfg is cfg_object

        assert params.cfg["scope_marker"] == "root"
        assert "outer_only" not in params.cfg
        assert "inner_only" not in params.cfg
        assert params.cfg is cfg_object
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_scope_restores_state_after_exception():
    from ptycho.config.legacy_state import legacy_params_scope

    cfg_object = params.cfg
    original = dict(params.cfg)
    try:
        params.cfg["scope_marker"] = "root"

        with pytest.raises(RuntimeError, match="boom"):
            with legacy_params_scope():
                params.cfg["scope_marker"] = "temporary"
                params.cfg["temporary_only"] = True
                raise RuntimeError("boom")

        assert params.cfg["scope_marker"] == "root"
        assert "temporary_only" not in params.cfg
        assert params.cfg is cfg_object
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_scope_restores_parent_seal_state():
    from ptycho.config.legacy_state import legacy_params_scope

    original_sealed = params._sealed
    try:
        params.seal()

        with legacy_params_scope():
            params.unseal()
            assert params._sealed is False

        assert params._sealed is True
    finally:
        if original_sealed:
            params.seal()
        else:
            params.unseal()


def test_overlapping_threads_fail_fast():
    from ptycho.config.legacy_state import legacy_params_scope

    outcomes = []

    def enter_scope():
        try:
            with legacy_params_scope():
                outcomes.append("entered")
        except RuntimeError as error:
            outcomes.append(str(error))

    with legacy_params_scope():
        worker = threading.Thread(target=enter_scope)
        worker.start()
        worker.join(timeout=2)

    assert not worker.is_alive()
    assert len(outcomes) == 1
    assert "already active" in outcomes[0]


def test_overlapping_asyncio_tasks_fail_fast():
    from ptycho.config.legacy_state import legacy_params_scope

    async def exercise_overlap():
        entered = asyncio.Event()
        release = asyncio.Event()
        outcomes = []

        async def owner():
            with legacy_params_scope():
                entered.set()
                await release.wait()

        async def contender():
            await entered.wait()
            try:
                with legacy_params_scope():
                    outcomes.append("entered")
            except RuntimeError as error:
                outcomes.append(str(error))
            finally:
                release.set()

        await asyncio.gather(owner(), contender())
        return outcomes

    outcomes = asyncio.run(exercise_overlap())

    assert len(outcomes) == 1
    assert "already active" in outcomes[0]


def test_forked_worker_configures_independently_of_parent_scope():
    from ptycho.config.legacy_state import legacy_params_scope

    context = multiprocessing.get_context("fork")
    result_queue = context.Queue()

    with legacy_params_scope():
        worker = context.Process(
            target=_enter_scope_in_forked_child,
            args=(result_queue,),
        )
        worker.start()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert worker.exitcode == 0
    assert result_queue.get(timeout=1) == "entered"


def test_archived_state_survives_all_surrounding_scopes():
    from ptycho.config.legacy_state import (
        archived_params_scope,
        legacy_params_scope,
    )

    cfg_object = params.cfg
    original = dict(params.cfg)
    try:
        params.cfg["archive_marker"] = "root"

        with legacy_params_scope():
            with legacy_params_scope():
                with archived_params_scope(
                    {"archive_marker": "loaded", "archive_only": True}
                ):
                    assert params.cfg["archive_marker"] == "loaded"

            assert params.cfg["archive_marker"] == "loaded"
            assert params.cfg["archive_only"] is True

        assert params.cfg["archive_marker"] == "loaded"
        assert params.cfg["archive_only"] is True
        assert params.cfg is cfg_object
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_archived_state_rolls_back_when_enclosing_load_fails():
    from ptycho.config.legacy_state import archived_params_scope

    original = dict(params.cfg)
    try:
        params.cfg["archive_marker"] = "root"

        with pytest.raises(RuntimeError, match="outer load failed"):
            with archived_params_scope({"archive_marker": "outer"}):
                with archived_params_scope({"archive_marker": "inner"}):
                    assert params.cfg["archive_marker"] == "inner"
                raise RuntimeError("outer load failed")

        assert params.cfg["archive_marker"] == "root"
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_scoped_entrypoint_restores_temporary_global_state():
    from ptycho.config.legacy_state import scoped_legacy_params

    original = dict(params.cfg)
    try:
        params.cfg["entrypoint_marker"] = "root"

        @scoped_legacy_params
        def temporary_entrypoint():
            params.cfg["entrypoint_marker"] = "temporary"
            return params.cfg["entrypoint_marker"]

        assert temporary_entrypoint() == "temporary"
        assert params.cfg["entrypoint_marker"] == "root"
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_backend_workflow_entrypoint_contains_legacy_bridge(monkeypatch):
    from ptycho.workflows import backend_selector
    from ptycho_torch.workflows import components as torch_components

    original = dict(params.cfg)
    try:
        params.cfg["workflow_marker"] = "root"

        def bridge(cfg, _config):
            cfg["workflow_marker"] = "temporary"

        monkeypatch.setattr(backend_selector, "update_legacy_dict", bridge)
        monkeypatch.setattr(
            torch_components,
            "run_cdi_example_torch",
            lambda *_args, **_kwargs: (None, None, {}),
        )

        backend_selector.run_cdi_example_with_backend(
            None,
            None,
            SimpleNamespace(backend="pytorch"),
            torch_execution_config=object(),
        )

        assert params.cfg["workflow_marker"] == "root"
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_tensorflow_bundle_failure_rolls_back_archived_state(tmp_path):
    from ptycho.model_manager import ModelManager

    model_dir = tmp_path / "diffraction_to_obj"
    model_dir.mkdir()
    with (model_dir / "params.dill").open("wb") as handle:
        dill.dump({"N": 64, "gridsize": 1, "loaded_only": True}, handle)

    original = dict(params.cfg)
    try:
        params.cfg["archive_marker"] = "root"

        with pytest.raises(FileNotFoundError):
            ModelManager.load_model(str(model_dir))

        assert params.cfg["archive_marker"] == "root"
        assert "loaded_only" not in params.cfg
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_torch_bundle_failure_rolls_back_archived_state(tmp_path, monkeypatch):
    from ptycho_torch import model_manager

    base_path = tmp_path / "wts.h5"
    archive = Path(f"{base_path}.zip")
    source = tmp_path / "archive"
    source.mkdir()
    model_names = ["autoencoder", "diffraction_to_obj"]
    with (source / "manifest.dill").open("wb") as handle:
        dill.dump({"models": model_names, "version": "2.0-pytorch"}, handle)
    for model_name in model_names:
        model_dir = source / model_name
        model_dir.mkdir(parents=True)
        with (model_dir / "params.dill").open("wb") as handle:
            dill.dump(
                {
                    "_version": "2.0-pytorch",
                    "N": 64,
                    "gridsize": 1,
                    "loaded_only": True,
                },
                handle,
            )
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.write(source / "manifest.dill", "manifest.dill")
        for model_name in model_names:
            bundle.write(
                source / model_name / "params.dill",
                f"{model_name}/params.dill",
            )

    monkeypatch.setattr(
        model_manager,
        "create_torch_model_with_gridsize",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("build failed")),
    )

    original = dict(params.cfg)
    try:
        params.cfg["archive_marker"] = "root"

        with pytest.raises(RuntimeError, match="build failed"):
            model_manager.load_torch_bundle(str(base_path))

        assert params.cfg["archive_marker"] == "root"
        assert "loaded_only" not in params.cfg
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_outer_bundle_route_rolls_back_inner_success(monkeypatch, tmp_path):
    from ptycho.workflows import backend_selector
    from ptycho_torch.workflows import components as torch_components

    def successful_inner_load(*_args, **_kwargs):
        params.cfg.update({"archive_marker": "loaded", "loaded_only": True})
        return {"autoencoder": object()}, {"N": 64, "gridsize": 1}

    monkeypatch.setattr(
        torch_components,
        "load_inference_bundle_torch",
        successful_inner_load,
    )

    original = dict(params.cfg)
    try:
        params.cfg["archive_marker"] = "root"

        with pytest.raises(KeyError, match="diffraction_to_obj"):
            backend_selector.load_inference_bundle_with_backend(
                tmp_path,
                SimpleNamespace(backend="pytorch"),
            )

        assert params.cfg["archive_marker"] == "root"
        assert "loaded_only" not in params.cfg
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_inference_adapter_restores_inferred_gridsize_after_call():
    import numpy as np

    from ptycho.workflows.components import DiffractionToObjectAdapter

    base_model = MagicMock()
    base_model.name = "base"

    def observe_gridsize(*_args, **_kwargs):
        assert params.cfg["gridsize"] == 2
        return "result"

    base_model.side_effect = observe_gridsize
    adapter = DiffractionToObjectAdapter(base_model)

    original = dict(params.cfg)
    try:
        params.cfg["gridsize"] = 1
        result = adapter.call(np.zeros((1, 8, 8, 4), dtype=np.float32))

        assert result == "result"
        assert params.cfg["gridsize"] == 1
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_configuration_transaction_persists_standalone_but_not_past_outer_scope():
    from ptycho.config.legacy_state import (
        configured_legacy_params,
        legacy_params_scope,
    )

    @configured_legacy_params
    def configure(value):
        params.cfg["configuration_marker"] = value

    original = dict(params.cfg)
    try:
        params.cfg["configuration_marker"] = "root"

        configure("standalone")
        assert params.cfg["configuration_marker"] == "standalone"

        with legacy_params_scope():
            configure("nested")
            assert params.cfg["configuration_marker"] == "nested"

        assert params.cfg["configuration_marker"] == "standalone"
    finally:
        params.cfg.clear()
        params.cfg.update(original)


def test_nested_entrypoint_reuses_parent_lifetime():
    from ptycho.config.legacy_state import (
        legacy_params_scope,
        scoped_legacy_params,
    )

    @scoped_legacy_params
    def compute_runtime_state():
        params.cfg["computed_marker"] = "available"

    original = dict(params.cfg)
    try:
        with legacy_params_scope():
            compute_runtime_state()
            assert params.cfg["computed_marker"] == "available"

        assert "computed_marker" not in params.cfg
    finally:
        params.cfg.clear()
        params.cfg.update(original)
