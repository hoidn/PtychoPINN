"""Build/step parity diagnostic helpers (model-construction seam).

The cross-eval matrix proved the mmap-trained MODEL itself differs from the
dictionary-trained reference under identical evaluation. These helpers
measure the remaining seam — how the two flows BUILD and STEP the model:

- ``state_dict_table`` / ``compare_state_dicts``: tensor-by-tensor identity
  (sha, shape, dtype, requires_grad) with first-divergence reporting.
- ``hparams_config_delta``: field-by-field diff of the config dataclasses a
  Lightning module persisted in ``hparams`` — the resolved build inputs.
- ``record_rng_draws``: counts torch RNG draws WITH call sites, so an init
  divergence comes with a mechanism (which site consumed the stream), not
  just a hash mismatch. Wrappers delegate to the original ops; determinism
  is untouched.
- ``one_loss_and_grads`` / ``compare_grads``: one fixed-RNG forward+loss+
  backward on a shared batch and identical weights; loss scalar and
  per-parameter gradient identity isolate objective differences.

All helpers are generic over torch modules (testable on stand-ins); the
driver applies them to the exact modules captured from the real flows via
``runtime_ladder_capture`` (import-linked, nothing reimplemented).
"""

from __future__ import annotations

import hashlib
import traceback
from contextlib import contextmanager
from dataclasses import fields, is_dataclass
from typing import Any, Iterator, Mapping

import numpy as np

__all__ = [
    "state_dict_table",
    "compare_state_dicts",
    "hparams_config_delta",
    "record_rng_draws",
    "one_loss_and_grads",
    "compare_grads",
]

_HPARAM_CONFIG_KEYS = (
    "model_config",
    "data_config",
    "training_config",
    "inference_config",
)


def _tensor_sha(tensor: Any) -> str:
    array = np.ascontiguousarray(tensor.detach().cpu().numpy())
    return hashlib.sha256(array.tobytes()).hexdigest()


def state_dict_table(module: Any) -> dict[str, dict[str, Any]]:
    """Per-tensor identity of a module's state_dict (params AND buffers)."""
    requires = {name: p.requires_grad for name, p in module.named_parameters()}
    table: dict[str, dict[str, Any]] = {}
    for name, tensor in module.state_dict().items():
        table[name] = {
            "sha256": _tensor_sha(tensor),
            "shape": tuple(tensor.shape),
            "dtype": str(tensor.dtype),
            "requires_grad": requires.get(name),
        }
    return table


def compare_state_dicts(module_a: Any, module_b: Any) -> dict[str, Any]:
    """Tensor-by-tensor comparison; first divergence in state_dict order."""
    table_a = state_dict_table(module_a)
    table_b = state_dict_table(module_b)
    state_a = module_a.state_dict()
    state_b = module_b.state_dict()
    only_in_a = sorted(set(table_a) - set(table_b))
    only_in_b = sorted(set(table_b) - set(table_a))
    differing: list[dict[str, Any]] = []
    requires_grad_delta: list[dict[str, Any]] = []
    first_divergent: str | None = None
    for name in state_a:
        if name not in table_b:
            continue
        entry_a, entry_b = table_a[name], table_b[name]
        if entry_a["requires_grad"] != entry_b["requires_grad"]:
            requires_grad_delta.append(
                {
                    "name": name,
                    "requires_grad": [
                        entry_a["requires_grad"],
                        entry_b["requires_grad"],
                    ],
                }
            )
        if entry_a["sha256"] == entry_b["sha256"]:
            continue
        a = state_a[name].detach().cpu().to(dtype=None)
        b = state_b[name].detach().cpu()
        max_abs = (
            float((a.float() - b.float()).abs().max())
            if a.shape == b.shape
            else None
        )
        differing.append(
            {
                "name": name,
                "sha256": [entry_a["sha256"], entry_b["sha256"]],
                "shape": [entry_a["shape"], entry_b["shape"]],
                "max_abs_diff": max_abs,
            }
        )
        if first_divergent is None:
            first_divergent = name
    equal = not (only_in_a or only_in_b or differing)
    return {
        "equal": equal,
        "only_in_a": only_in_a,
        "only_in_b": only_in_b,
        "differing": differing,
        "requires_grad_delta": requires_grad_delta,
        "first_divergent": first_divergent,
        "compared": sum(1 for name in table_a if name in table_b),
    }


def _config_mapping(holder: Any) -> Mapping[str, Any]:
    hparams = getattr(holder, "hparams", {})
    if hasattr(hparams, "items"):
        return dict(hparams.items())
    return dict(hparams)


def hparams_config_delta(model_a: Any, model_b: Any) -> dict[str, Any]:
    """Field-by-field diff of the persisted config dataclasses."""
    configs_a = _config_mapping(model_a)
    configs_b = _config_mapping(model_b)
    delta: dict[str, Any] = {}
    for key in _HPARAM_CONFIG_KEYS:
        cfg_a, cfg_b = configs_a.get(key), configs_b.get(key)
        if cfg_a is None and cfg_b is None:
            continue
        if not (is_dataclass(cfg_a) and is_dataclass(cfg_b)):
            if cfg_a != cfg_b:
                delta[key] = {"__object__": [repr(cfg_a), repr(cfg_b)]}
            continue
        field_delta: dict[str, list[Any]] = {}
        names = {f.name for f in fields(cfg_a)} | {f.name for f in fields(cfg_b)}
        for name in sorted(names):
            value_a = getattr(cfg_a, name, "<absent>")
            value_b = getattr(cfg_b, name, "<absent>")
            if _values_differ(value_a, value_b):
                field_delta[name] = [value_a, value_b]
        if field_delta:
            delta[key] = field_delta
    return delta


def _values_differ(a: Any, b: Any) -> bool:
    try:
        import torch

        if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
            if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
                return True
            return a.shape != b.shape or not torch.equal(a, b)
    except ImportError:  # pragma: no cover - torch is mandatory (POLICY-001)
        pass
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return not (
            isinstance(a, np.ndarray)
            and isinstance(b, np.ndarray)
            and a.shape == b.shape
            and np.array_equal(a, b)
        )
    return bool(a != b)


_TENSOR_DRAW_METHODS = (
    "uniform_",
    "normal_",
    "bernoulli_",
    "exponential_",
    "cauchy_",
    "log_normal_",
    "geometric_",
    "random_",
)
_TORCH_DRAW_FUNCTIONS = ("rand", "randn", "randint", "randperm", "bernoulli")


def _caller_site() -> str:
    for frame in reversed(traceback.extract_stack()[:-2]):
        filename = frame.filename
        if "runtime_ladder_step_parity" in filename:
            continue
        return f"{filename}:{frame.lineno}"
    return "<unknown>"


@contextmanager
def record_rng_draws() -> Iterator[list[dict[str, Any]]]:
    """Record torch RNG-consuming calls (fn + caller site), non-perturbing.

    Covers the in-place tensor initializers (which ``torch.nn.init`` — and
    therefore every module ``reset_parameters`` — bottoms out in) and the
    module-level sampling functions. Wrappers call straight through to the
    originals, so the consumed stream is bit-identical with and without
    instrumentation (pinned by test).
    """
    import torch

    draws: list[dict[str, Any]] = []
    originals: list[tuple[Any, str, Any]] = []

    def install(owner: Any, name: str) -> None:
        original = getattr(owner, name)

        def wrapper(*args: Any, __orig: Any = original, __name: str = name, **kwargs: Any) -> Any:
            draws.append({"fn": __name, "site": _caller_site()})
            return __orig(*args, **kwargs)

        originals.append((owner, name, original))
        setattr(owner, name, wrapper)

    for method in _TENSOR_DRAW_METHODS:
        if hasattr(torch.Tensor, method):
            install(torch.Tensor, method)
    for function in _TORCH_DRAW_FUNCTIONS:
        if hasattr(torch, function):
            install(torch, function)
    try:
        yield draws
    finally:
        for owner, name, original in originals:
            setattr(owner, name, original)


def one_loss_and_grads(model: Any, batch: Any, *, seed: int) -> dict[str, Any]:
    """One fixed-RNG forward+loss+backward; loss scalar + per-param grads.

    Uses ``compute_loss`` — the objective entry ``training_step`` wraps
    (``ptycho_torch/model.py``) — so no Trainer/optimizer coupling leaks in.
    """
    import torch

    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    model.train()
    model.zero_grad(set_to_none=True)
    loss = model.compute_loss(batch)
    loss_value = loss[0] if isinstance(loss, (tuple, list)) else loss
    loss_value.backward()
    grads: dict[str, dict[str, Any]] = {}
    for name, parameter in model.named_parameters():
        gradient = parameter.grad
        if gradient is None:
            grads[name] = {"sha256": None, "norm": None}
            continue
        grads[name] = {
            "sha256": _tensor_sha(gradient),
            "norm": float(gradient.detach().norm()),
            "tensor": gradient.detach().clone(),
        }
    return {"loss": float(loss_value.detach()), "grads": grads}


def compare_grads(
    report_a: Mapping[str, Any], report_b: Mapping[str, Any]
) -> dict[str, Any]:
    """Compare two one-step reports: loss delta + per-parameter grad parity."""
    import torch

    grads_a, grads_b = report_a["grads"], report_b["grads"]
    names = sorted(set(grads_a) | set(grads_b))
    differing: list[dict[str, Any]] = []
    max_rel_diff = 0.0
    first_divergent: str | None = None
    for name in names:
        entry_a, entry_b = grads_a.get(name), grads_b.get(name)
        if entry_a is None or entry_b is None or (
            (entry_a["sha256"] is None) != (entry_b["sha256"] is None)
        ):
            differing.append({"name": name, "reason": "presence"})
            first_divergent = first_divergent or name
            continue
        if entry_a["sha256"] == entry_b["sha256"]:
            continue
        tensor_a, tensor_b = entry_a.get("tensor"), entry_b.get("tensor")
        rel = None
        if isinstance(tensor_a, torch.Tensor) and isinstance(
            tensor_b, torch.Tensor
        ) and tensor_a.shape == tensor_b.shape:
            denom = float(tensor_a.abs().max())
            rel = float((tensor_a - tensor_b).abs().max()) / max(denom, 1e-30)
            max_rel_diff = max(max_rel_diff, rel)
        differing.append({"name": name, "max_rel_diff": rel})
        if first_divergent is None:
            first_divergent = name
    loss_delta = abs(float(report_a["loss"]) - float(report_b["loss"]))
    return {
        "identical": not differing and loss_delta == 0.0,
        "loss_delta": loss_delta,
        "max_rel_diff": max_rel_diff,
        "differing": differing,
        "first_divergent": first_divergent,
    }
