"""Unit tests for ptycho_torch.train_lightning_only._resolve_seed.

Covers PTYCHO_TORCH_SEED environment-variable resolution only -- no
training, no Lightning invocation. See
.superpowers/sdd/ext/seed-plumbing-brief.md for the seed-plumbing spec.
"""
import pytest

from ptycho_torch.train_lightning_only import _resolve_seed


def test_resolve_seed_defaults_to_42_when_env_unset(monkeypatch):
    monkeypatch.delenv("PTYCHO_TORCH_SEED", raising=False)

    assert _resolve_seed() == 42


def test_resolve_seed_defaults_to_42_when_env_empty(monkeypatch):
    monkeypatch.setenv("PTYCHO_TORCH_SEED", "")

    assert _resolve_seed() == 42


def test_resolve_seed_uses_env_value_when_set(monkeypatch):
    monkeypatch.setenv("PTYCHO_TORCH_SEED", "11")

    assert _resolve_seed() == 11


def test_resolve_seed_raises_value_error_on_non_integer(monkeypatch):
    monkeypatch.setenv("PTYCHO_TORCH_SEED", "banana")

    with pytest.raises(ValueError, match="PTYCHO_TORCH_SEED"):
        _resolve_seed()
