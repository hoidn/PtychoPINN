from types import SimpleNamespace
from ptycho_torch.workflows import components


def test_resolve_nphotons_metadata_precedence():
    config = SimpleNamespace(nphotons=1e6)
    data = SimpleNamespace(metadata={"physics_parameters": {"nphotons": 1e9}})
    nphotons, source = components._resolve_nphotons(data, config)
    assert nphotons == 1e9
    assert source == "metadata"


def test_resolve_nphotons_fallback_to_config():
    config = SimpleNamespace(nphotons=1e6)
    data = SimpleNamespace(metadata=None)
    nphotons, source = components._resolve_nphotons(data, config)
    assert nphotons == 1e6
    assert source == "config"
