# Config Bridge Spec

- Dataclass configs are defined in `ptycho/config/config.py`.
- `update_legacy_dict()` copies dataclass values into `params.cfg`.
- Core modules read from `params.cfg` at runtime.
