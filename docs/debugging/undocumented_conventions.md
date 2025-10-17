# Undocumented Conventions & Gotchas

This file captures behaviors that are easy to miss in formal specs but routinely trip up developers.

- **Two-System Awareness:** Many helper modules implicitly assume either the legacy grid-based workflow or the modern coordinate-based pipeline. Confirm which system you are in before calling shared utilities to avoid leaking global `params.cfg` state into modern code paths.
- **Legacy Sync Order:** Anytime you modify a dataclass configuration during a debugging session, immediately propagate the change with `update_legacy_dict(params.cfg, config)`—otherwise legacy helpers will keep operating with stale values.
