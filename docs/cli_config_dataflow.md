# CLI Config Dataflow

- `scripts/training/train.py` loads YAML -> builds `TrainingConfig` -> calls
  `update_legacy_dict(params.cfg, config)`.
- Core modules still read from `params.cfg`.
- `ptycho/train.py` uses argparse and writes directly to `params.cfg`.
