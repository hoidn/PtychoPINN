# Debugging Guide

## Suggested Triage Steps

1. Confirm `params.cfg` matches your intended `N`, `gridsize`, and `nphotons`.
2. Verify NPZ keys and shapes match `specs/data_contracts.md`.
3. Run `pytest tests/test_generic_loader.py` to validate loader behavior.
4. Reproduce with a small dataset (few patterns) to isolate shape issues.
