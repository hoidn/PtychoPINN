# Test Strategy — DEBUG-SIM-LINES-DOSE-001

## Scope
This initiative is primarily evidence-driven. Tests are only added if a concrete regression is identified in a core component (e.g., reassembly, probe normalization, or grouping).

## Test Triggers
- Add a targeted pytest if Phase C introduces a code change that affects:
  - `ptycho/tf_helper.py` reassembly or translation
  - `ptycho/probe.py` normalization/masking
  - `ptycho/raw_data.py` grouping/offset generation

## Test Style
- Prefer a single, minimal pytest with a tiny synthetic input (<= 8 patches).
- Mark slow paths with `@pytest.mark.slow`.
- Avoid GPU-only assumptions; tests must run on CPU.

## Evidence First
- Phase A/B evidence runs do not require pytest.
- If no code changes are made, note "N/A (evidence-only)" in the checklist.

## Reporting
- Log selector(s) and results under `plans/active/DEBUG-SIM-LINES-DOSE-001/reports/<timestamp>/`.
