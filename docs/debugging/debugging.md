# Standard Debugging Methodology

To resolve issues quickly and consistently, follow this four-step procedure for every new bug or anomaly:

1. **Verify Data Contracts**  
   Confirm that all inputs and intermediates conform to the authoritative specifications in `specs/data_contracts.md`. Mis-shaped arrays or missing keys often masquerade as model bugs.

2. **Check Configuration Synchronization**  
   Ensure modern dataclasses and legacy `params.cfg` state are aligned (call `update_legacy_dict` when required). Mismatched gridsize or intensity parameters are the most frequent root causes of TensorFlow/PyTorch divergence.

3. **Isolate the Component**  
   Narrow the failing pathway by running the smallest script or helper that exhibits the issue. Replace external dependencies with fixtures or canned data to confirm whether the fault lives in data loading, physics layers, or training loops.

4. **Write a Minimal Failing Test**  
   Capture the scenario as an automated test before fixing it. This locks in the failure mode, prevents regressions, and documents the intended behavior once the fix lands.

Document each step in the active task log (`docs/fix_plan.md` or relevant initiative notes) so that future investigators can replay your reasoning.
