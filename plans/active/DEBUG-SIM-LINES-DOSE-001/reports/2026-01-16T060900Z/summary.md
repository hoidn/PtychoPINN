### Turn Summary
Implemented `_update_max_position_jitter_from_offsets()` with padded-size parity handling, wired it into the workflow container factory, and aligned the SIM-LINES reassembly telemetry to use the new jitter updates.
Resolved the integration test failure caused by odd padded sizes by enforcing N-parity in the required canvas calculation, then added pytest coverage and refreshed test docs.
Re-ran the targeted workflow selector, the integration marker, and the gs1/gs2 custom reassembly CLI to confirm `fits_canvas=True` with zero loss.
Next: run the Phase C2 gs1/gs2 ideal telemetry or move to the inference smoke validation once this padded-size update is accepted.
Artifacts: plans/active/DEBUG-SIM-LINES-DOSE-001/reports/2026-01-16T060900Z/ (reassembly_cli.log, pytest_integration.log)
