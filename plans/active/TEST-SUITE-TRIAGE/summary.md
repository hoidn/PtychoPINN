## 2026-02-05
Ran non-integration pytest (`-m "not integration"`) and captured full log in `plans/active/TEST-SUITE-TRIAGE/reports/2026-02-05T064357Z/pytest.log`.
Observed 40 failures and triaged into clusters (ptychodus interop API, patch-stats CLI/config factory gaps, workflow wiring, params.cfg seeding, probe_big decoder mismatch, fixture checksum).
Drafted updated implementation plan in `docs/plans/2026-02-05-non-integration-test-failures.md` covering all remaining failures.
