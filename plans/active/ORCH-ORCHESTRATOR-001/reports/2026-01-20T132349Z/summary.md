### Turn Summary
Added `test_combined_review_last_prompt_actor` regression test ensuring combined mode honors the `last_prompt_actor` state annotation for review cadence.
The test drives `run_combined_iteration` with review_every_n=2 on iteration 2, confirms reviewer runs only on galph turn, and validates state transitions so ralph selects `main.md`.
Updated `docs/TESTING_GUIDE.md` and `docs/development/TEST_SUITE_INDEX.md` with the new selector reference; combined mode now mirrors sync router cadence tests.
Next: commit changes if user approves.
Artifacts: plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-20T132349Z/ (pytest_combined_review.log, pytest_sync_router_review.log, pytest_orchestrator.log)
