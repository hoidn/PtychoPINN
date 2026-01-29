### Turn Summary
Centralized PTY selection with a helper that disables PTY for Claude in auto mode, and wired it through orchestrator, supervisor, and loop to avoid the hang regression.
Added a unit test asserting the Claude PTY selection behavior and hardened supervisor no-git tests to tolerate external workflow env values and new tee_run args.
Ran targeted orchestration tests and confirmed the suite passes after adjustments.
Next: rerun the real combined orchestrator with Claude to verify live output and clean exit.
Artifacts: plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-29T022122Z/ (pytest_orchestration_rerun2.log)

### Turn Summary
Disabled PTY usage for Claude by centralizing PTY selection logic and wiring it through orchestrator/loop/supervisor, keeping other agents on the existing PTY behavior.
Added a focused test that asserts the auto PTY mode avoids Claude so the harness won't reintroduce the hang.
Ran the orchestrator test module to confirm the new PTY selection behavior passes.
Next: rerun the orchestrator with Claude to confirm live streaming output now completes cleanly.
Artifacts: plans/active/ORCH-ORCHESTRATOR-001/reports/2026-01-29T021931Z/ (pytest_orchestrator.log)
