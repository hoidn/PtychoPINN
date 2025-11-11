#!/bin/bash
set -e

export AUTHORITATIVE_CMDS_DOC=./docs/TESTING_GUIDE.md
export HUB="$PWD/plans/active/STUDY-SYNTH-FLY64-DOSE-OVERLAP-001/reports/2025-11-12T010500Z/phase_g_dense_full_run_verifier"

echo "Running test collection..."
pytest --collect-only tests/study/test_phase_g_dense_orchestrator.py -k post_verify_only_executes_chain -vv 2>&1 | tee "$HUB"/collect/pytest_collect_post_verify_only.log

echo "Running test execution..."
pytest tests/study/test_phase_g_dense_orchestrator.py::test_run_phase_g_dense_post_verify_only_executes_chain -vv 2>&1 | tee "$HUB"/green/pytest_post_verify_only.log

echo "Tests completed successfully!"
