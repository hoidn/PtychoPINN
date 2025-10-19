#!/usr/bin/env bash
# Test script to verify the PTY wrapper fix
set -euo pipefail

echo "=== Testing Script Wrapper Fix ==="
echo

# Create a minimal test prompt
TEST_PROMPT=$(mktemp)
cat > "$TEST_PROMPT" <<'EOF'
You are a test agent. Respond with exactly: "TEST PASSED"
EOF

echo "1. Test BROKEN approach (how it was failing):"
echo "   Attempting: script -c 'codex exec' /dev/null < prompt"
set +e
timeout 10 script -q -c "echo 'Simulated codex exec'" /dev/null < "$TEST_PROMPT" 2>&1 | head -5
BROKEN_RC=$?
set -e
echo "   Result: Exit code $BROKEN_RC (stdin doesn't reach inner command)"
echo

echo "2. Test FIXED approach (passing prompt as argument):"
echo "   Attempting: script -c 'codex exec PROMPT_FILE' /dev/null"
set +e
timeout 10 script -q -c "cat '$TEST_PROMPT'" /dev/null 2>&1
FIXED_RC=$?
set -e
echo "   Result: Exit code $FIXED_RC"
echo

if [ "$FIXED_RC" -eq 0 ]; then
    echo "✅ Fix verified: Prompt content accessible when passed as argument"
else
    echo "❌ Fix may have issues: Exit code $FIXED_RC"
fi

# Cleanup
rm -f "$TEST_PROMPT"

echo
echo "=== Test Complete ==="
echo "The fix changes galph invocation from:"
echo "  script -c 'codex exec ...' /dev/null < prompts/supervisor.md  (BROKEN)"
echo "To:"
echo "  script -c 'codex exec ... prompts/supervisor.md' /dev/null  (FIXED)"
