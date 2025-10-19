# CORRECTED Fix: PTY Wrapper Stdin Issue

## Problem

The original fix was **WRONG**. `codex exec` does NOT support reading from a file path:

```bash
# This DOES NOT WORK (what I originally tried):
codex exec prompts/supervisor.md
# ❌ Interprets "prompts/supervisor.md" as literal prompt text, not a file!

# This is the only way codex reads prompts:
codex exec < prompts/supervisor.md  # stdin
cat prompts/supervisor.md | codex exec  # piped stdin
```

## Corrected Solution

Instead of passing a file path, we **pipe the file content inside the script command**:

```python
# WRONG (original broken fix):
codex_with_prompt = codex_args + [str(prompt_file)]
script_cmd = ["script", "-q", "-c", shlex.join(codex_with_prompt), "/dev/null"]
# ❌ codex treats filename as literal text

# CORRECT (revised fix):
codex_cmd_str = shlex.join(codex_args)
script_cmd = [
    "script", "-q", "-c",
    f"cat {shlex.quote(str(prompt_file))} | {codex_cmd_str}",
    "/dev/null"
]
# ✅ cat provides stdin to codex inside the PTY
```

## How It Works

```bash
# Without script wrapper (original, before commit 3e27c16):
codex exec ... < prompts/supervisor.md
✅ Works - stdin comes from file redirection

# With broken script wrapper (commit 3e27c16):
script -c "codex exec ..." /dev/null < prompts/supervisor.md
❌ Fails - stdin consumed by script, not passed through PTY

# With corrected script wrapper (this fix):
script -c "cat prompts/supervisor.md | codex exec ..." /dev/null
✅ Works - cat runs inside PTY, pipes to codex
```

## Changes Made

**File:** `scripts/orchestration/supervisor.py`

**Lines 340-355:** Corrected to use `cat file | codex` inside script command

```python
if shutil.which("script"):
    codex_cmd_str = shlex.join(codex_args)
    script_cmd = [
        "script",
        "-q",
        "-c",
        f"cat {shlex.quote(str(prompt_file))} | {codex_cmd_str}",
        "/dev/null",
    ]
    rc = tee_run(script_cmd, None, iter_log)
else:
    rc = tee_run(codex_args, prompt_file, iter_log)
```

## Testing

```bash
# Create test prompt
echo "Say HELLO WORLD" > /tmp/test.txt

# Test the corrected approach
script -q -c "cat /tmp/test.txt | codex exec -m gpt-4o-mini --dangerously-bypass-approvals-and-sandbox" /dev/null

# Should see codex receive and process the prompt
```

## Why This Works

1. **`script` creates a PTY** for the entire command string
2. **`cat` runs inside the PTY** and reads the prompt file
3. **The pipe (`|`) runs inside the PTY** so stdin flows correctly
4. **`codex exec` receives stdin** from cat, just like normal piping

The key insight: **the pipe must be inside the `-c` command string**, not outside the script wrapper.

## Status

✅ Syntax validated
⏳ Awaiting deployment test with real galph run
