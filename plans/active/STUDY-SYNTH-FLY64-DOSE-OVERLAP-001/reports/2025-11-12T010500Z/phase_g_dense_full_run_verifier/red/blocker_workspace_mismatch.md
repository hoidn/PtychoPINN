# Blocker: Workspace Mismatch

**Timestamp:** 2025-11-11 (Ralph loop sync i=283)
**Status:** BLOCKED - Cannot proceed

## Issue

Ralph invoked from `/home/ollie/Documents/PtychoPINN2` but the Do Now in `input.md` requires execution from `/home/ollie/Documents/PtychoPINN`.

## Evidence

```bash
$ pwd -P
/home/ollie/Documents/PtychoPINN2
```

## Requirements (from input.md)

- Line 32: "Pitfalls To Avoid - Running any command from `/home/ollie/Documents/PtychoPINN2`"
- How-To Map line 1: `test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN"`

## Impact

Cannot execute:
- pytest selectors (will fail to find test modules)
- `run_phase_g_dense.py` execution (hub paths will resolve incorrectly)
- Phase D-G pipeline (will abort as documented in prior `analysis/blocker.log`)

## Prior Occurrences

This is identical to blockers documented in:
- 2025-11-11T115954Z
- 2025-11-11T120800Z

## Required Action

**Supervisor must re-invoke Ralph from `/home/ollie/Documents/PtychoPINN` working directory.**

The execution environment must satisfy:
```bash
test "$(pwd -P)" = "/home/ollie/Documents/PtychoPINN" || exit 1
```

## Workspace Verification

Both repositories exist:
```
drwxrwxr-x 310 ollie ollie   32768 Nov 11 04:08 PtychoPINN
drwxrwxr-x 218 ollie ollie   16384 Nov 11 05:10 PtychoPINN2
```

The task explicitly requires `PtychoPINN` (not `PtychoPINN2`).
