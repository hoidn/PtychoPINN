# Traceback Logging Analysis - PtychoPINN Enhanced Logging System

## Problem Statement

The PtychoPINN enhanced logging system is not capturing Python tracebacks in log files. While console output and regular logging work correctly, uncaught exceptions and stderr output bypass the logging system entirely.

## Root Cause Analysis

### Current Implementation Limitations

The enhanced logging system in `ptycho/log_config.py` has a critical gap:

1. **Only stdout is redirected** - The `TeeStream` class redirects `sys.stdout` but not `sys.stderr`
2. **No exception hook** - Python's uncaught exception handler (`sys.excepthook`) is not configured
3. **stderr bypass** - All stderr output (including tracebacks) goes directly to console only

### What Currently Works ✅

- **Caught exceptions**: `logger.exception()` calls capture full tracebacks
- **Print statements**: `print()` calls are captured via stdout redirection
- **Manual logging**: All `logging.*` calls work correctly
- **Console output**: Everything still appears on console as expected

### What Doesn't Work ❌

- **Uncaught exceptions**: Tracebacks only appear in console, not log files
- **Stderr output**: `print(..., file=sys.stderr)` not captured
- **Library errors**: Many libraries write error messages to stderr
- **System errors**: OS-level errors often go to stderr

## Technical Investigation

### Testing Methodology

Created test script using the enhanced logging system to verify behavior:

```python
from ptycho.log_config import setup_logging
from ptycho.cli_args import get_logging_config
import logging
import sys

# Setup enhanced logging
setup_logging(Path("test_logs"), {})
logger = logging.getLogger(__name__)

# Test scenarios:
# 1. Caught exception with logger.exception() - ✅ WORKS
# 2. Print to stderr - ❌ MISSING FROM LOGS  
# 3. Uncaught exception - ❌ MISSING FROM LOGS
```

### Results Confirmed

- **stdout capture**: Perfect - all print statements logged
- **stderr capture**: Missing - stderr output not in log files
- **Exception capture**: Partial - only caught exceptions logged

## Solution Architecture

### Required Components

1. **Stderr Redirection**: Apply `TeeStream` to `sys.stderr`
2. **Exception Hook**: Install custom `sys.excepthook` for uncaught exceptions
3. **Restoration**: Ensure proper cleanup on shutdown

### Implementation Strategy

#### Component 1: Stderr Redirection
```python
# In setup_logging():
stderr_logger = logging.getLogger('stderr')
stderr_logger.setLevel(logging.ERROR)
stderr_logger.addHandler(file_handler)

original_stderr = sys.stderr
sys.stderr = TeeStream(original_stderr, LoggerWriter(stderr_logger, logging.ERROR))
```

#### Component 2: Exception Hook
```python
def exception_handler(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    
    # Log the uncaught exception
    logger = logging.getLogger('uncaught_exceptions')
    logger.critical("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
    # Call original handler for console output
    sys.__excepthook__(exc_type, exc_value, exc_traceback)

# Install the hook
sys.excepthook = exception_handler
```

#### Component 3: Cleanup
```python
def restore_logging():
    sys.stdout = getattr(sys.stdout, 'original', sys.stdout)
    sys.stderr = getattr(sys.stderr, 'original', sys.stderr)
    sys.excepthook = sys.__excepthook__

# Store restoration function for cleanup
_restore_function = restore_logging
```

## Implementation Plan

### Phase 1: Core Enhancement (Low Risk)
1. Add stderr redirection to `setup_logging()`
2. Install custom exception hook
3. Test with existing scripts

### Phase 2: Testing & Validation
1. Verify traceback capture in log files
2. Ensure console output unchanged
3. Test cleanup/restoration
4. Validate backward compatibility

### Phase 3: Integration
1. No changes needed to existing scripts
2. All scripts automatically benefit
3. Update documentation

## Expected Benefits

### Immediate Impact
- ✅ **Complete output capture**: All stderr and tracebacks logged
- ✅ **Better debugging**: Full context in log files
- ✅ **Automation friendly**: Errors captured in unattended runs
- ✅ **Zero breaking changes**: Existing code continues working

### Operational Benefits
- **Troubleshooting**: Complete error context in log files
- **Monitoring**: Systematic error tracking across all scripts
- **Debugging**: Full traceback capture for production issues
- **Compliance**: Complete audit trail of all script output

## Risk Assessment

### Risk Level: **LOW**

- **Backward compatibility**: 100% preserved
- **Console behavior**: Unchanged
- **Performance impact**: Minimal (same as current stdout redirection)
- **Complexity**: Low - extends existing proven pattern

### Mitigation
- **Rollback**: Simple restoration of original handlers
- **Testing**: Comprehensive validation before deployment
- **Incremental**: Can be deployed to subset of scripts first

## Recommended Next Steps

1. **Implement enhancement** in `ptycho/log_config.py`
2. **Test thoroughly** with various error scenarios
3. **Deploy gradually** to validate behavior
4. **Document** the complete output capture capability

This enhancement will provide **complete output capture** for the PtychoPINN logging system, ensuring all error information is preserved in log files while maintaining full backward compatibility.