# Enhanced Logging Implementation Plan

## 🎯 **Objective**
Implement tee-style logging that provides flexible control over console vs file output, with options for quiet mode (automated scripts) and verbose mode (debugging).

## 📋 **Implementation Phases**

### **Phase 1: Core Logging Enhancement**

#### 1.1 Update `ptycho/log_config.py`
**Current Function:**
```python
def setup_logging(output_dir: Path):
    """Basic centralized logging with fixed DEBUG->file, INFO->console"""
```

**Enhanced Function:**
```python
def setup_logging(
    output_dir: Path,
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG, 
    quiet: bool = False,
    verbose: bool = False
) -> logging.Logger:
    """
    Tee-style logging with flexible console/file control.
    
    Args:
        output_dir: Directory for log files
        console_level: Minimum level for console (default: INFO)
        file_level: Minimum level for file (default: DEBUG)
        quiet: If True, disable console output entirely
        verbose: If True, set console_level to DEBUG
    """
```

**Key Features:**
- Always write to file (preserves complete record)
- Optional console output (can be disabled for automation)
- Separate level control for console vs file
- Convenience flags: `quiet` and `verbose`

#### 1.2 Backward Compatibility
- Default behavior unchanged: INFO to console, DEBUG to file
- All existing calls to `setup_logging(output_dir)` continue working

### **Phase 2: Command-Line Integration**

#### 2.1 Add Shared Argument Parser Components
Create `ptycho/cli_args.py`:
```python
def add_logging_arguments(parser):
    """Add logging-related arguments to any argument parser"""
    logging_group = parser.add_argument_group('logging options')
    logging_group.add_argument('--quiet', action='store_true',
                              help='Disable console output (file logging only)')
    logging_group.add_argument('--verbose', action='store_true', 
                              help='Enable DEBUG output to console')
    logging_group.add_argument('--console-level', choices=['DEBUG', 'INFO', 'WARN', 'ERROR'],
                              default='INFO', help='Console logging level')
    return logging_group
```

#### 2.2 Update Training Script
**File:** `scripts/training/train.py`
```python
# Add logging arguments to existing parser
from ptycho.cli_args import add_logging_arguments

def parse_arguments():
    parser = argparse.ArgumentParser(...)
    # ... existing arguments ...
    add_logging_arguments(parser)
    return parser.parse_args()

def main():
    args = parse_arguments()
    config = setup_configuration(args, args.config)
    
    # Enhanced logging setup
    setup_logging(
        Path(config.output_dir),
        console_level=getattr(logging, args.console_level),
        quiet=args.quiet,
        verbose=args.verbose
    )
```

#### 2.3 Update Inference Script
**File:** `scripts/inference/inference.py`
- Same pattern as training script
- Add logging arguments to existing parser
- Update `setup_logging()` call with new parameters

### **Phase 3: Testing & Validation**

#### 3.1 Enhanced Unit Tests
**File:** `tests/test_log_config.py`

**New Test Cases:**
```python
def test_quiet_mode_disables_console():
    """Test that quiet=True creates no console handler"""
    
def test_verbose_mode_enables_debug_console():
    """Test that verbose=True sets console to DEBUG level"""
    
def test_custom_console_level():
    """Test custom console_level parameter"""
    
def test_backward_compatibility():
    """Test that old setup_logging(output_dir) calls still work"""
```

#### 3.2 Integration Testing
- Test with actual training runs using different flag combinations
- Verify log content matches expected levels
- Verify console output matches expected levels

### **Phase 4: Documentation Updates**

#### 4.1 Update Core Documentation
**Files to Update:**
- `docs/DEVELOPER_GUIDE.md` - Section 6: Centralized Logging
- `CLAUDE.md` - Add logging options examples
- `scripts/training/README.md` - Add logging options section
- `scripts/inference/README.md` - Add logging options section

#### 4.2 Usage Examples
```bash
# Quiet mode (automation-friendly)
ptycho_train --output_dir my_run --quiet

# Verbose mode (debugging)
ptycho_train --output_dir my_run --verbose  

# Custom console level
ptycho_train --output_dir my_run --console-level WARN

# Normal mode (unchanged)
ptycho_train --output_dir my_run
```

### **Phase 5: Orchestration Script Integration**

#### 5.1 Update Probe Generalization Study
**File:** `scripts/studies/run_probe_generalization_study.sh`

**Enhancement:** Add option to run individual workflows in quiet mode:
```bash
# In execute_arm() function
if [ "$QUIET_WORKFLOWS" = true ]; then
    local comparison_command="./scripts/run_comparison.sh '$train_data' '$test_data' '$arm_output_dir' --n-train-images 2000 --quiet"
else
    local comparison_command="./scripts/run_comparison.sh '$train_data' '$test_data' '$arm_output_dir' --n-train-images 2000"
fi
```

#### 5.2 Update Comparison Script
**File:** `scripts/run_comparison.sh`
- Add `--quiet` flag that gets passed to training/inference calls
- Maintain orchestration-level progress output while quieting individual workflows

## 🔄 **Implementation Order**

1. **Core Enhancement** (Phase 1): Update `log_config.py` with new function signature
2. **CLI Integration** (Phase 2): Add command-line options to scripts  
3. **Testing** (Phase 3): Comprehensive test coverage
4. **Documentation** (Phase 4): Update all relevant docs
5. **Orchestration** (Phase 5): Enhance study scripts for automation

## 🎛️ **Flag Combinations & Behavior**

| Command | Console Output | File Output | Use Case |
|---------|---------------|-------------|----------|
| `ptycho_train --output_dir run` | INFO+ | DEBUG+ | **Default** - Interactive training |
| `ptycho_train --output_dir run --quiet` | None | DEBUG+ | **Automation** - Script orchestration |
| `ptycho_train --output_dir run --verbose` | DEBUG+ | DEBUG+ | **Debugging** - Detailed troubleshooting |
| `ptycho_train --output_dir run --console-level WARN` | WARN+ | DEBUG+ | **Reduced noise** - Less console clutter |

## ✅ **Success Criteria**

- [ ] All existing code continues working (backward compatibility)
- [ ] New logging options work as specified
- [ ] Unit tests pass with >90% coverage
- [ ] Documentation accurately reflects new capabilities  
- [ ] Orchestration scripts can run workflows quietly
- [ ] Real-world testing confirms improved usability

## 🚀 **Benefits Delivered**

1. **For Interactive Users**: Unchanged experience, with options for more/less output
2. **For Automation**: Clean orchestration with `--quiet` mode
3. **For Debugging**: Enhanced visibility with `--verbose` mode  
4. **For Integration**: Flexible console levels for different environments

This plan maintains the single-source-of-truth principle while adding the flexibility needed for both human and automated workflows.