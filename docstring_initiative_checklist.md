# PtychoPINN Documentation Agent Checklist

**Mission:** Complete module-level documentation for the `ptycho/` library
**Target:** High-quality, interface-focused docstrings for all Python modules
**Constraint:** Each docstring must be <15% of module size
**Standard:** Professional API documentation with realistic usage examples

## 🎯 Executive Summary

This initiative will document every `.py` module in the `ptycho/` library with comprehensive docstrings that explain:
- **Purpose & Role** in the PtychoPINN architecture
- **Public Interface** with parameter effects
- **Data Contracts** for input/output formats
- **Integration Patterns** with realistic multi-step examples
- **Dependencies** including legacy system interactions

## 📋 Implementation Phases

### Phase 1: Discovery & Analysis
| Task | Status | Description | Verification |
|------|--------|-------------|--------------|
| **1.1** | `[ ]` | **Module Inventory** <br>Generate complete list of target modules excluding `__init__.py` files | `find ptycho -name "*.py" -not -name "__init__.py" \| wc -l` reports correct count |
| **1.2** | `[ ]` | **Dependency Mapping** <br>Create visual and textual dependency analysis | Both `dependency_graph.svg` and `dependency_report.txt` exist in `ptycho/` |
| **1.3** | `[ ]` | **Architecture Summary** <br>Extract key patterns from developer guide for sub-agent context | Document captures dual-system architecture and data flow patterns |
| **1.4** | `[ ]` | **Documentation Audit** <br>Identify which modules already have quality docstrings vs need work | Progress tracker shows current state and remaining work |

### Phase 2: Documentation Generation  
| Task | Status | Description | Verification |
|------|--------|-------------|--------------|
| **2.1** | `[ ]` | **Core Module Documentation** <br>Document critical infrastructure modules first (model.py, loader.py, etc.) | Sub-agents complete docstrings following template structure |
| **2.2** | `[ ]` | **Data Pipeline Documentation** <br>Document data processing and transformation modules | All data contracts and tensor shapes are explicitly documented |
| **2.3** | `[ ]` | **Utility Module Documentation** <br>Document helper and support modules | Usage examples show realistic integration with core modules |
| **2.4** | `[ ]` | **Quality Gate Review** <br>Verify each docstring meets size and quality requirements | All docstrings pass anti-pattern checks and size constraints |

### Phase 3: Integration & Validation
| Task | Status | Description | Verification |
|------|--------|-------------|--------------|
| **3.1** | `[ ]` | **Cross-Reference Validation** <br>Ensure module relationships are accurately documented | Dependency claims match actual import relationships |
| **3.2** | `[ ]` | **Consistency Review** <br>Verify terminology and style consistency across all docstrings | No conflicting architectural descriptions or terminology |
| **3.3** | `[ ]` | **Style Compliance** <br>Run automated linting and fix any formatting issues | `pydocstyle` passes with project-appropriate ignore flags |
| **3.4** | `[ ]` | **Final Integration** <br>Commit all documentation with proper attribution | Git commit includes all modified files with comprehensive message |

## 🤖 Sub-Agent Specifications

### Documentation Sub-Agent Instructions

**Role:** Module Documentation Specialist  
**Input:** Target module path, dependency report, architectural context  
**Output:** Complete module docstring meeting all requirements

#### Analysis Requirements
1. **Module Classification** - Determine if module is:
   - Data transformation focused (document tensor shapes/formats)
   - Logic/workflow focused (document conditional behavior)
   - Interface/API focused (document public methods/classes)

2. **Dependency Investigation** - Identify:
   - Direct imports and exports
   - Hidden dependencies (especially `ptycho.params`)
   - Consumer modules and usage patterns
   - Position in data pipeline

3. **Legacy System Detection** - Check for:
   - Global state dependencies
   - Configuration system usage (modern vs legacy)
   - Dual-system architecture interactions

#### Docstring Structure Template

**Choose appropriate style based on module nature:**

##### Style A: Data Transformation Modules
```python
"""
[One-line summary of transformation purpose]

[2-3 sentences describing role in PtychoPINN pipeline and primary consumers]

Data Contracts:
    Input: [Specific tensor formats with shapes]
    Output: [Specific tensor formats with shapes]
    
Key Functions:
    function_name(params) -> return_type
        Brief description of purpose and critical parameter effects
        
Workflow Integration:
    ```python
    # Multi-step realistic example showing:
    # 1. Data preparation
    # 2. Function call with real parameters  
    # 3. Result usage in downstream module
    ```

Dependencies & Notes:
    - [Any legacy system dependencies]
    - [Performance characteristics]
    - [Special requirements or constraints]
"""
```

##### Style B: Logic/Control Modules
```python
"""
[One-line summary of control/logic purpose]

[2-3 sentences describing decision-making role and integration points]

Behavioral Modes:
    - Condition A: [Specific behavior description]
    - Condition B: [Alternative behavior description]
    
Public Interface:
    primary_function(config, data) -> result
        [Parameter effects on system behavior]
        [Configuration dependencies]
        
Usage Pattern:
    ```python
    # Realistic workflow showing:
    # 1. Configuration setup
    # 2. Data preparation
    # 3. Function execution
    # 4. Result handling
    ```
    
Architectural Integration:
    - [Position in data flow]
    - [Integration with legacy/modern systems]
    - [State management approach]
"""
```

#### Quality Requirements
- **Size Constraint:** <15% of module file size
- **No Anti-Patterns:** Avoid vague summaries, marketing language, implementation details
- **Realistic Examples:** Multi-step workflows, not isolated function calls  
- **Accurate Dependencies:** Based on actual code analysis, not assumptions
- **Consistent Terminology:** Align with existing high-quality docstrings

### Verification Sub-Agent Instructions

**Role:** Documentation Quality Assurance  
**Input:** All new docstrings, dependency reports, architecture documentation  
**Output:** Consistency report with pass/fail assessment

#### Verification Checklist
1. **Architectural Accuracy**
   - Module roles correctly described
   - Data flow relationships accurate
   - Integration patterns realistic

2. **Dependency Consistency** 
   - Import relationships match documented consumers
   - Legacy system dependencies properly noted
   - Cross-references between modules align

3. **Documentation Quality**
   - All examples are executable and realistic
   - Terminology consistent across modules
   - Size constraints satisfied
   - Template structure followed

4. **Technical Correctness**
   - Data shapes and formats accurate
   - Parameter descriptions match actual function signatures
   - Behavioral descriptions match code logic

## 🚀 Execution Strategy

### Module Priority Tiers

**Tier 1: Core Infrastructure** (Immediate Priority)
- `model.py` - Neural network architecture
- `loader.py` - Data loading and transformation  
- `tf_helper.py` - Tensor operations
- `raw_data.py` - Data ingestion
- `evaluation.py` - Metrics and assessment

**Tier 2: Data Pipeline** (High Priority)  
- `diffsim.py` - Physics simulation
- `physics.py` - Domain-specific computations
- `fourier.py` - Frequency domain operations
- `image/` modules - Image processing

**Tier 3: Utilities & Support** (Medium Priority)
- `visualization.py` - Display and plotting
- `misc.py` - Helper functions
- `cli_args.py` - Command-line interface
- `log_config.py` - Logging system

**Tier 4: Specialized Features** (Lower Priority)
- `experimental.py` - Research features
- `workflows/` modules - High-level orchestration
- `config/` modules - Configuration management

### Success Metrics

- **Coverage:** 100% of modules have docstrings meeting size requirements
- **Quality:** All docstrings pass anti-pattern review
- **Consistency:** Cross-references validated against dependency graph
- **Integration:** Documentation enables new developers to understand module interfaces
- **Maintainability:** Docstrings provide sufficient context for future modifications

### Deliverables

1. **Documented Codebase** - All `ptycho/` modules with comprehensive docstrings
2. **Dependency Analysis** - Visual and textual maps of module relationships  
3. **Progress Documentation** - Tracking of completion status and quality reviews
4. **Consistency Report** - Final validation of architectural accuracy and cross-references
5. **Integration Commit** - Professional git commit with all documentation updates

This systematic approach ensures thorough, accurate, and maintainable documentation that serves both current development needs and future codebase evolution.