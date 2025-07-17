# Path Conventions & File Organization Guide

*Version 2.0 - The authoritative guide for project file organization*

---

## 📁 **DIRECTORY STRUCTURE**

```
project-root/
│
├── PROJECT_STATUS.md                    # Master status tracker (ALWAYS at root)
│
├── plans/                              # All initiative planning documents
│   ├── active/                         # Currently active initiatives
│   │   └── <initiative-name>/          # One folder per active initiative
│   │       ├── plan.md                 # R&D specification document
│   │       ├── implementation.md       # Phased implementation plan
│   │       ├── phase_1_checklist.md    # Detailed checklist for phase 1
│   │       ├── phase_2_checklist.md    # Checklist for phase 2 (if exists)
│   │       ├── ...                     # Additional phase checklists
│   │       └── phase_final_checklist.md # Final validation phase checklist
│   │
│   ├── archive/                        # Completed initiatives (read-only)
│   │   └── <YYYY-MM>-<initiative-name>/ # Timestamped archive folders
│   │       └── <all files from active>  # Complete snapshot at completion
│   │
│   └── templates/                      # Document templates (optional)
│       ├── plan_template.md
│       ├── implementation_template.md
│       └── checklist_template.md
│
├── src/                                # Source code
├── tests/                              # Test files  
├── docs/                               # Project documentation
└── README.md                           # Project overview
```

---

## 📋 **NAMING CONVENTIONS**

### Initiative Names
- **Format:** `kebab-case` (lowercase with hyphens)
- **Length:** 2-5 words, descriptive but concise
- **Language:** English, technical terms allowed

✅ **Good Examples:**
- `coordinate-based-alignment`
- `multi-gpu-support`
- `frc-metric-implementation`
- `legacy-config-removal`

❌ **Bad Examples:**
- `CoordinateBasedAlignment` (wrong case)
- `coordinate_based_alignment` (underscores)
- `align` (too vague)
- `implement-new-feature-for-testing-validation-and-documentation` (too long)

### File Names Within Initiatives
| File Type | Name | Purpose |
|-----------|------|---------|
| R&D Plan | `plan.md` | Problem statement and objectives |
| Implementation | `implementation.md` | Phased execution plan |
| Phase Checklists | `phase_<n>_checklist.md` | Detailed task lists |
| Final Phase | `phase_final_checklist.md` | Validation & documentation |

**Phase Numbering:**
- Use integers: `1`, `2`, `3` (not `01`, `02`)
- Final phase is always `final` (not a number)
- No gaps in numbering

### Archive Timestamps
- **Format:** `YYYY-MM-<initiative-name>`
- **Example:** `2024-03-coordinate-based-alignment`
- **Purpose:** Chronological sorting and uniqueness

---

## 🔄 **WORKFLOW PATHS**

### 1. Starting a New Initiative

```bash
# AI creates these:
plans/active/my-new-feature/
├── plan.md                 # Created by /customplan
└── implementation.md       # Created by /implementation

# Then generates:
├── phase_1_checklist.md    # Created by /complete-phase
├── phase_2_checklist.md    # As needed
└── phase_final_checklist.md # Always last
```

### 2. During Development

```bash
# Working files always in:
plans/active/<current-initiative>/

# Check progress:
cat PROJECT_STATUS.md
cat plans/active/<n>/phase_*_checklist.md
```

### 3. Completing an Initiative

```bash
# AI moves entire folder:
plans/active/my-feature/ → plans/archive/2024-03-my-feature/

# Updates PROJECT_STATUS.md:
- Removes from "Current Active Initiative"
- Adds to "Completed Initiatives"
```

---

## 📜 **PATH RESOLUTION RULES**

### From PROJECT_STATUS.md
```markdown
**Path:** `plans/active/coordinate-based-alignment/`
```
This path is the source of truth for the current initiative location.

### From Any Command
1. Read `PROJECT_STATUS.md` → Get initiative path
2. Navigate to path → Find specific file
3. Verify file markers → Ensure correct file

### File Markers
Each file type MUST contain a unique marker in the first few lines:

| File | Required Marker |
|------|-----------------|
| plan.md | `# R&D Plan:` |
| implementation.md | `<!-- ACTIVE IMPLEMENTATION PLAN -->` |
| checklist.md | `# Phase <n>:` |

---

## 🛡️ **VALIDATION RULES**

### Path Validation
```python
def validate_initiative_path(path):
    """Ensure path follows conventions."""
    assert path.startswith("plans/active/") or path.startswith("plans/archive/")
    assert path.endswith("/")
    name = path.split("/")[-2]
    assert name.islower()
    assert "_" not in name
    assert name.replace("-", "").isalnum()
```

### File Existence Checks
Before any operation:
1. Verify `PROJECT_STATUS.md` exists at root
2. Verify initiative folder exists
3. Verify expected files present
4. Check file markers match expected type

---

## 🚀 **MIGRATION GUIDE**

### From Legacy Structure
```bash
# Old structure:
docs/studies/multirun/plan_xyz.md
docs/refactor/eval/implementation_abc.md

# Migration commands:
mkdir -p plans/active plans/archive
mv docs/studies/*/plan_*.md plans/active/
mv docs/refactor/*/implementation_*.md plans/active/

# Rename to standard:
cd plans/active/multirun/
mv plan_xyz.md plan.md
mv implementation_abc.md implementation.md
```

### Bulk Archive Operation
```bash
# Archive all completed initiatives
for dir in plans/active/*/; do
    if grep -q "✅ Complete" "$dir/implementation.md"; then
        name=$(basename "$dir")
        mv "$dir" "plans/archive/$(date +%Y-%m)-$name/"
    fi
done
```

---

## 🤖 **AUTOMATION HELPERS**

### Create Initiative Structure
```bash
#!/bin/bash
# create-initiative.sh
NAME=$1
mkdir -p "plans/active/$NAME"
echo "# R&D Plan: $NAME" > "plans/active/$NAME/plan.md"
echo "Created: plans/active/$NAME/"
```

### Archive Initiative
```bash
#!/bin/bash  
# archive-initiative.sh
NAME=$1
ARCHIVE="plans/archive/$(date +%Y-%m)-$NAME"
mv "plans/active/$NAME" "$ARCHIVE"
echo "Archived to: $ARCHIVE/"
```

### Find Current Initiative
```bash
#!/bin/bash
# current-initiative.sh
grep "**Path:**" PROJECT_STATUS.md | grep -o 'plans/active/[^/]*'
```

---

## ❓ **FREQUENTLY ASKED QUESTIONS**

**Q: Can I have multiple active initiatives?**
A: The system is designed for single initiative focus, but you can adapt PROJECT_STATUS.md to track primary/secondary initiatives.

**Q: What if an initiative name needs to change?**
A: Rename the folder and update all references in PROJECT_STATUS.md. Use git to track the rename.

**Q: How do I handle hotfixes?**
A: Create a minimal plan in `plans/active/hotfix-<issue>/` and fast-track through phases.

**Q: Where do I put research/exploration that isn't an initiative?**
A: Use `docs/research/` or `docs/explorations/` outside the plans structure.

**Q: Can I nest initiatives?**
A: No. Keep initiatives flat. Use phase decomposition for complexity.

---

## 📐 **QUICK REFERENCE CARD**

| What | Where | Example |
|------|-------|---------|
| Current status | `./PROJECT_STATUS.md` | Always at root |
| Active work | `plans/active/<n>/` | `plans/active/frc-metric/` |
| Completed work | `plans/archive/<YYYY-MM>-<n>/` | `plans/archive/2024-03-frc-metric/` |
| R&D spec | `<initiative>/plan.md` | Always named `plan.md` |
| Implementation | `<initiative>/implementation.md` | Always named `implementation.md` |
| Phase details | `<initiative>/phase_<n>_checklist.md` | `phase_1_checklist.md` |

**Initiative Lifecycle:**
1. `/customplan` → Creates `plans/active/<n>/plan.md`
2. `/implementation` → Creates `implementation.md`  
3. `/complete-phase` → Creates checklists, tracks progress
4. Archive when done → Move to `plans/archive/`

**Git Integration:**
```bash
git add plans/active/<initiative-name>/
git commit -m "[<Initiative>] <Description>"
git push origin feature/<initiative-name>
```