Analyze the PtychoPINN codebase and create a prioritized plan for adding module docstrings.

TASK: Identify all .py files in the ptycho/ directory (excluding __init__.py) and prioritize them based on dependency relationships. Foundational modules (with fewer dependencies) should come first.

FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS - NO OTHER TEXT:

---PRIORITIZED_MODULES_START---
ptycho/config.py
ptycho/tf_helper.py  
ptycho/diffsim.py
ptycho/raw_data.py
ptycho/loader.py
ptycho/model.py
ptycho/evaluation.py
ptycho/workflows.py
---PRIORITIZED_MODULES_END---

---DEPENDENCY_REPORT_START---
ptycho/config.py: No internal dependencies
ptycho/tf_helper.py: Depends on ptycho/config.py
ptycho/diffsim.py: Depends on ptycho/tf_helper.py, ptycho/config.py
ptycho/raw_data.py: Depends on ptycho/config.py, ptycho/tf_helper.py
ptycho/loader.py: Depends on ptycho/raw_data.py, ptycho/tf_helper.py
ptycho/model.py: Depends on ptycho/tf_helper.py, ptycho/diffsim.py, ptycho/config.py
ptycho/evaluation.py: Depends on ptycho/tf_helper.py, ptycho/model.py
ptycho/workflows.py: Depends on ptycho/model.py, ptycho/loader.py, ptycho/evaluation.py
---DEPENDENCY_REPORT_END---