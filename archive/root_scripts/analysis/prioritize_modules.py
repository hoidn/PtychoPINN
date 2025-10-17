#!/usr/bin/env python3
"""
Script to prioritize modules for documentation based on dependency analysis.
Modules with fewer dependencies (more foundational) should be documented first.
"""

import json
import sys
from pathlib import Path

def parse_dependency_report(report_path):
    """Parse the pydeps JSON output to extract dependency information."""
    with open(report_path, 'r') as f:
        data = json.load(f)
    
    dependencies = {}
    for module_name, module_info in data.items():
        if module_name.startswith('ptycho.') and 'imports' in module_info:
            # Count only internal ptycho dependencies
            internal_deps = [dep for dep in module_info['imports'] 
                           if dep.startswith('ptycho.') and dep != module_name]
            dependencies[module_name] = internal_deps
    
    return dependencies

def load_modules_list(modules_file):
    """Load the list of modules to document."""
    with open(modules_file, 'r') as f:
        modules = [line.strip() for line in f if line.strip()]
    
    # Convert file paths to module names
    module_names = []
    for module_path in modules:
        if module_path.endswith('.py'):
            # Convert path like 'ptycho/config/config.py' to 'ptycho.config.config'
            module_name = module_path.replace('/', '.').replace('.py', '')
            module_names.append((module_path, module_name))
    
    return module_names

def prioritize_modules(modules_list, dependencies):
    """Sort modules by dependency count (ascending)."""
    def dependency_key(module_tuple):
        module_path, module_name = module_tuple
        dep_count = len(dependencies.get(module_name, []))
        
        # Special prioritization rules:
        # 1. Core config and params modules first (infrastructure)
        if 'config' in module_name or 'params' in module_name:
            return (0, dep_count, module_name)
        
        # 2. Foundational modules next (fewer dependencies)
        # 3. Regular prioritization by dependency count
        return (1, dep_count, module_name)
    
    return sorted(modules_list, key=dependency_key)

def main():
    # File paths
    dependency_report = Path('ptycho/dependency_report.txt')
    modules_file = Path('modules_to_document.txt')
    output_file = Path('modules_prioritized.txt')
    
    if not dependency_report.exists():
        print(f"Error: {dependency_report} not found")
        sys.exit(1)
    
    if not modules_file.exists():
        print(f"Error: {modules_file} not found")
        sys.exit(1)
    
    # Parse dependencies and load modules
    dependencies = parse_dependency_report(dependency_report)
    modules_list = load_modules_list(modules_file)
    
    # Prioritize modules
    prioritized = prioritize_modules(modules_list, dependencies)
    
    # Write prioritized list
    with open(output_file, 'w') as f:
        for module_path, module_name in prioritized:
            dep_count = len(dependencies.get(module_name, []))
            f.write(f"{module_path}  # {module_name} ({dep_count} deps)\n")
    
    print(f"Created prioritized module list: {output_file}")
    print(f"Total modules to document: {len(prioritized)}")
    
    # Show first few for verification
    print("\nFirst 10 modules (highest priority):")
    for i, (module_path, module_name) in enumerate(prioritized[:10]):
        dep_count = len(dependencies.get(module_name, []))
        print(f"  {i+1:2d}. {module_path} ({dep_count} dependencies)")

if __name__ == '__main__':
    main()