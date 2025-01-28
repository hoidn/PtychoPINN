from pathlib import Path
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import yaml
import sys

def run_taskspec(taskspec_path: str, summary_path: str, architect_model: str = '4o-mini', editor_model: str = '4o-mini'):
    """
    Process files according to the task specification.

    Args:
        taskspec_path (str): Path to task specification file (e.g. taskspec.md)
        summary_path (str): Path to task summary file (tochange.yaml) # TODO deprecated
        architect_model (str): Name of the model to use for architecture decisions (default: '4o-mini')
        editor_model (str): Name of the model to use for code editing (default: '4o-mini')
    """
    # Read the task spec
    spec_path = Path(taskspec_path)
    if not spec_path.exists():
        raise FileNotFoundError(
            f"Task spec file {taskspec_path} not found"
        )
    with open(spec_path, "r") as spec_file:
        spec_content = spec_file.read()

    # Read the task summary yaml
    yaml_path = Path(summary_path) 
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Task summary file {summary_path} not found"
        )
    with open(yaml_path, "r") as yaml_file:
        summary_data = yaml.safe_load(yaml_file)

    # Extract files from Beginning Context section in task spec
    import re
    context_match = re.search(r'### Beginning Context\n\n(.*?)\n\n', spec_content, re.DOTALL)
    if not context_match:
        raise ValueError("Could not find Beginning Context section in task spec")
        
    # Extract and clean up file paths
    editable_files = []
    for line in context_match.group(1).split('\n'):
        # Remove leading/trailing whitespace and bullet points if present
        line = line.strip().lstrip('- ').strip('`')
        if line:
            # Convert ./ptycho/ paths to ../ptycho/ since we're in specs directory
            # TODO 1: parameterize this prefix 
            editable_files.append(line.replace("./", "../"))

    # Setup BIG THREE: context, prompt, and model

    # Files to be edited - from summary yaml
    context_editable = editable_files

    # No read-only files needed
    context_read_only = []

    # Use the task spec as the prompt
    prompt = spec_content

    # Initialize the AI model
    model = Model(
        architect_model,
        editor_model=editor_model,
        editor_edit_format="diff",
    )

    # Initialize the AI Coding Assistant
    coder = Coder.create(
        main_model=model,
        edit_format="architect",
        io=InputOutput(yes=True),
        fnames=context_editable,
        read_only_fnames=context_read_only,
        auto_commits=False,
        suggest_shell_commands=False,
    )

    print("PROMPT:")
    print(prompt)
    print("END PROMPT")

    # Run the code modification
    coder.run(prompt)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("taskspec", help="Path to task specification file (e.g. taskspec.md)")
    parser.add_argument("summary", help="Path to task summary file (tochange.yaml)")
    # TODO 2: separate params for architect and editor
    parser.add_argument("--architect-model", default="4o-mini", help="Model to use for architecture decisions (default: 4o-mini)")
    parser.add_argument("--editor-model", default="4o-mini", help="Model to use for code editing (default: 4o-mini)")
    args = parser.parse_args()

    run_taskspec(args.taskspec, args.summary, args.architect_model, args.editor_model)
