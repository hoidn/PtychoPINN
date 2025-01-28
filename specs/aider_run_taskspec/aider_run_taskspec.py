from pathlib import Path
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import yaml
import sys

def run_taskspec(taskspec_path: str, summary_path: str, model_name: str = '4o-mini'):
    """
    Process files according to the task specification.

    Args:
        taskspec_path (str): Path to task specification file (e.g. taskspec.md)
        summary_path (str): Path to task summary file (tochange.yaml)
        model_name (str): Name of the model to use (default: '4o-mini')
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

    # Get list of files to edit from the summary
    editable_files = [file["path"].replace("./", "../") for file in summary_data["Files_Requiring_Updates"]]

    # Setup BIG THREE: context, prompt, and model

    # Files to be edited - from summary yaml
    context_editable = editable_files

    # No read-only files needed
    context_read_only = []

    # Use the task spec as the prompt
    prompt = spec_content

    # Initialize the AI model
    model = Model(
        model_name,
        editor_model=model_name,
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
    parser.add_argument("--model", default="4o-mini", help="Model to use (default: 4o-mini)")
    args = parser.parse_args()

    run_taskspec(args.taskspec, args.summary, args.model)
