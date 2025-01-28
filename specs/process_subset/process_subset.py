from pathlib import Path
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import sys
import json
import yaml

def process_subset(description: str, answers_file: str = None):
    """
    Process files listed in edit_paths.json according to the description.

    Args:
        description (str): Description of the changes to make.
        answers_file (str, optional): Path to file containing answers to questions.
    """
    # Read the spec from process_subset.md
    spec_path = Path.cwd() / "process_subset.md"
    if not spec_path.exists():
        raise FileNotFoundError(
            "process_subset.md not found in current directory - please make sure it exists"
        )
    with open(spec_path, "r") as spec_file:
        spec_content = spec_file.read()

    # Load and format Q&A if provided
    questions_text = ""
    if answers_file:
        # TODO get the changes from tochange.yaml instead 
        # Load questions
        with open("questions.json", "r") as qf:
            questions = json.load(qf)["questions"]
        
        # Load answers
        with open(answers_file, "r") as af:
            answers = [line.strip() for line in af.readlines() if line.strip()]
        
        # Format Q&A pairs
        qa_pairs = []
        for q, a in zip(questions, answers):
            qa_pairs.extend([f"Q: {q}", f"A: {a}"])
        questions_text = "\n".join(qa_pairs)

    # Include both description and questions in the spec prompt
    spec_prompt = spec_content.replace("<description>", description)
    spec_prompt = spec_prompt.replace("<questions>", questions_text)

    # Read the list of files to process from tochange.yaml
    yaml_path = Path.cwd() / "tochange.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            "tochange.yaml not found in current directory - please make sure it exists"
        )
    with open(yaml_path, "r") as yaml_file:
        tochange_data = yaml.safe_load(yaml_file)
        file_list = [item["path"] for item in tochange_data["Files_Requiring_Updates"]]

    def format_files_section(files_data):
        """Format the Files_Requiring_Updates section for the prompt."""
        sections = []
        for file in files_data:
            sections.append(f"""
File: {file['path']}
Reason: {file['reason']}
Changes Needed:
{chr(10).join('- ' + change for change in file['spec_of_changes'])}
Dependencies: {', '.join(file['dependencies_affected'])}
""")
        return "\n".join(sections)

    # Extract sections from tochange.yaml
    files_section = format_files_section(tochange_data["Files_Requiring_Updates"])
    arch_impact = tochange_data["Architectural_Impact_Assessment"]["description"]

    # Setup BIG THREE: context, prompt, and model

    # Files to be edited - use the list from edit_paths.json
    context_editable = file_list

    # No read-only files needed
    context_read_only = []

    # Define the prompt for the AI model
    prompt = f"""
{spec_prompt}

<architectural_impact>
{arch_impact}
</architectural_impact>

<files_to_modify>
{files_section}
</files_to_modify>
"""

    # Initialize the AI model
    model = Model(
        "o1-preview",
        editor_model="o1-preview",
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
    parser.add_argument("description", help="Description of the changes to make")
    parser.add_argument("--answers", help="Optional path to file containing answers to questions")
    args = parser.parse_args()

    process_subset(args.description, args.answers)
