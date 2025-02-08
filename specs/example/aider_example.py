from pathlib import Path
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import sys
import json

def scriptname(description: str, answers_file: str = None):
    """
    Process files listed in edit_paths.json according to the description.

    Args:
        description (str): Description of the changes to make.
        answers_file (str, optional): Path to file containing answers to questions.
    """
    # Read the spec from scriptname.md
    spec_path = Path.cwd() / "scriptname.md"
    if not spec_path.exists():
        raise FileNotFoundError(
            "scriptname.md not found in current directory - please make sure it exists"
        )
    with open(spec_path, "r") as spec_file:
        spec_content = spec_file.read()

    # Load and format Q&A if provided
    questions_text = ""
    if answers_file:
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

    # Read the list of files to process from edit_paths.json
    json_path = Path.cwd() / "edit_paths.json"
    if not json_path.exists():
        raise FileNotFoundError(
            "edit_paths.json not found in current directory - please make sure it exists"
        )
    with open(json_path, "r") as json_file:
        file_list = json.load(json_file)["files"]

    # Setup BIG THREE: context, prompt, and model

    # Files to be edited - use the list from edit_paths.json
    context_editable = file_list

    # No read-only files needed
    context_read_only = []

    # Define the prompt for the AI model
    prompt = spec_prompt

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

    scriptname(args.description, args.answers)

