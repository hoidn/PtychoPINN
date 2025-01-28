from pathlib import Path
import sys
import json
import yaml
from typing import Dict, Any
import subprocess
import tempfile
import os
from datetime import datetime
import logging

logging.basicConfig(
    filename='process_subset.log',
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_config(config_path: str | Path) -> Dict[str, Any]:
    """
    Load and validate yaml configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Dict containing description and other config fields
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file {config_path} not found")
        
    with open(path, "r") as f:
        config = yaml.safe_load(f)
        
    if "description" not in config:
        raise ValueError("Config file must contain a 'description' field")
        
    return config

# TODO: this is either not returning the correct files or not getting incorporated
# intot the full prompt. help me debug.
def process_subset(description: str, answers_file: str = None):
    """
    Process files according to the description using raw LLM access.

    Args:
        description (str): Description of the changes to make.
        answers_file (str, optional): Path to file containing answers to questions.
    """
    # Read the spec from process_subset.md
    spec_path = Path(__file__).parent / "process_subset.md"
    if not spec_path.exists():
        raise FileNotFoundError(
            f"process_subset.md not found in {Path(__file__).parent} - please make sure it exists"
        )
    with open(spec_path, "r") as spec_file:
        spec_content = spec_file.read()

    # Load and format Q&A if provided
    questions_text = ""
    if answers_file:
        # Load questions from tochange.yaml
        yaml_path = Path.cwd() / "tochange.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                "tochange.yaml not found in current directory - please make sure it exists"
            )
        with open(yaml_path, "r") as yaml_file:
            tochange_data = yaml.safe_load(yaml_file)
            questions = tochange_data["Questions_for_Clarification"]
        
        # Load answers
        with open(answers_file, "r") as af:
            answers = [line.strip() for line in af.readlines() if line.strip()]
        
        # Format Q&A pairs
        qa_pairs = []
        for q, a in zip(questions, answers):
            qa_pairs.extend([f"Q: {q}", f"A: {a}"])
        questions_text = "\n".join(qa_pairs)

    # Read the spec prompt guide
    guide_path = Path(__file__).parent.parent / "spec_prompt_guide.xml"
    if not guide_path.exists():
        raise FileNotFoundError(
            f"spec_prompt_guide.xml not found in {guide_path} - please make sure it exists"
        )
    with open(guide_path, "r") as guide_file:
        guide_content = guide_file.read()

    # Include description, questions and spec prompt guide in the spec prompt
    spec_prompt = spec_content.replace("<description>", description)
    spec_prompt = spec_prompt.replace("<questions>", questions_text)
    spec_prompt = spec_prompt.replace("<spec_prompt_guide>", guide_content)

    # Read tochange.yaml for files info
    yaml_path = Path.cwd() / "tochange.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(
            "tochange.yaml not found in current directory - please make sure it exists"
        )
    with open(yaml_path, "r") as yaml_file:
        tochange_data = yaml.safe_load(yaml_file)

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

    def get_file_contents(file_paths):
        """Read and format contents of the specified files."""
        contents = []
        print(f"\nProcessing files: {[f['path'] for f in file_paths]}")
        logging.debug(f"Processing files: {[f['path'] for f in file_paths]}")
        
        for file_path in file_paths:
            # Convert ./ptycho/ to ../ptycho/ since we're in specs directory
            # TODO: parameterize this prefix
            path = Path(file_path['path'].replace('./ptycho/', '../ptycho/'))
            print(f"Attempting to read: {path}")
            logging.debug(f"Attempting to read: {path}")
            
            try:
                with open(path, 'r') as f:
                    content = f.read()
                    formatted = f"=== {path} ===\n{content}\n"
                    contents.append(formatted)
                    print(f"Successfully read {path} ({len(content)} bytes)")
                    logging.debug(f"Successfully read {path} ({len(content)} bytes)")
            except FileNotFoundError:
                print(f"ERROR: File not found: {path}")
                logging.error(f"File not found: {path}")
            except Exception as e:
                print(f"ERROR: Error reading {path}: {str(e)}")
                logging.error(f"Error reading {path}: {str(e)}")
        
        result = "\n".join(contents)
        print(f"Total content length: {len(result)} bytes")
        logging.debug(f"Total content length: {len(result)} bytes")
        return result

    # Log section sizes before constructing prompt
    logging.debug("Building full prompt with sections:")
    logging.debug(f"Spec prompt length: {len(spec_prompt)}")
    logging.debug(f"Arch impact length: {len(arch_impact)}")
    logging.debug(f"Files section length: {len(files_section)}")

    # Construct the full prompt
    full_prompt = f"""
{spec_prompt}

<architectural_impact>
{arch_impact}
</architectural_impact>

<context_files>
{get_file_contents(tochange_data["Files_Requiring_Updates"])}
</context_files>

<files_to_modify>
{files_section}
</files_to_modify>
"""

    # Save the full prompt to a debug file
    debug_file = Path.cwd() / "debug_prompt.txt"
    with open(debug_file, "w") as f:
        f.write(full_prompt)

    # Git add the debug file
    try:
        subprocess.run(["git", "add", debug_file], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Warning: Could not add debug file to git: {e}")

    # Create temp file and run llm command
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
        tmp.write(full_prompt)
        tmp_path = tmp.name

    try:
        with open(tmp_path, 'r') as input_file:
            result = subprocess.run(
                ["llm", "--model", "o1-mini"],
                stdin=input_file,
                capture_output=True,
                text=True,
                check=True
            )
        response = result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running llm command: {e}")
        return
    finally:
        os.unlink(tmp_path)

    # Write response to taskspec.md
    output_file = "taskspec.md"
    
    # Extract content between ```md ``` markers
    import re
    md_match = re.search(r'```md\s*([\s\S]*?)\s*```(?:\s*$|\n|$)', response, re.DOTALL)
    if md_match:
        md_content = md_match.group(1).strip()
    else:
        print("Error: Could not find ```md ``` section in LLM response")
        print("Writing full response instead")
        md_content = response

    with open(output_file, "w") as f:
        f.write(md_content)

    # Git commit the new file
    try:
        subprocess.run(["git", "add", output_file], check=True)
        commit_msg = f"Add AI-generated task spec from {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        subprocess.run(["git", "commit", "-m", commit_msg], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during git operations: {e}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("config_path", help="Path to YAML config file containing the description")
    parser.add_argument("--answers", help="Optional path to file containing answers to questions")
    args = parser.parse_args()
    
    # Load config containing description
    config = load_config(args.config_path)
    
    process_subset(config["description"], args.answers)
