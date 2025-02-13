import json
from jinja2 import Environment, BaseLoader, StrictUndefined
from openai import OpenAI

class DependencyDetectorError(Exception):
    """Custom exception for dependency detector errors."""
    pass

def generate_required_files(task: str, context_file: str, prompt_template: str = None) -> list:
    """
    Generate a list of file paths that the work described by `task` will likely require,
    using the provided context file content for extra background.
    
    Args:
        task: A string describing the work to be done.
        context_file: The full content of a context file.
        prompt_template: Optional Jinja2 template to customize the LLM prompt.
    
    Returns:
        A list of file path strings parsed from the LLM's JSON output.
    
    Raises:
        DependencyDetectorError: If the LLM call fails or the output cannot be parsed.
    """
    if prompt_template is None:
        prompt_template = (
            "You are an expert developer. The provided context file content consists of multiple concatenated files. "
            "Each file is delineated with XML-like tags in the following format:\n\n"
            "<file path=\"FILE_PATH\" project=\"PROJECT_NAME\">\n"
            "FILE CONTENTS\n"
            "</file>\n\n"
            "Based on the above context, and the task described below, determine which file paths in the codebase will need "
            "to be present or modified to complete the task.\n\n"
            "== Context File ==\n"
            "{{ context_file }}\n\n"
            "== Task Description ==\n"
            "{{ task }}\n\n"
            "Return only a valid JSON object with two keys: \"editable\" and \"read_only\". "
            "\"editable\" is an array of file paths (strings) that can be modified, and \"read_only\" is an array of file paths (strings) that cannot be modified. "
            "Do not include any extra commentary."
        )
    
    env = Environment(loader=BaseLoader(), undefined=StrictUndefined, autoescape=False)
    template = env.from_string(prompt_template)
    prompt = template.render(task=task, context_file=context_file)
    
    openai_client = OpenAI()
    try:
        response = openai_client.chat.completions.create(
            model="o3-mini",  # Adjust the model as needed.
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as e:
        raise DependencyDetectorError(f"LLM call failed: {e}")
    
    llm_output = response.choices[0].message.content.strip()
    
    if "```" in llm_output:
        # Strip markdown formatting if present.
        content = llm_output.split("```", 2)[1].strip()
        # Remove any language specifier (e.g., "json") at the beginning.
        if content.lower().startswith("json"):
            # Split into lines and remove the first line.
            lines = content.splitlines()
            if len(lines) > 1:
                content = "\n".join(lines[1:]).strip()
            else:
                content = ""
        llm_output = content
    
    try:
        files_config = json.loads(llm_output)
    except Exception as e:
        raise DependencyDetectorError(f"Failed to parse JSON from LLM output: {llm_output}. Error: {e}")

    # For backward compatibility: if the response is a list, wrap it in a dict using an empty "read_only" list.
    if isinstance(files_config, list):
        files_config = {"editable": files_config, "read_only": []}
    elif not isinstance(files_config, dict):
        raise DependencyDetectorError("Output JSON is not an object with keys 'editable' and 'read_only' as expected.")

    if "editable" not in files_config or "read_only" not in files_config:
        raise DependencyDetectorError("JSON output must contain both 'editable' and 'read_only' keys.")

    if not isinstance(files_config["editable"], list) or not isinstance(files_config["read_only"], list):
        raise DependencyDetectorError("Values for 'editable' and 'read_only' must be lists.")

    return files_config

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generate a list of dependency file paths based on a task and a context file."
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="A string describing the work to be done."
    )
    parser.add_argument(
        "--context-file",
        type=str,
        required=True,
        help="Path to a file containing the full content of a context file."
    )
    parser.add_argument(
        "--prompt-template",
        type=str,
        help="Optional path to a file containing a custom Jinja2 prompt template."
    )
    
    args = parser.parse_args()
    
    try:
        with open(args.context_file, "r") as f:
            context_content = f.read()
    except Exception as e:
        print(f"Error reading context file: {e}")
        exit(1)
    
    custom_template = None
    if args.prompt_template:
        try:
            with open(args.prompt_template, "r") as f:
                custom_template = f.read()
        except Exception as e:
            print(f"Error reading prompt template file: {e}")
            exit(1)
    
    try:
        file_dependencies = generate_required_files(args.task, context_content, custom_template)
        print(json.dumps(file_dependencies))
    except DependencyDetectorError as e:
        print(f"Error: {e}")
        exit(1)
