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
            "You are an expert developer. Based on the following context file content and task, determine what "
            "file paths in the codebase will need to be present or modified to complete the task.\n\n"
            "== Context File ==\n"
            "{{ context_file }}\n\n"
            "== Task Description ==\n"
            "{{ task }}\n\n"
            "Return only a valid JSON array of file paths (strings), with no extra commentary."
        )
    
    env = Environment(loader=BaseLoader(), undefined=StrictUndefined, autoescape=False)
    template = env.from_string(prompt_template)
    prompt = template.render(task=task, context_file=context_file)
    
    openai_client = OpenAI()
    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o",  # Adjust the model as needed.
            messages=[
                {"role": "user", "content": prompt},
            ]
        )
    except Exception as e:
        raise DependencyDetectorError(f"LLM call failed: {e}")
    
    llm_output = response.choices[0].message.content.strip()
    
    if "```" in llm_output:
        # Strip markdown formatting if present.
        llm_output = llm_output.split("```", 2)[1].strip()
    
    try:
        file_list = json.loads(llm_output)
    except Exception as e:
        raise DependencyDetectorError(f"Failed to parse JSON from LLM output: {llm_output}. Error: {e}")
    
    if not isinstance(file_list, list):
        raise DependencyDetectorError("Output JSON is not a list as expected.")
    
    return file_list

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
