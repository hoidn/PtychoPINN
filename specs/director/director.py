from pydantic import BaseModel
from typing import Optional, List, Literal, Dict, Any
from pathlib import Path
from aider.coders import Coder
from aider.models import Model
from aider.io import InputOutput
import sys
import yaml
import argparse
from openai import OpenAI
import subprocess
from jinja2 import Environment, BaseLoader, Template, TemplateSyntaxError, StrictUndefined

class TemplateError(Exception):
    """Base class for template-related errors"""
    pass

class TemplateSyntaxError(TemplateError):
    """Raised when template syntax is invalid"""
    pass

class TemplateValueError(TemplateError):
    """Raised when template values are invalid or missing"""
    pass

class EvaluationResult(BaseModel):
    success: bool
    feedback: Optional[str]

class DirectorConfig(BaseModel):
    prompt: str
    coder_model: str
    evaluator_model: Literal["gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"]
    max_iterations: int
    execution_command: str
    context_editable: List[str]
    context_read_only: List[str]
    evaluator: Literal["default"]
    template_values: Optional[Dict[str, Any]] = None

class Director:
    """
    Self Directed AI Coding Assistant with template support
    """

    def __init__(self, config_path: str, template_values: Optional[Dict[str, Any]] = None):
        """
        Initialize Director with config file and optional template values.
        
        Args:
            config_path: Path to YAML config file
            template_values: Optional CLI-provided template values that override config values
        """
        self.template_values = template_values or {}
        
        # Initialize Jinja2 environment early
        self._jinja_env = Environment(
            loader=BaseLoader(),
            undefined=StrictUndefined,  # Raise errors on undefined variables
            autoescape=False  # Don't escape by default - prompts aren't HTML
        )
        
        # 1. Load and validate basic config structure
        self.config = self._load_and_validate_config(Path(config_path))
        
        # 2. Process templates if needed
        self._process_config_templates()
        
        # 3. Initialize OpenAI client
        self.llm_client = OpenAI()

    def _load_and_validate_config(self, config_path: Path) -> DirectorConfig:
        """
        Load and validate the basic config structure.
        No template processing - just YAML loading and validation.
        """
        # Validate file exists
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        # Load and parse YAML
        try:
            with open(config_path) as f:
                config_dict = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in config file: {str(e)}")

        # Handle .md file references
        if isinstance(config_dict.get("prompt"), str) and config_dict["prompt"].endswith(".md"):
            prompt_path = Path(config_dict["prompt"])
            if not prompt_path.exists():
                raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
            with open(prompt_path) as f:
                config_dict["prompt"] = f.read()

        # Create config object for basic validation
        config = DirectorConfig(**config_dict)

        # Validate evaluator model
        allowed_evaluator_models = {"gpt-4o", "gpt-4o-mini", "o1-mini", "o1-preview"}
        if config.evaluator_model not in allowed_evaluator_models:
            raise ValueError(
                f"evaluator_model must be one of {allowed_evaluator_models}, "
                f"got {config.evaluator_model}"
            )

        # Validate file paths
        if not config.context_editable:
            raise ValueError("At least one editable context file must be specified")

        for path in config.context_editable + config.context_read_only:
            if not Path(path).exists():
                raise FileNotFoundError(f"File not found: {path}")

        return config

    def _process_config_templates(self) -> None:
        """
        Process any templates in the config prompt.
        This runs after basic config validation but before the config is used.
        """
        # Merge template values (CLI overrides config)
        template_values = {}
        if self.config.template_values:
            template_values.update(self.config.template_values)
        if self.template_values:  # CLI values take precedence
            template_values.update(self.template_values)

        # Only process if we have template values
        if template_values:
            try:
                # First validate template syntax
                self._validate_template(self.config.prompt)
                # Then render with values
                rendered_prompt = self._render_template(self.config.prompt, template_values)
                # Update config with rendered prompt
                self.config.prompt = rendered_prompt
            except TemplateError as e:
                raise ValueError(f"Template processing failed: {str(e)}")

    def _validate_template(self, template_str: str) -> None:
        """Validate template syntax without rendering."""
        try:
            self._jinja_env.parse(template_str)
        except Exception as e:
            raise TemplateSyntaxError(f"Invalid template syntax: {str(e)}")

    def _render_template(self, template_str: str, values: Dict[str, Any]) -> str:
        """Render a template with the given values."""
        try:
            template = self._jinja_env.from_string(template_str)
            return template.render(**values)
        except Exception as e:
            raise TemplateValueError(f"Template rendering failed: {str(e)}")

    def parse_llm_json_response(self, response_str: str) -> str:
        """
        Parse and fix the response from an LLM that is expected to return JSON.
        
        Args:
            response_str: The raw response string from the LLM
            
        Returns:
            Cleaned JSON string
        """
        if "```" not in response_str:
            response_str = response_str.strip()
            self.file_log(f"raw pre-json-parse: {response_str}", print_message=False)
            return response_str

        # Remove opening backticks and language identifier
        response_str = response_str.split("```", 1)[-1].split("\n", 1)[-1]

        # Remove closing backticks
        response_str = response_str.rsplit("```", 1)[0]

        response_str = response_str.strip()

        self.file_log(f"post-json-parse: {response_str}", print_message=False)

        return response_str

    def file_log(self, message: str, print_message: bool = True) -> None:
        """
        Log a message to file and optionally print to console.
        
        Args:
            message: Message to log
            print_message: Whether to also print to console
        """
        if print_message:
            print(message)
        with open("director_log.txt", "a+") as f:
            f.write(message + "\n")

    def create_new_ai_coding_prompt(
        self,
        iteration: int,
        base_input_prompt: str,
        execution_output: str,
        evaluation: EvaluationResult,
    ) -> str:
        """
        Create a new prompt for the AI coder based on the current iteration.
        
        Args:
            iteration: Current iteration number
            base_input_prompt: Original user prompt
            execution_output: Output from the last execution
            evaluation: Evaluation result from the last attempt
            
        Returns:
            New prompt string for the AI
        """
        if iteration == 0:
            return base_input_prompt
        
        return f"""
# Generate the next iteration of code to achieve the user's desired result based on their original instructions and the feedback from the previous attempt.
> Generate a new prompt in the same style as the original instructions for the next iteration of code.

## This is your {iteration}th attempt to generate the code.
> You have {self.config.max_iterations - iteration} attempts remaining.

## Here's the user's original instructions for generating the code:
{base_input_prompt}

## Here's the output of your previous attempt:
{execution_output}

## Here's feedback on your previous attempt:
{evaluation.feedback}"""

    def ai_code(self, prompt: str) -> None:
        """
        Run the AI coder with the given prompt.
        
        Args:
            prompt: The prompt to send to the AI coder
        """
        model = Model(self.config.coder_model)
        coder = Coder.create(
            main_model=model,
            io=InputOutput(yes=True),
            fnames=self.config.context_editable,
            read_only_fnames=self.config.context_read_only,
            auto_commits=False,
            suggest_shell_commands=False,
        )
        coder.run(prompt)

    def execute(self) -> str:
        """
        Execute the tests and return the output.
        
        Returns:
            Output from execution
        """
        result = subprocess.run(
            self.config.execution_command,
            capture_output=True,
            text=True,
            shell=True,  # Add this line to execute the command in a shell
        )
        self.file_log(
            f"Execution output: \n{result.stdout + result.stderr}",
            print_message=False,
        )
        return result.stdout + result.stderr

    def evaluate(self, execution_output: str) -> EvaluationResult:
        """
        Evaluate the execution output and determine if it was successful.
        
        Args:
            execution_output: Output from the execution
            
        Returns:
            EvaluationResult with success status and feedback
            
        Raises:
            ValueError: If using unsupported evaluator
        """
        if self.config.evaluator != "default":
            raise ValueError(f"Custom evaluator {self.config.evaluator} not implemented")

        map_editable_fname_to_files = {
            Path(fname).name: Path(fname).read_text()
            for fname in self.config.context_editable
        }

        map_read_only_fname_to_files = {
            Path(fname).name: Path(fname).read_text()
            for fname in self.config.context_read_only
        }

        evaluation_prompt = f"""Evaluate this execution output and determine if it was successful based on the execution command, the user's desired result, the editable files, checklist, and the read-only files.

## Checklist:
- Is the execution output reporting success or failure?
- Did we miss any tasks? Review the User's Desired Result to see if we have satisfied all tasks.
- Did we satisfy the user's desired result?
- Ignore warnings

## User's Desired Result:
{self.config.prompt}

## Editable Files:
{map_editable_fname_to_files}

## Read-Only Files:
{map_read_only_fname_to_files}

## Execution Command:
{self.config.execution_command}
                                        
## Execution Output:
{execution_output}

## Response Format:
> Be 100% sure to output JSON.parse compatible JSON.
> That means no new lines.

Return a structured JSON response with the following structure: {{
    success: bool - true if the execution output generated by the execution command matches the Users Desired Result
    feedback: str | None - if unsuccessful, provide detailed feedback explaining what failed and how to fix it, or None if successful
}}"""

        self.file_log(
            f"Evaluation prompt: ({self.config.evaluator_model}):\n{evaluation_prompt}",
            print_message=False,
        )

        try:
            completion = self.llm_client.chat.completions.create(
                model=self.config.evaluator_model,
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt,
                    },
                ],
            )

            self.file_log(
                f"Evaluation response: ({self.config.evaluator_model}):\n{completion.choices[0].message.content}",
                print_message=False,
            )

            evaluation = EvaluationResult.model_validate_json(
                self.parse_llm_json_response(completion.choices[0].message.content)
            )

            return evaluation
        except Exception as e:
            self.file_log(
                f"Error evaluating execution output for '{self.config.evaluator_model}'. Error: {e}. Falling back to gpt-4o & structured output."
            )

            # Fallback
            completion = self.llm_client.beta.chat.completions.parse(
                model="gpt-4o",
                messages=[
                    {
                        "role": "user",
                        "content": evaluation_prompt,
                    },
                ],
                response_format=EvaluationResult,
            )

            message = completion.choices[0].message
            if message.parsed:
                return message.parsed
            else:
                raise ValueError("Failed to parse the response")

    def direct(self) -> None:
        """
        Main method to direct the AI coding process.
        Runs iterations until success or max iterations reached.
        """
        evaluation = EvaluationResult(success=False, feedback=None)
        execution_output = ""
        success = False

        for i in range(self.config.max_iterations):
            self.file_log(f"\nIteration {i+1}/{self.config.max_iterations}")

            self.file_log("🧠 Creating new prompt...")
            new_prompt = self.create_new_ai_coding_prompt(
                i, self.config.prompt, execution_output, evaluation
            )

            self.file_log("🤖 Generating AI code...")
            self.ai_code(new_prompt)

            self.file_log(f"💻 Executing code... '{self.config.execution_command}'")
            execution_output = self.execute()

            self.file_log(
                f"🔍 Evaluating results... '{self.config.evaluator_model}' + '{self.config.evaluator}'"
            )
            evaluation = self.evaluate(execution_output)

            self.file_log(
                f"🔍 Evaluation result: {'✅ Success' if evaluation.success else '❌ Failed'}"
            )
            if evaluation.feedback:
                self.file_log(f"💬 Feedback: \n{evaluation.feedback}")

            if evaluation.success:
                success = True
                self.file_log(
                    f"\n🎉 Success achieved after {i+1} iterations! Breaking out of iteration loop."
                )
                break
            else:
                self.file_log(
                    f"\n🔄 Continuing with next iteration... Have {self.config.max_iterations - i - 1} attempts remaining."
                )

        if not success:
            self.file_log(
                "\n🚫 Failed to achieve success within the maximum number of iterations."
            )

        self.file_log("\nDone.")

def main():
    """Entry point for the director script."""
    parser = argparse.ArgumentParser(
        description="Run the AI Coding Director with a config file and optional template values"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="specs/basic.yaml",
        help="Path to the YAML config file",
    )
    parser.add_argument(
        "--template-values",
        type=str,
        help="JSON string of template values to substitute",
    )
    
    args = parser.parse_args()
    
    template_values = None
    if args.template_values:
        import json
        try:
            template_values = json.loads(args.template_values)
            if not isinstance(template_values, dict):
                raise ValueError("Template values must be a JSON object")
        except json.JSONDecodeError as e:
            print(f"Error parsing template values JSON: {str(e)}")
            sys.exit(1)
        except ValueError as e:
            print(str(e))
            sys.exit(1)
            
    try:
        director = Director(args.config, template_values)
        director.direct()
    except FileNotFoundError as e:
        print(f"File not found: {str(e)}")
        sys.exit(1)
    except ValueError as e:
        print(f"Configuration error: {str(e)}")
        sys.exit(1)
    except TemplateError as e:
        print(f"Template error: {str(e)}")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

