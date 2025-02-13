from jinja2 import Environment, BaseLoader, StrictUndefined
from typing import Dict, Any
from .director import TemplateError, TemplateSyntaxError, TemplateValueError

def get_jinja_env() -> Environment:
    return Environment(loader=BaseLoader(), undefined=StrictUndefined, autoescape=False)

def validate_template(jinja_env: Environment, template_str: str) -> None:
    try:
        jinja_env.parse(template_str)
    except Exception as e:
        raise TemplateSyntaxError(f"Invalid template syntax: {str(e)}")

def render_template(jinja_env: Environment, template_str: str, values: Dict[str, Any]) -> str:
    try:
        template = jinja_env.from_string(template_str)
        return template.render(**values)
    except Exception as e:
        raise TemplateValueError(f"Template rendering failed: {str(e)}")

def process_config_templates(director_instance) -> None:
    """
    Merge the template values from config and CLI, validate then render the prompt.
    Updates director_instance.config.prompt.
    """
    template_values = {}
    if director_instance.config.template_values:
        template_values.update(director_instance.config.template_values)
    if director_instance.template_values:
        template_values.update(director_instance.template_values)
    
    if template_values:
        jinja_env = get_jinja_env()
        validate_template(jinja_env, director_instance.config.prompt)
        rendered_prompt = render_template(jinja_env, director_instance.config.prompt, template_values)
        director_instance.file_log(f"Populated prompt: {rendered_prompt}", print_message=True)
        director_instance.config.prompt = rendered_prompt
