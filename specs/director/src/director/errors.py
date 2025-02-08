class TemplateError(Exception):
    """Base class for template-related errors"""
    pass

class TemplateSyntaxError(TemplateError):
    """Raised when template syntax is invalid"""
    pass

class TemplateValueError(TemplateError):
    """Raised when template values are invalid or missing"""
    pass
