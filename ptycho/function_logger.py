import functools
import logging

# Set up logging configuration
logging.basicConfig(
    filename='function_calls.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

def log_function_call(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logging.info(f"Called function: {func.__name__}")
        result = func(*args, **kwargs)
        return result
    return wrapper