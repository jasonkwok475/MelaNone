import sys
from enum import Enum
import functools

class ErrorType(Enum):
    SERIAL = 0
    CAMERA = 1
    IMAGE_SAVE = 2
    MESH = 3
    ANALYSIS = 4

def catch_exceptions(error_type: ErrorType):
    """A decorator to catch and log exceptions from a function."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_type == ErrorType.SERIAL:
                    print(f"Serial connection failed: {e}. Using camera only.")
                elif error_type == ErrorType.CAMERA:
                    print(f"Warning: Camera capture failed: {e}")
                elif error_type == ErrorType.IMAGE_SAVE:
                    print(f"Failed to save image: {e}")
                elif error_type == ErrorType.MESH:
                    print(f"Warning: Mesh reconstruction failed: {e}")
                elif error_type == ErrorType.ANALYSIS:
                    print(f"Warning: Analysis failed: {e}")

                print(f"[bold red]An exception occurred in function '{func.__name__}'")
                print(e)
                sys.exit(1)
        return wrapper
    return decorator