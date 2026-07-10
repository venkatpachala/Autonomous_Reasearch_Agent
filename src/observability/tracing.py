"""
Tracing helpers - make it easy to add LangSmith traces to our agents
"""

from functools import wraps
from typing import Callable, Any
from loguru import logger

from src.observability.langsmith_config import setup_langsmith


def ensure_tracing(project_name: str = "research-agent"):
    """Call this once at application startup."""
    return setup_langsmith(project_name=project_name)


# Convenience decorator (works with both sync and async)
def traced(name: str = None, run_type: str = "chain"):
    """
    Decorator that adds LangSmith tracing if available.
    Falls back gracefully if LangSmith is not configured.
    """
    def decorator(func: Callable):
        try:
            from langsmith import traceable
            return traceable(name=name or func.__name__, run_type=run_type)(func)
        except ImportError:
            # langsmith not installed → no-op
            return func
        except Exception as e:
            logger.debug(f"Could not apply tracing to {func.__name__}: {e}")
            return func
    return decorator

