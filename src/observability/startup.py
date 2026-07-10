"""
Call this at the beginning of any entrypoint (chat.py, app.py, monitor.py, etc.)
"""

from src.observability.langsmith_config import setup_langsmith
from loguru import logger


def init_observability(project_name: str = "research-agent"):
    """
    Initialize LangSmith tracing.
    Safe to call multiple times.
    """
    enabled = setup_langsmith(project_name=project_name)
    if enabled:
        logger.info("Observability initialized with LangSmith")
    else:
        logger.info("Running without LangSmith tracing (set LANGCHAIN_API_KEY to enable)")
    return enabled
