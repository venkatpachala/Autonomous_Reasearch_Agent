"""
LangSmith Observability Configuration
=====================================
Central place to enable/disable tracing and configure LangSmith.
"""

import os
from typing import Optional
from loguru import logger

from src.config import settings


def setup_langsmith(
    project_name: str = "research-agent",
    enable: bool = True,
) -> bool:
    """
    Configure LangSmith environment variables and enable tracing.
    
    Returns True if LangSmith is successfully enabled.
    """
    if not enable:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        logger.info("LangSmith tracing disabled")
        return False

    # Prefer values from .env / settings if available
    api_key = os.getenv("LANGCHAIN_API_KEY") or getattr(settings, "langchain_api_key", None)
    endpoint = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    if not api_key:
        logger.warning(
            "LANGCHAIN_API_KEY not found. LangSmith tracing will be disabled.\n"
            "Get a free key at https://smith.langchain.com and add it to .env"
        )
        os.environ["LANGCHAIN_TRACING_V2"] = "false"
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ["LANGCHAIN_ENDPOINT"] = endpoint
    os.environ["LANGCHAIN_PROJECT"] = project_name

    logger.success(f"LangSmith tracing enabled → project: {project_name}")
    return True


def get_langsmith_client():
    """Return a LangSmith Client instance (or None if not configured)."""
    try:
        from langsmith import Client
        return Client()
    except Exception as e:
        logger.warning(f"Could not create LangSmith client: {e}")
        return None

