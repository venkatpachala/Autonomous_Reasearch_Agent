import os
from typing import Tuple, Optional
from loguru import logger
from src.config import settings
from src.gateway.model_registry import get_task_config, ModelConfig

class ModelRouter:
    def __init__(self):
        # Cache API key presence
        self.openai_key = os.environ.get("OPENAI_API_KEY")

    def route(self, task: str) -> Tuple[str, str, Optional[str]]:
        """
        Determine the appropriate provider and model for a task.
        If a cloud provider is selected but credentials are missing,
        it automatically routes to a local fallback (Ollama).
        
        Returns:
            Tuple[provider, model, fallback_model]
        """
        config: ModelConfig = get_task_config(task)
        
        provider = config.provider
        model = config.model_name
        fallback = config.fallback

        # Check API key availability for OpenAI
        if provider.lower() == "openai" and not self.openai_key:
            logger.warning(
                f"Task '{task}' requested OpenAI but OPENAI_API_KEY is not set. "
                f"Routing to local Ollama fallback instead."
            )
            provider = "Ollama"
            model = settings.default_model or "qwen2.5:7b"
            fallback = None

        return provider, model, fallback

router = ModelRouter()
