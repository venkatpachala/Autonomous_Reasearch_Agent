import os
from typing import List, Optional
from loguru import logger
from src.gateway.provider_manager import OllamaProvider, OpenAIProvider

class EmbeddingsGateway:
    """
    Unified entry point for generating document and query embeddings.
    Handles fallbacks and provider routing.
    """
    def __init__(self):
        self.ollama = OllamaProvider()
        self.openai = OpenAIProvider()

    async def embed(self, text: str, model: Optional[str] = None, provider: Optional[str] = None) -> List[float]:
        """
        Generate embedding vector for a given text input.
        """
        chosen_provider = provider or "Ollama"
        chosen_model = model or "nomic-embed-text"

        # Safe key check
        if chosen_provider.lower() == "openai" and not os.environ.get("OPENAI_API_KEY"):
            logger.warning(
                "OpenAI embeddings requested but OPENAI_API_KEY is missing. "
                "Falling back to local Ollama with model 'nomic-embed-text'."
            )
            chosen_provider = "Ollama"
            chosen_model = "nomic-embed-text"

        try:
            if chosen_provider.lower() == "openai":
                return await self.openai.embed(chosen_model, text)
            else:
                return await self.ollama.embed(chosen_model, text)
        except Exception as e:
            logger.error(f"Embedding generation failed via {chosen_provider}/{chosen_model}: {e}")
            
            # Fallback loop
            if chosen_provider.lower() == "openai":
                logger.info("Attempting local Ollama fallback for embeddings...")
                try:
                    return await self.ollama.embed("nomic-embed-text", text)
                except Exception as inner_err:
                    logger.error(f"Fallback local embedding generation failed: {inner_err}")
                    raise inner_err
            raise e

embeddings_gateway = EmbeddingsGateway()
