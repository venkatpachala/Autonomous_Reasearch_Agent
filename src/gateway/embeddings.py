import os
from typing import List, Optional
from loguru import logger
from src.gateway.provider_manager import OllamaProvider, OpenAIProvider
from src.config import settings


class EmbeddingsGateway:
    """
    Unified embedding gateway.
    
    Rules:
      - If OPENAI_API_KEY is set → always use OpenAI (1024-dim)
      - If OpenAI fails → raise error (do NOT fall back to 768-dim Ollama)
      - Only use Ollama when no OpenAI key is present AND the index is 768-dim
    """

    def __init__(self):
        self.ollama = OllamaProvider()
        self.openai = OpenAIProvider()
        self.target_dim = settings.pinecone_embedding_dim  # should be 1024

        if os.environ.get("OPENAI_API_KEY"):
            logger.info(
                f"Embeddings: Using OpenAI text-embedding-3-small "
                f"(dimensions={self.target_dim}) → Pinecone index"
            )
        else:
            logger.warning(
                "OPENAI_API_KEY not set. Will use Ollama nomic-embed-text (768 dims). "
                "Make sure your Pinecone index is also 768-dimensional."
            )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> List[float]:
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))

        # Decide which provider to use
        if provider:
            chosen_provider = provider.lower()
        elif has_openai:
            chosen_provider = "openai"
        else:
            chosen_provider = "ollama"

        chosen_model = model or (
            "text-embedding-3-small" if chosen_provider == "openai" else "nomic-embed-text"
        )

        try:
            if chosen_provider == "openai":
                return await self.openai.embed(
                    chosen_model,
                    text,
                    dimensions=self.target_dim
                )
            else:
                # Only allowed when no OpenAI key is present
                return await self.ollama.embed(chosen_model, text)

        except Exception as e:
            logger.error(f"Embedding via {chosen_provider}/{chosen_model} failed: {e}")

            # CRITICAL: Never fall back from OpenAI (1024) → Ollama (768)
            if chosen_provider == "openai":
                raise RuntimeError(
                    f"OpenAI embedding failed: {e}\n"
                    "Refusing to fall back to Ollama (768-dim) because the Pinecone index "
                    f"is configured for {self.target_dim} dimensions.\n"
                    "Fix the OpenAI API key / quota, or recreate the index with matching dimensions."
                ) from e

            # If we were already using Ollama and it failed, just raise
            raise e


embeddings_gateway = EmbeddingsGateway()