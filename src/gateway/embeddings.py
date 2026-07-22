import os
from typing import List, Optional
from loguru import logger
from src.gateway.provider_manager import OllamaProvider, OpenAIProvider
from src.config import settings


class EmbeddingsGateway:
    """
    Unified entry point for generating document and query embeddings.
    
    Provider priority:
      1. OpenAI text-embedding-3-small (dimensions=PINECONE_EMBEDDING_DIM) — if OPENAI_API_KEY set
      2. Ollama nomic-embed-text (768 dims) — local fallback only
    
    IMPORTANT: Pinecone index dimension must match the embedding dimension.
    Set PINECONE_EMBEDDING_DIM in .env to match your Pinecone index.
    """
    def __init__(self):
        self.ollama = OllamaProvider()
        self.openai = OpenAIProvider()
        self.target_dim = settings.pinecone_embedding_dim

        # Log embedding strategy on startup
        if os.environ.get("OPENAI_API_KEY"):
            logger.info(
                f"Embeddings: Using OpenAI text-embedding-3-small "
                f"(dimensions={self.target_dim}) → Pinecone index"
            )
        else:
            logger.warning(
                "OPENAI_API_KEY not set. Embeddings will use Ollama nomic-embed-text (768 dims). "
                f"Set PINECONE_EMBEDDING_DIM=768 in .env to match, or provide OpenAI key for "
                f"{self.target_dim}-dim embeddings."
            )

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        provider: Optional[str] = None
    ) -> List[float]:
        """
        Generate embedding vector for a given text input.
        Defaults to OpenAI text-embedding-3-small if API key is available.
        Falls back to Ollama nomic-embed-text (768 dims) otherwise.
        """
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))

        # Determine provider and model
        if provider:
            chosen_provider = provider
            chosen_model = model or ("text-embedding-3-small" if provider.lower() == "openai" else "nomic-embed-text")
        elif has_openai:
            chosen_provider = "openai"
            chosen_model = model or "text-embedding-3-small"
        else:
            chosen_provider = "ollama"
            chosen_model = model or "nomic-embed-text"

        try:
            if chosen_provider.lower() == "openai":
                # Pass dimensions to OpenAI for truncated embeddings matching Pinecone index
                return await self.openai.embed(
                    chosen_model, text,
                    dimensions=self.target_dim
                )
            else:
                return await self.ollama.embed(chosen_model, text)

        except Exception as e:
            logger.error(f"Embedding via {chosen_provider}/{chosen_model} failed: {e}")

            # Fallback: if OpenAI failed, try Ollama
            if chosen_provider.lower() == "openai":
                logger.warning("Falling back to Ollama nomic-embed-text (768 dims). "
                               "NOTE: If Pinecone index is not 768 dims, upsert will fail.")
                try:
                    return await self.ollama.embed("nomic-embed-text", text)
                except Exception as inner_err:
                    logger.error(f"Ollama fallback embedding also failed: {inner_err}")
                    raise inner_err
            raise e


embeddings_gateway = EmbeddingsGateway()
