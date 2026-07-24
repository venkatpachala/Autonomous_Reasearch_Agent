import os
from typing import List, Optional
from loguru import logger
from src.gateway.provider_manager import OllamaProvider, OpenAIProvider
from src.config import settings

# Last-resort cap if a caller skips memory_manager split
MAX_EMBED_CHARS = 8000


def _safe_text(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return " "
    if len(t) > MAX_EMBED_CHARS:
        return t[:MAX_EMBED_CHARS]
    return t


class EmbeddingsGateway:
    """
    Unified embedding gateway.

    Rules:
      - If OPENAI_API_KEY is set → always use OpenAI (target_dim)
      - If OpenAI fails → raise (no Ollama 768 fallback into a 1024 index)
      - Ollama only when no OpenAI key
    """

    def __init__(self):
        self.ollama = OllamaProvider()
        self.openai = OpenAIProvider()
        self.target_dim = settings.pinecone_embedding_dim

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
        provider: Optional[str] = None,
    ) -> List[float]:
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        chosen_provider = (provider or ("openai" if has_openai else "ollama")).lower()
        chosen_model = model or (
            "text-embedding-3-small"
            if chosen_provider == "openai"
            else "nomic-embed-text"
        )
        text = _safe_text(text)

        try:
            if chosen_provider == "openai":
                return await self.openai.embed(
                    chosen_model, text, dimensions=self.target_dim
                )
            return await self.ollama.embed(chosen_model, text)
        except Exception as e:
            logger.error(f"Embedding via {chosen_provider}/{chosen_model} failed: {e}")
            if chosen_provider == "openai":
                raise RuntimeError(
                    f"OpenAI embedding failed: {e}\n"
                    "Refusing to fall back to Ollama (dim mismatch risk)."
                ) from e
            raise

    async def embed_batch(
        self,
        texts: List[str],
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[List[float]]:
        """Batch embed. OpenAI uses one request per BATCH; else sequential."""
        if not texts:
            return []

        dim = self.target_dim
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        use_openai = (provider or "").lower() == "openai" or (
            provider is None and has_openai
        )
        BATCH = 64
        results: List[Optional[List[float]]] = [None] * len(texts)

        # Normalize inputs once
        safe_texts = [_safe_text(t) for t in texts]

        try:
            if use_openai:
                chosen_model = model or "text-embedding-3-small"
                for start in range(0, len(safe_texts), BATCH):
                    batch = safe_texts[start : start + BATCH]
                    if hasattr(self.openai, "embed_batch"):
                        vecs = await self.openai.embed_batch(
                            chosen_model, batch, dimensions=dim
                        )
                    else:
                        vecs = [
                            await self.openai.embed(
                                chosen_model, t, dimensions=dim
                            )
                            for t in batch
                        ]
                    for j, v in enumerate(vecs):
                        results[start + j] = v
            else:
                for i, t in enumerate(safe_texts):
                    results[i] = await self.embed(t, model=model, provider="ollama")
        except Exception as e:
            logger.error(f"Batch embed failed ({e}); falling back per-text")
            for i, t in enumerate(safe_texts):
                if results[i] is None:
                    try:
                        results[i] = await self.embed(
                            t, model=model, provider=provider
                        )
                    except Exception as inner:
                        logger.error(f"Embed failed for text[{i}]: {inner}")
                        results[i] = [0.0] * dim

        return [r if r is not None else [0.0] * dim for r in results]


embeddings_gateway = EmbeddingsGateway()