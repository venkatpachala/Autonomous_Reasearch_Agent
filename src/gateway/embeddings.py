"""
Unified embedding gateway with in-process LRU cache.
"""

import hashlib
import os
from collections import OrderedDict
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
      - Identical texts reuse a cached vector (LRU, process-local)
    """

    def __init__(self):
        self.ollama = OllamaProvider()
        self.openai = OpenAIProvider()
        self.target_dim = settings.pinecone_embedding_dim

        # Query / text embedding cache
        self._cache: "OrderedDict[str, List[float]]" = OrderedDict()
        self._cache_max = 256

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

    def _cache_key(
        self,
        text: str,
        model: Optional[str],
        provider: Optional[str],
    ) -> str:
        raw = f"{provider or ''}|{model or ''}|{self.target_dim}|{text}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[List[float]]:
        if key not in self._cache:
            return None
        self._cache.move_to_end(key)
        return self._cache[key]

    def _cache_put(self, key: str, vector: List[float]) -> None:
        self._cache[key] = vector
        self._cache.move_to_end(key)
        while len(self._cache) > self._cache_max:
            self._cache.popitem(last=False)

    async def embed(
        self,
        text: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> List[float]:
        text = _safe_text(text)
        key = self._cache_key(text, model, provider)
        hit = self._cache_get(key)
        if hit is not None:
            logger.debug("Embedding cache HIT")
            return hit

        vector = await self._embed_uncached(text, model=model, provider=provider)
        self._cache_put(key, vector)
        return vector

    async def _embed_uncached(
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
        """Batch embed with per-text cache. OpenAI batches uncached misses."""
        if not texts:
            return []

        dim = self.target_dim
        has_openai = bool(os.environ.get("OPENAI_API_KEY"))
        use_openai = (provider or "").lower() == "openai" or (
            provider is None and has_openai
        )
        BATCH = 64

        safe_texts = [_safe_text(t) for t in texts]
        results: List[Optional[List[float]]] = [None] * len(safe_texts)
        miss_indices: List[int] = []

        for i, t in enumerate(safe_texts):
            key = self._cache_key(t, model, provider)
            hit = self._cache_get(key)
            if hit is not None:
                results[i] = hit
            else:
                miss_indices.append(i)

        if not miss_indices:
            logger.debug(f"Embedding batch cache HIT all {len(safe_texts)}")
            return [r if r is not None else [0.0] * dim for r in results]

        logger.debug(
            f"Embedding batch: {len(safe_texts) - len(miss_indices)} hits, "
            f"{len(miss_indices)} misses"
        )

        try:
            if use_openai:
                chosen_model = model or "text-embedding-3-small"
                for start in range(0, len(miss_indices), BATCH):
                    batch_idxs = miss_indices[start : start + BATCH]
                    batch = [safe_texts[i] for i in batch_idxs]
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
                        idx = batch_idxs[j]
                        results[idx] = v
                        self._cache_put(
                            self._cache_key(safe_texts[idx], model, provider), v
                        )
            else:
                for i in miss_indices:
                    v = await self._embed_uncached(
                        safe_texts[i], model=model, provider="ollama"
                    )
                    results[i] = v
                    self._cache_put(
                        self._cache_key(safe_texts[i], model, provider), v
                    )
        except Exception as e:
            logger.error(f"Batch embed failed ({e}); falling back per-text")
            for i in miss_indices:
                if results[i] is not None:
                    continue
                try:
                    v = await self.embed(
                        safe_texts[i], model=model, provider=provider
                    )
                    results[i] = v
                except Exception as inner:
                    logger.error(f"Embed failed for text[{i}]: {inner}")
                    results[i] = [0.0] * dim

        return [r if r is not None else [0.0] * dim for r in results]


embeddings_gateway = EmbeddingsGateway()