"""
Pinecone Vector DB Client
==========================
Drop-in replacement for chroma_client.py.
Public interface is identical — callers do not need to change.

Key differences vs ChromaDB:
- Explicit embedding via OpenAI text-embedding-3-small (1536 dims)
- Cloud-hosted, persistent, production-grade
- Metadata filter syntax: {"topic": {"$eq": "..."}} instead of {"topic": "..."}
- query() returns same dict structure as ChromaDB for compatibility
"""

import os
import asyncio
from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("pinecone-client not installed. Run: pip install pinecone-client")

from src.config import settings


def _get_embedding(text: str) -> List[float]:
    """
    Generate embedding using OpenAI text-embedding-3-small via the gateway embedding module.
    Falls back to a zero vector if unavailable.
    """
    try:
        from src.gateway.embeddings import embeddings_gateway
        import asyncio
        # Handle both sync and async contexts
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We're inside an async context — use a thread pool
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, embeddings_gateway.embed(text))
                    return future.result()
            else:
                return loop.run_until_complete(embeddings_gateway.embed(text))
        except RuntimeError:
            return asyncio.run(embeddings_gateway.embed(text))
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}. Using zero vector.")
        return [0.0] * settings.pinecone_embedding_dim


class PineconeVectorClient:
    """
    Production Pinecone client with the same interface as ChromaClient.
    
    Chroma compatibility mapping:
      add_knowledge_note(note_id, document, metadata, embedding=None) → upsert
      query(query_text, n_results, where=None) → returns same dict structure as Chroma
      get_collection_stats() → {"count": N, "name": index_name}
    """

    def __init__(self, index_name: Optional[str] = None):
        self.index_name = index_name or settings.pinecone_index_name
        self.embedding_dim = settings.pinecone_embedding_dim
        self._index = None
        self._pc = None
        self._connected = False

        if not PINECONE_AVAILABLE:
            logger.error("Pinecone not installed. Vector operations will be disabled.")
            return

        api_key = settings.pinecone_api_key
        if not api_key:
            logger.error(
                "PINECONE_API_KEY not set in .env. "
                "Add it and set PINECONE_INDEX_NAME to enable Pinecone vector storage."
            )
            return

        try:
            self._pc = Pinecone(api_key=api_key)

            # Create index if it doesn't exist
            existing = [idx.name for idx in self._pc.list_indexes()]
            if self.index_name not in existing:
                logger.info(f"Creating Pinecone index '{self.index_name}' (dim={self.embedding_dim}, metric=cosine)...")
                self._pc.create_index(
                    name=self.index_name,
                    dimension=self.embedding_dim,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud=settings.pinecone_cloud,
                        region=settings.pinecone_region
                    )
                )
                logger.success(f"Pinecone index '{self.index_name}' created.")
            else:
                logger.info(f"Pinecone index '{self.index_name}' already exists.")

            self._index = self._pc.Index(self.index_name)
            self._connected = True
            stats = self._index.describe_index_stats()
            logger.success(
                f"Pinecone connected: index='{self.index_name}', "
                f"vectors={stats.get('total_vector_count', 0)}, "
                f"dim={self.embedding_dim}"
            )

        except Exception as e:
            logger.error(f"Pinecone connection failed: {e}")
            self._connected = False

    def is_connected(self) -> bool:
        return self._connected and self._index is not None

    def add_knowledge_note(
        self,
        note_id: str,
        document: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ):
        """Upsert a knowledge note into Pinecone. Generates embedding if not provided."""
        if not self.is_connected():
            logger.warning(f"Pinecone not connected. Skipping upsert for {note_id}.")
            return

        try:
            # Generate embedding if not supplied
            vector = embedding if embedding else _get_embedding(document)

            # Pinecone metadata values must be str/int/float/bool/list[str]
            # Sanitize metadata: coerce to safe types
            safe_meta = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_meta[k] = v
                elif isinstance(v, list):
                    safe_meta[k] = [str(i) for i in v]
                else:
                    safe_meta[k] = str(v)

            # Store document text in metadata for retrieval
            safe_meta["_document"] = document[:1000]  # Pinecone metadata limit

            self._index.upsert(vectors=[{
                "id": note_id,
                "values": vector,
                "metadata": safe_meta
            }])
            logger.success(f"Stored note in Pinecone: {note_id}")

        except Exception as e:
            logger.error(f"Failed to store note {note_id} in Pinecone: {e}")
            raise

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Semantic search over Pinecone index.
        Returns same structure as ChromaDB for full compatibility with ResearchRetriever.
        """
        if not self.is_connected():
            logger.warning("Pinecone not connected. Returning empty results.")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        try:
            query_vector = _get_embedding(query_text)

            # Convert Chroma-style where filter to Pinecone filter
            pinecone_filter = None
            if where:
                pinecone_filter = {k: {"$eq": v} for k, v in where.items()}

            result = self._index.query(
                vector=query_vector,
                top_k=n_results,
                filter=pinecone_filter,
                include_metadata=True
            )

            # Convert Pinecone result format → ChromaDB-compatible format
            ids, documents, metadatas, distances = [], [], [], []
            for match in result.get("matches", []):
                ids.append(match["id"])
                meta = match.get("metadata", {})
                # Extract document text stored in metadata
                doc_text = meta.pop("_document", "")
                documents.append(doc_text)
                metadatas.append(meta)
                # Pinecone returns similarity score (1=identical). Convert to distance for Chroma compat.
                score = match.get("score", 0.0)
                distances.append(1.0 - score)  # distance = 1 - cosine_similarity

            return {
                "ids": [ids],
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances]
            }

        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.is_connected():
            return {"count": 0, "name": self.index_name, "connected": False}
        try:
            stats = self._index.describe_index_stats()
            return {
                "count": stats.get("total_vector_count", 0),
                "name": self.index_name,
                "connected": True,
                "dimension": self.embedding_dim
            }
        except Exception as e:
            return {"count": 0, "name": self.index_name, "error": str(e)}


# Global singleton — same name as chroma_client for drop-in compatibility
pinecone_client = PineconeVectorClient()
# Alias so any code using chroma_client can import this directly
chroma_client = pinecone_client
