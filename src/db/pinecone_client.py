"""
Pinecone Vector DB Client
==========================
Fully async embedding + chunk-ready storage.
Fixed: No more asyncio.run() inside running event loop.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("pinecone-client not installed. Run: pip install pinecone-client")

from src.config import settings


async def _get_embedding(text: str) -> List[float]:
    """
    Fully async embedding generation.
    NEVER use asyncio.run() here.
    """
    try:
        from src.gateway.embeddings import embeddings_gateway
        return await embeddings_gateway.embed(text)
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}. Using zero vector.")
        return [0.0] * settings.pinecone_embedding_dim


class PineconeVectorClient:
    """
    Production Pinecone client with async interface.
    Compatible with previous chroma_client API.
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
                "Add it and set PINECONE_INDEX_NAME to enable Pinecone."
            )
            return

        try:
            self._pc = Pinecone(api_key=api_key)

            existing = [idx.name for idx in self._pc.list_indexes()]
            if self.index_name not in existing:
                logger.info(
                    f"Creating Pinecone index '{self.index_name}' "
                    f"(dim={self.embedding_dim}, metric=cosine)..."
                )
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

    async def add_knowledge_note(
        self,
        note_id: str,
        document: str,
        metadata: Dict[str, Any],
        embedding: Optional[List[float]] = None
    ):
        """
        Async upsert of a document/chunk into Pinecone.
        Generates embedding asynchronously if not provided.
        """
        if not self.is_connected():
            logger.warning(f"Pinecone not connected. Skipping upsert for {note_id}.")
            return

        try:
            # Fully async embedding
            vector = embedding if embedding is not None else await _get_embedding(document)

            # Safety: never send pure zero vector
            if all(v == 0.0 for v in vector):
                logger.warning(f"Skipping zero vector for {note_id}")
                return

            # Sanitize metadata (Pinecone only accepts str/int/float/bool/list[str])
            safe_meta = {}
            for k, v in metadata.items():
                if isinstance(v, (str, int, float, bool)):
                    safe_meta[k] = v
                elif isinstance(v, list):
                    safe_meta[k] = [str(i) for i in v]
                else:
                    safe_meta[k] = str(v)

            # Store a truncated version of the document for retrieval
            safe_meta["_document"] = document[:1000]

            self._index.upsert(vectors=[{
                "id": note_id,
                "values": vector,
                "metadata": safe_meta
            }])
            logger.debug(f"Stored in Pinecone: {note_id}")

        except Exception as e:
            logger.error(f"Failed to store {note_id} in Pinecone: {e}")
            raise

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None
    ) -> Dict[str, Any]:
        if not self.is_connected():
            logger.warning("Pinecone not connected. Returning empty results.")
            return {"ids": [[]], "documents": [[]], "metadatas": [[]], "distances": [[]]}

        try:
            # Generate embedding
            import asyncio
            try:
                loop = asyncio.get_running_loop()
                # Already in async context
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, _get_embedding(query_text))
                    query_vector = future.result()
            except RuntimeError:
                query_vector = asyncio.run(_get_embedding(query_text))

            # Convert filter
            pinecone_filter = None
            if where:
                # Pinecone expects {"topic": {"$eq": "value"}}
                pinecone_filter = {}
                for k, v in where.items():
                    pinecone_filter[k] = {"$eq": v}

            result = self._index.query(
                vector=query_vector,
                top_k=n_results,
                filter=pinecone_filter,
                include_metadata=True
            )

            ids, documents, metadatas, distances = [], [], [], []
            for match in result.get("matches", []):
                ids.append(match["id"])
                meta = match.get("metadata", {})
                doc_text = meta.pop("_document", "") or meta.get("text", "")
                documents.append(doc_text)
                metadatas.append(meta)
                score = match.get("score", 0.0)
                distances.append(1.0 - score)

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


# Global singleton
pinecone_client = PineconeVectorClient()
chroma_client = pinecone_client  # backward compatibility