"""
Pinecone Vector DB Client
==========================
Async embedding + chunk storage + batch upsert.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    logger.warning("pinecone-client not installed. Run: pip install pinecone")

from src.config import settings


async def _get_embedding(text: str) -> List[float]:
    try:
        from src.gateway.embeddings import embeddings_gateway
        return await embeddings_gateway.embed(text)
    except Exception as e:
        logger.warning(f"Embedding generation failed: {e}. Using zero vector.")
        return [0.0] * settings.pinecone_embedding_dim


def _sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    safe_meta: Dict[str, Any] = {}
    for k, v in (metadata or {}).items():
        if isinstance(v, (str, int, float, bool)):
            safe_meta[k] = v
        elif isinstance(v, list):
            safe_meta[k] = [str(i) for i in v]
        elif v is None:
            continue
        else:
            safe_meta[k] = str(v)
    return safe_meta


class PineconeVectorClient:
    def __init__(self, index_name: Optional[str] = None):
        self.index_name = index_name or settings.pinecone_index_name
        self.embedding_dim = settings.pinecone_embedding_dim
        self._index = None
        self._pc = None
        self._connected = False

        if not PINECONE_AVAILABLE:
            logger.error("Pinecone not installed. Vector operations disabled.")
            return

        api_key = settings.pinecone_api_key
        if not api_key:
            logger.error("PINECONE_API_KEY not set in .env.")
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
                        region=settings.pinecone_region,
                    ),
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
        embedding: Optional[List[float]] = None,
    ):
        """Single-vector upsert (legacy path). Prefer upsert_vectors for batches."""
        if not self.is_connected():
            logger.warning(f"Pinecone not connected. Skipping upsert for {note_id}.")
            return

        try:
            vector = embedding if embedding is not None else await _get_embedding(document)
            if not vector or all(float(v) == 0.0 for v in vector):
                logger.warning(f"Skipping zero vector for {note_id}")
                return

            safe_meta = _sanitize_metadata(metadata)
            safe_meta["_document"] = (document or "")[:35000]

            self._index.upsert(
                vectors=[{
                    "id": note_id,
                    "values": vector,
                    "metadata": safe_meta,
                }]
            )
            logger.debug(f"Stored in Pinecone: {note_id}")
        except Exception as e:
            logger.error(f"Failed to store {note_id} in Pinecone: {e}")
            raise

    async def upsert_vectors(
        self,
        items: List[Dict[str, Any]],
        batch_size: int = 100,
    ) -> int:
        """
        Batch upsert pre-embedded vectors.

        items: [{"id": str, "values": List[float], "metadata": dict}, ...]
        Returns number of vectors upserted.
        """
        if not self.is_connected() or not items:
            return 0

        stored = 0
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            clean = []
            for it in batch:
                vals = it.get("values") or []
                if not vals or all(float(x) == 0.0 for x in vals):
                    logger.warning(f"Skip zero vector id={it.get('id')}")
                    continue
                meta = _sanitize_metadata(it.get("metadata") or {})
                clean.append({
                    "id": it["id"],
                    "values": vals,
                    "metadata": meta,
                })
            if not clean:
                continue
            try:
                self._index.upsert(vectors=clean)
                stored += len(clean)
                logger.debug(f"Pinecone upserted batch of {len(clean)}")
            except Exception as e:
                logger.error(f"Pinecone batch upsert failed: {e}")
                raise
        return stored

    def _to_pinecone_filter(self, where: Optional[Dict]) -> Optional[Dict]:
        if not where:
            return None

        if "$and" in where or "$or" in where:
            op = "$and" if "$and" in where else "$or"
            clauses = where[op]
            if not isinstance(clauses, list):
                raise ValueError(f"{op} must be a list of filter clauses")
            normalized = []
            for clause in clauses:
                if not isinstance(clause, dict):
                    continue
                item = {}
                for k, v in clause.items():
                    if isinstance(v, dict) and any(
                        str(opk).startswith("$") for opk in v.keys()
                    ):
                        item[k] = v
                    else:
                        item[k] = {"$eq": v}
                if item:
                    normalized.append(item)
            return {op: normalized} if normalized else None

        pinecone_filter = {}
        for k, v in where.items():
            if isinstance(v, dict) and any(
                str(opk).startswith("$") for opk in v.keys()
            ):
                pinecone_filter[k] = v
            else:
                pinecone_filter[k] = {"$eq": v}
        return pinecone_filter

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        if not self.is_connected():
            logger.warning("Pinecone not connected. Returning empty results.")
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

        try:
            import asyncio
            import concurrent.futures

            try:
                asyncio.get_running_loop()
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    query_vector = pool.submit(
                        lambda: asyncio.run(_get_embedding(query_text))
                    ).result()
            except RuntimeError:
                query_vector = asyncio.run(_get_embedding(query_text))

            pinecone_filter = self._to_pinecone_filter(where)

            result = self._index.query(
                vector=query_vector,
                top_k=n_results,
                filter=pinecone_filter,
                include_metadata=True,
            )

            ids, documents, metadatas, distances = [], [], [], []
            for match in result.get("matches", []):
                ids.append(match["id"])
                meta = dict(match.get("metadata") or {})
                doc_text = meta.pop("_document", "") or meta.get("text", "")
                documents.append(doc_text)
                metadatas.append(meta)
                score = match.get("score", 0.0)
                distances.append(1.0 - score)

            return {
                "ids": [ids],
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
            }
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            return {
                "ids": [[]],
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
            }

    def get_collection_stats(self) -> Dict[str, Any]:
        if not self.is_connected():
            return {"count": 0, "name": self.index_name, "connected": False}
        try:
            stats = self._index.describe_index_stats()
            return {
                "count": stats.get("total_vector_count", 0),
                "name": self.index_name,
                "connected": True,
                "dimension": self.embedding_dim,
            }
        except Exception as e:
            return {"count": 0, "name": self.index_name, "error": str(e)}

    def paper_has_vectors(self, paper_id: str, topic: Optional[str] = None) -> bool:
        """True if at least one vector exists for this paper (and topic)."""
        if not self.is_connected():
            return False
        try:
            where = {"paper_id": paper_id}
            if topic:
                where = {"$and": [{"paper_id": paper_id}, {"topic": topic}]}
            # Cheap probe: neutral query + hard filter
            result = self.query(
                query_text=f"paper {paper_id}",
                n_results=1,
                where=where,
            )
            ids = (result.get("ids") or [[]])[0]
            return bool(ids)
        except Exception as e:
            logger.warning(f"paper_has_vectors check failed: {e}")
        return False
    
pinecone_client = PineconeVectorClient()
chroma_client = pinecone_client