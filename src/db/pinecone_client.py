"""
Pinecone Vector DB Client
==========================
Async embedding + chunk storage + batch upsert.
Sync query() runs embed via a safe event-loop bridge + stage timings.
get_by_paper_id() uses a cached neutral vector (no per-question embed).
"""

from __future__ import annotations

import asyncio
import time
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


def _run_async(coro):
    """Run async coroutine from sync code (query path)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(asyncio.run, coro).result()
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


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
    # One neutral query vector per process (paper_id filter fetches)
    _neutral_vector: Optional[List[float]] = None

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
        n_results: int = 8,
        where: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        empty = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "timings_ms": {},
        }
        if not self.is_connected():
            return empty

        t0 = time.perf_counter()
        stages: Dict[str, float] = {}

        try:
            # A. Embed
            t = time.perf_counter()
            vector = _run_async(_get_embedding(query_text))
            stages["embed_ms"] = (time.perf_counter() - t) * 1000.0

            if not vector or all(float(v) == 0.0 for v in vector):
                logger.error("Query embedding is zero — aborting Pinecone query")
                stages["total_ms"] = (time.perf_counter() - t0) * 1000.0
                empty["timings_ms"] = stages
                return empty

            # B. Filter
            t = time.perf_counter()
            pinecone_filter = self._to_pinecone_filter(where)
            stages["filter_ms"] = (time.perf_counter() - t) * 1000.0

            # C. ANN
            t = time.perf_counter()
            result = self._index.query(
                vector=vector,
                top_k=n_results,
                filter=pinecone_filter,
                include_metadata=True,
                include_values=False,
            )
            stages["ann_ms"] = (time.perf_counter() - t) * 1000.0

            # D. Parse
            t = time.perf_counter()
            ids, documents, metadatas, distances = [], [], [], []
            for match in result.get("matches", []) or []:
                ids.append(match["id"])
                meta = dict(match.get("metadata") or {})
                doc_text = meta.pop("_document", "") or meta.get("text", "") or ""
                documents.append(doc_text)
                metadatas.append(meta)
                try:
                    score = float(match.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                distances.append(1.0 - score)
            stages["parse_ms"] = (time.perf_counter() - t) * 1000.0

            stages["total_ms"] = (time.perf_counter() - t0) * 1000.0
            # IMPORTANT: format stages values, not the string keys
            logger.info(
                f"Pinecone detail ms | "
                f"embed={float(stages.get('embed_ms', 0)):.0f} "
                f"filter={float(stages.get('filter_ms', 0)):.0f} "
                f"ann={float(stages.get('ann_ms', 0)):.0f} "
                f"parse={float(stages.get('parse_ms', 0)):.0f} "
                f"total={float(stages.get('total_ms', 0)):.0f} "
                f"top_k={n_results} filtered={bool(where)}"
            )

            return {
                "ids": [ids],
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
                "timings_ms": stages,
            }
        except Exception as e:
            logger.error(f"Pinecone query failed: {e}")
            stages["total_ms"] = (time.perf_counter() - t0) * 1000.0
            empty["timings_ms"] = stages
            return empty

    async def get_by_paper_id(
        self,
        paper_id: str,
        topic: Optional[str] = None,
        n_results: int = 50,
    ) -> Dict[str, Any]:
        """
        Load chunks for a known paper_id without embedding the user question.
        Uses one cached neutral vector + metadata filter.
        """
        import json

        empty = {
            "ids": [[]],
            "documents": [[]],
            "metadatas": [[]],
            "distances": [[]],
            "timings_ms": {"embed": 0.0, "ann": 0.0, "total": 0.0},
        }
        if not self.is_connected():
            return empty

        paper_id = str(paper_id).strip()
        topic_s = str(topic).strip() if topic else None
        n_results = int(n_results) if n_results else 50

        try:
            # --- neutral vector (plain list[float], JSON-safe) ---
            if PineconeVectorClient._neutral_vector is None:
                t_e = time.perf_counter()
                raw = await _get_embedding("paper content overview")
                PineconeVectorClient._neutral_vector = [float(x) for x in raw]
                embed_ms = (time.perf_counter() - t_e) * 1000.0
                logger.info(
                    f"Cached neutral query vector for paper_id fetches "
                    f"(embed={embed_ms:.0f}ms, dim={len(PineconeVectorClient._neutral_vector)})"
                )
            else:
                embed_ms = 0.0

            neutral = [float(x) for x in PineconeVectorClient._neutral_vector]
            if not neutral or all(v == 0.0 for v in neutral):
                logger.error("Neutral vector is zero — aborting get_by_paper_id")
                return empty

            # --- filter: only str values, no Ellipsis / odd types ---
            if topic_s:
                pinecone_filter: Dict[str, Any] = {
                    "$and": [
                        {"paper_id": {"$eq": paper_id}},
                        {"topic": {"$eq": topic_s}},
                    ]
                }
            else:
                pinecone_filter = {"paper_id": {"$eq": paper_id}}

            # Fail fast if anything non-JSON sneaks in (catches Ellipsis)
            try:
                json.dumps(pinecone_filter)
                json.dumps(neutral[:2])  # sample; full vector is large but pure floats
            except TypeError as je:
                logger.error(f"get_by_paper_id preflight JSON fail: {je} filter={pinecone_filter!r}")
                return empty

            t0 = time.perf_counter()
            result = self._index.query(
                vector=neutral,
                top_k=n_results,
                filter=pinecone_filter,
                include_metadata=True,
            )
            ann_ms = (time.perf_counter() - t0) * 1000.0

            ids, documents, metadatas, distances = [], [], [], []
            for match in result.get("matches", []) or []:
                ids.append(match["id"])
                meta = dict(match.get("metadata") or {})
                doc_text = meta.pop("_document", "") or meta.get("text", "") or ""
                documents.append(doc_text)
                metadatas.append(meta)
                try:
                    score = float(match.get("score") or 0.0)
                except (TypeError, ValueError):
                    score = 0.0
                distances.append(1.0 - score)

            total_ms = float(embed_ms) + float(ann_ms)
            logger.info(
                f"Pinecone paper_id fetch | paper={paper_id} hits={len(ids)} "
                f"embed={float(embed_ms):.0f} ann={float(ann_ms):.0f} "
                f"total={total_ms:.0f}"
            )
            return {
                "ids": [ids],
                "documents": [documents],
                "metadatas": [metadatas],
                "distances": [distances],
                "timings_ms": {
                    "embed": float(embed_ms),
                    "ann": float(ann_ms),
                    "total": total_ms,
                },
            }
        except Exception as e:
            logger.error(f"get_by_paper_id failed: {e}")
            return empty

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
        if not self.is_connected():
            return False
        try:
            where: Dict[str, Any] = {"paper_id": paper_id}
            if topic:
                where = {"$and": [{"paper_id": paper_id}, {"topic": topic}]}
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