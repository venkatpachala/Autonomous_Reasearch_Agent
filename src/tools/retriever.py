"""
Research Retriever — Hybrid (dense + BM25) + cross-encoder rerank.
Includes get_chunks_for_paper() and collection grouping.
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.db.pinecone_client import pinecone_client
from src.db.neo4j_client import neo4j_client


def _merge_candidates(
    dense: List[Dict[str, Any]],
    lexical: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Union by chunk_id (fallback: paper_id + content prefix)."""
    merged: Dict[str, Dict[str, Any]] = {}

    def key(c: Dict[str, Any]) -> str:
        if c.get("chunk_id"):
            return str(c["chunk_id"])
        return f"{c.get('paper_id')}|{(c.get('content') or '')[:100]}"

    for c in dense:
        k = key(c)
        item = dict(c)
        item.setdefault("source", "dense")
        merged[k] = item

    for c in lexical:
        k = key(c)
        if k in merged:
            merged[k]["bm25_score"] = c.get("bm25_score")
            merged[k]["source"] = "hybrid"
            if not merged[k].get("chunk_id") and c.get("chunk_id"):
                merged[k]["chunk_id"] = c["chunk_id"]
        else:
            item = dict(c)
            item.setdefault("dense_score", 0.0)
            item.setdefault("source", "bm25")
            merged[k] = item

    return list(merged.values())


class ResearchRetriever:

    async def search(
        self,
        query: str,
        topic: Optional[str] = None,
        n_results: int = 8,
        min_score: float = 0.20,
        retrieve_k: int = 30,
        bm25_k: int = 20,
        use_rerank: bool = True,
        use_bm25: bool = True,
    ) -> Dict[str, Any]:
        if not pinecone_client.is_connected():
            return {
                "papers": [],
                "graph_triplets": [],
                "retrieval_confidence": 0.0,
            }

        try:
            pool_k = max(retrieve_k, n_results) if use_rerank else n_results

            # ── Dense (Pinecone ANN) ──────────────────────────────────
            result = pinecone_client.query(
                query_text=query,
                n_results=pool_k,
                where={"topic": topic} if topic else None,
            )

            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]
            ids_list = (result.get("ids") or [[]])[0]

            dense_candidates: List[Dict[str, Any]] = []
            for i, (doc, meta, dist) in enumerate(zip(docs, metas, distances)):
                dense_score = 1.0 - dist if dist is not None else 0.0
                if dense_score < min_score:
                    continue
                cid = ids_list[i] if i < len(ids_list) else meta.get("chunk_id")
                dense_candidates.append({
                    "paper_id": meta.get("paper_id"),
                    "title": meta.get("title", "Untitled"),
                    "content": doc,
                    "score": dense_score,
                    "dense_score": dense_score,
                    "chunk_id": cid or meta.get("chunk_id"),
                    "arxiv_url": f"https://arxiv.org/abs/{meta.get('paper_id')}",
                    "chunk_type": meta.get("chunk_type"),
                    "section": meta.get("section"),
                    "source": "dense",
                })

            # ── Lexical (BM25) ────────────────────────────────────────
            lexical: List[Dict[str, Any]] = []
            if use_bm25:
                try:
                    from src.tools.bm25_store import bm25_store
                    lexical = bm25_store.search(
                        query, topic=topic, top_k=bm25_k
                    )
                except Exception as e:
                    logger.warning(f"BM25 search skipped: {e}")

            candidates = _merge_candidates(dense_candidates, lexical)

            if not candidates:
                return {
                    "papers": [],
                    "graph_triplets": [],
                    "retrieval_confidence": 0.0,
                }

            logger.info(
                f"Hybrid pool: dense={len(dense_candidates)} "
                f"bm25={len(lexical)} merged={len(candidates)}"
            )

            # ── Rerank ────────────────────────────────────────────────
            if use_rerank and len(candidates) > 1:
                from src.tools.reranker import rerank
                papers = rerank(
                    query, candidates, top_k=n_results, text_key="content"
                )
            else:
                papers = sorted(
                    candidates,
                    key=lambda x: (
                        x.get("dense_score")
                        or x.get("bm25_score")
                        or x.get("score")
                        or 0
                    ),
                    reverse=True,
                )[:n_results]

            # ── Graph enrichment (optional) ───────────────────────────
            graph_triplets: List[str] = []
            try:
                if neo4j_client.is_connected() and papers:
                    names = []
                    for p in papers[:5]:
                        if p.get("title"):
                            names.append(str(p["title"])[:80])
                    if hasattr(neo4j_client, "get_related_triplets") and names:
                        graph_triplets = neo4j_client.get_related_triplets(names)
            except Exception as e:
                logger.debug(f"Graph enrichment skipped: {e}")

            conf = 0.0
            if papers:
                conf = float(
                    papers[0].get("dense_score")
                    or papers[0].get("score")
                    or 0.0
                )
                if conf > 1.0:
                    conf = min(1.0, conf / 10.0)

            return {
                "papers": papers,
                "graph_triplets": graph_triplets,
                "retrieval_confidence": conf,
            }

        except Exception as e:
            logger.error(f"Retriever search failed: {e}")
            return {
                "papers": [],
                "graph_triplets": [],
                "retrieval_confidence": 0.0,
            }

    async def get_all_notes_for_topic(self, topic: Optional[str]) -> List[Dict]:
        if not pinecone_client.is_connected():
            return []

        try:
            result = pinecone_client.query(
                query_text=topic or "overview",
                n_results=100,
                where={"topic": topic} if topic else None,
            )

            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]

            notes = []
            for doc, meta in zip(docs, metas):
                notes.append({
                    "paper_id": meta.get("paper_id"),
                    "title": meta.get("title", "Untitled"),
                    "content": doc,
                    "arxiv_url": f"https://arxiv.org/abs/{meta.get('paper_id')}",
                    "score": 1.0,
                    "chunk_type": meta.get("chunk_type"),
                    "section": meta.get("section"),
                })

            logger.info(f"Loaded {len(notes)} chunks for topic '{topic}'")
            return notes

        except Exception as e:
            logger.error(f"get_all_notes_for_topic failed: {e}")
            return []

    async def get_chunks_for_paper(
        self,
        paper_id: str,
        topic: Optional[str] = None,
        n_results: int = 50,
    ) -> List[Dict]:
        """All chunks for a paper_id (metadata filter; no min_score)."""
        if not pinecone_client.is_connected():
            return []

        try:
            where_filter: Dict[str, Any] = {"paper_id": paper_id}
            if topic:
                where_filter["topic"] = topic

            result = pinecone_client.query(
                query_text=f"overview of paper {paper_id}",
                n_results=n_results,
                where=where_filter,
            )

            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            papers = []
            for doc, meta, dist in zip(docs, metas, distances):
                if meta.get("paper_id") != paper_id:
                    continue
                score = 1.0 - dist if dist is not None else 0.5
                papers.append({
                    "paper_id": meta.get("paper_id"),
                    "title": meta.get("title", "Untitled"),
                    "content": doc,
                    "score": score,
                    "arxiv_url": f"https://arxiv.org/abs/{meta.get('paper_id')}",
                    "chunk_type": meta.get("chunk_type"),
                    "section": meta.get("section"),
                })

            def _priority(c: Dict) -> int:
                ctype = c.get("chunk_type")
                if ctype == "table":
                    return 0
                if ctype == "section":
                    return 1
                return 2

            papers = sorted(papers, key=_priority)
            logger.info(f"get_chunks_for_paper({paper_id}) → {len(papers)} chunks")
            return papers

        except Exception as e:
            logger.error(f"get_chunks_for_paper failed: {e}")
            return []

    async def get_grouped_notes_for_topic(
        self,
        topic: Optional[str],
        max_chars_per_paper: int = 2500,
        max_chunks_per_paper: int = 4,
    ) -> List[Dict]:
        raw_chunks = await self.get_all_notes_for_topic(topic)
        if not raw_chunks:
            return []

        by_paper: Dict[str, List[Dict]] = {}
        order: List[str] = []
        for chunk in raw_chunks:
            pid = chunk.get("paper_id")
            if not pid:
                continue
            if pid not in by_paper:
                by_paper[pid] = []
                order.append(pid)
            by_paper[pid].append(chunk)

        titles: Dict[str, str] = {}
        try:
            from src.tools.research_index import research_index
            for pid in order:
                info = research_index.data.get("papers", {}).get(pid, {})
                if info.get("title"):
                    titles[pid] = info["title"]
        except Exception as e:
            logger.warning(f"Could not resolve titles from research_index: {e}")

        grouped: List[Dict] = []
        for pid in order:
            chunks = by_paper[pid]

            def _priority(c: Dict) -> int:
                ctype = c.get("chunk_type")
                if ctype == "table":
                    return 0
                if ctype == "section":
                    return 1
                return 2

            chunks_sorted = sorted(chunks, key=_priority)
            combined_parts = []
            total_len = 0
            for c in chunks_sorted[:max_chunks_per_paper]:
                text = (c.get("content") or "").strip()
                if not text:
                    continue
                if total_len + len(text) > max_chars_per_paper:
                    text = text[: max(0, max_chars_per_paper - total_len)]
                combined_parts.append(text)
                total_len += len(text)
                if total_len >= max_chars_per_paper:
                    break

            grouped.append({
                "paper_id": pid,
                "title": titles.get(pid, chunks[0].get("title", "Untitled")),
                "content": "\n\n".join(combined_parts),
                "arxiv_url": f"https://arxiv.org/abs/{pid}",
                "num_chunks": len(chunks),
                "score": 1.0,
            })

        logger.info(
            f"Grouped {len(raw_chunks)} chunks into {len(grouped)} papers "
            f"for topic '{topic}'"
        )
        return grouped


research_retriever = ResearchRetriever()