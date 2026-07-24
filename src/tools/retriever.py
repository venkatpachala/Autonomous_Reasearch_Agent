"""
Research Retriever — Final Fixed Version
Includes Paper Resolver support via get_chunks_for_paper().
"""

from typing import List, Dict, Any, Optional
from loguru import logger

from src.db.pinecone_client import pinecone_client
from src.db.neo4j_client import neo4j_client


class ResearchRetriever:

    async def search(
        self,
        query: str,
        topic: Optional[str] = None,
        n_results: int = 8,
        min_score: float = 0.30
    ) -> Dict[str, Any]:
        if not pinecone_client.is_connected():
            return {"papers": [], "graph_triplets": [], "retrieval_confidence": 0.0}

        try:
            result = pinecone_client.query(
                query_text=query,
                n_results=n_results,
                where={"topic": topic} if topic else None
            )

            docs = result.get("documents", [[]])[0]
            metas = result.get("metadatas", [[]])[0]
            distances = result.get("distances", [[]])[0]

            papers = []
            for doc, meta, dist in zip(docs, metas, distances):
                score = 1.0 - dist if dist is not None else 0.0
                if score < min_score:
                    continue

                papers.append({
                    "paper_id": meta.get("paper_id"),
                    "title": meta.get("title", "Untitled"),
                    "content": doc,
                    "score": score,
                    "arxiv_url": f"https://arxiv.org/abs/{meta.get('paper_id')}",
                    "chunk_type": meta.get("chunk_type"),
                    "section": meta.get("section")
                })

            graph_triplets = []
            if neo4j_client.is_connected() and papers:
                concepts = [p["paper_id"] for p in papers]
                graph_triplets = neo4j_client.get_related_triplets(concepts)

            confidence = sum(p["score"] for p in papers) / len(papers) if papers else 0.0

            return {
                "papers": papers,
                "graph_triplets": graph_triplets,
                "retrieval_confidence": confidence
            }

        except Exception as e:
            logger.error(f"Search failed: {e}")
            return {"papers": [], "graph_triplets": [], "retrieval_confidence": 0.0}

    async def get_all_notes_for_topic(self, topic: Optional[str]) -> List[Dict]:
        if not pinecone_client.is_connected():
            return []

        try:
            result = pinecone_client.query(
                query_text=topic or "overview",
                n_results=100,
                where={"topic": topic} if topic else None
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
        n_results: int = 50) -> List[Dict]:
        """
    Return ALL available chunks for a specific paper_id.

    For ordinal questions ("describe paper 5") we want the whole paper,
    not a similarity-ranked subset. So:
      - Filter strictly by paper_id (and topic if given)
      - Use a neutral query only to satisfy the vector API
      - Do NOT apply a high min_score cutoff
    """
        if not pinecone_client.is_connected():
            return []

        try:
            where_filter: Dict[str, Any] = {"paper_id": paper_id}
            if topic:
                where_filter = {
                    "$and": [
                        {"paper_id": paper_id},
                        {"topic": topic},
                    ]
                }

        # Neutral query — we only care about the metadata filter
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
                # Hard filter again in case the backend ignored part of the where clause
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

        # Prefer section/table chunks first for better coverage
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
        """
        Collection-level retrieval, grouped by PAPER (not by chunk).
        Returns exactly one entry per unique paper.
        """
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

        # Resolve real titles from research index
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
            f"Grouped {len(raw_chunks)} chunks into {len(grouped)} papers for topic '{topic}'"
        )
        return grouped


research_retriever = ResearchRetriever()