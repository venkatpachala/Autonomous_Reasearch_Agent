"""
Research Retriever — Final Fixed Version
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
                    "score": 1.0
                })

            logger.info(f"Loaded {len(notes)} chunks for topic '{topic}'")
            return notes

        except Exception as e:
            logger.error(f"get_all_notes_for_topic failed: {e}")
            return []

    async def get_grouped_notes_for_topic(
        self,
        topic: Optional[str],
        max_chars_per_paper: int = 2500,
        max_chunks_per_paper: int = 4,
    ) -> List[Dict]:
        """
        Collection-level retrieval, grouped by PAPER (not by chunk).

        get_all_notes_for_topic() returns one entry per vector chunk — with
        chunk-based storage a single paper can produce 10-20 chunks, which
        breaks any consumer that expects "one entry = one paper" (e.g.
        SynthesisAgent's collection overview). This method fetches the same
        raw chunks, groups them by paper_id, and returns exactly one summary
        entry per unique paper — preferring table/section chunks over plain
        text chunks when picking representative content, since those are the
        highest-signal pieces of a paper.
        """
        raw_chunks = await self.get_all_notes_for_topic(topic)
        if not raw_chunks:
            return []

        # Group chunks by paper_id, preserving first-seen order
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

        # Resolve real titles from the research index (chunk metadata
        # currently has no "title" field, so this is otherwise always
        # "Untitled")
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

            # Prefer table/section chunks as representative content —
            # they carry the highest information density per character.
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