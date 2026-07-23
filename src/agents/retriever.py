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


research_retriever = ResearchRetriever()