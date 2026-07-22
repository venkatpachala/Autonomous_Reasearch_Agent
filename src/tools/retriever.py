"""
Semantic Retriever over the research knowledge base (Chroma + Artifact Store + Neo4j).
Upgraded with:
  - Score threshold filtering (discards low-confidence matches)
  - Retrieval confidence estimation
  - Collection-mode: load ALL notes for a topic (for synthesis queries)
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import json
from pathlib import Path

from src.db.pinecone_client import chroma_client  # Pinecone replaces ChromaDB (same interface)
from src.storage.artifact_store import artifact_store
from src.models.schemas import KnowledgeNote


class ResearchRetriever:
    # Papers below this score are discarded before being sent to the LLM
    MIN_SCORE_THRESHOLD = 0.25

    def __init__(self, n_results: int = 6):
        self.chroma = chroma_client
        self.artifact = artifact_store
        self.n_results = n_results

    def search(
        self,
        query: str,
        topic: Optional[str] = None,
        n_results: Optional[int] = None,
        min_score: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Semantic search with score threshold filtering and graph traversal.
        Returns: {papers, graph_triplets, retrieval_confidence}
        """
        n = n_results or self.n_results
        threshold = min_score if min_score is not None else self.MIN_SCORE_THRESHOLD
        where = {"topic": topic} if topic else None

        # Retrieve more candidates from Chroma, then filter by quality
        candidate_n = max(n * 3, 20)

        try:
            results = self.chroma.query(query_text=query, n_results=candidate_n, where=where)
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return {"papers": [], "graph_triplets": [], "retrieval_confidence": 0.0}

        if not results or not results.get("ids") or not results["ids"][0]:
            return {"papers": [], "graph_triplets": [], "retrieval_confidence": 0.0}

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)

        all_contexts = []
        for i, paper_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""
            raw_dist = distances[i]
            score = (1 - raw_dist) if raw_dist is not None else 0.5

            full_note = self._load_full_note(paper_id)
            paper_meta = self._load_metadata(paper_id)

            all_contexts.append({
                "paper_id": paper_id,
                "title": meta.get("title") or (full_note.title if full_note else "Unknown"),
                "content": doc,
                "full_note": full_note,
                "metadata": paper_meta or meta,
                "score": score,
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_path": str(self.artifact.base_dir / paper_id / "paper.pdf")
                            if (self.artifact.base_dir / paper_id / "paper.pdf").exists() else None,
            })

        # === SCORE THRESHOLD FILTERING ===
        filtered = [c for c in all_contexts if c["score"] >= threshold]
        filtered = sorted(filtered, key=lambda x: x["score"], reverse=True)[:n]

        if not filtered and all_contexts:
            # If threshold eliminates everything, take the best match regardless
            best = max(all_contexts, key=lambda x: x["score"])
            logger.warning(
                f"All {len(all_contexts)} results below threshold {threshold:.2f}. "
                f"Best score was {best['score']:.3f}. Taking top 1 result."
            )
            filtered = [best]

        retrieval_confidence = filtered[0]["score"] if filtered else 0.0

        if retrieval_confidence < 0.35:
            logger.warning(
                f"Low retrieval confidence ({retrieval_confidence:.3f}) for query: '{query[:60]}'. "
                "Results may not be relevant."
            )

        # === GRAPH RAG RELATIONSHIPS EXTRACTION ===
        import re
        from src.db.neo4j_client import neo4j_client

        entities_to_query = []
        for ctx in filtered:
            note = ctx.get("full_note")
            if note and note.concepts:
                entities_to_query.extend(note.concepts)

        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z0-9\-\.]+\b', query)
        entities_to_query.extend(proper_nouns)
        entities_to_query = list(set([e.strip() for e in entities_to_query if len(e.strip()) > 1]))

        graph_triplets = []
        if neo4j_client.is_connected() and entities_to_query:
            graph_triplets = neo4j_client.get_related_triplets(entities_to_query)

        return {
            "papers": filtered,
            "graph_triplets": graph_triplets,
            "retrieval_confidence": retrieval_confidence
        }

    def get_all_notes_for_topic(self, topic: str) -> List[KnowledgeNote]:
        """
        Load ALL KnowledgeNote objects indexed under a given topic.
        Used by the SynthesisAgent for collection-level queries.
        """
        try:
            # Query Chroma with a broad topic description to get all paper IDs
            results = self.chroma.query(
                query_text=topic,
                n_results=50,
                where={"topic": topic}
            )
            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            notes = []
            seen_ids = set()
            for paper_id in results["ids"][0]:
                if paper_id in seen_ids:
                    continue
                seen_ids.add(paper_id)
                note = self._load_full_note(paper_id)
                if note:
                    notes.append(note)

            logger.info(f"Loaded {len(notes)} full notes for topic '{topic}'")
            return notes

        except Exception as e:
            logger.error(f"Failed to load collection notes for topic '{topic}': {e}")
            return []

    def _load_full_note(self, paper_id: str) -> Optional[KnowledgeNote]:
        try:
            note_path = self.artifact.base_dir / paper_id / "knowledge_note.json"
            if note_path.exists():
                data = json.loads(note_path.read_text(encoding="utf-8"))
                return KnowledgeNote(**data)
        except Exception as e:
            logger.debug(f"Could not load full note for {paper_id}: {e}")
        return None

    def _load_metadata(self, paper_id: str) -> Optional[Dict]:
        try:
            meta_path = self.artifact.base_dir / paper_id / "metadata.json"
            if meta_path.exists():
                return json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass
        return None


research_retriever = ResearchRetriever()