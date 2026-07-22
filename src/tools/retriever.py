"""
Semantic Retriever over the research knowledge base (Chroma + Artifact Store).
"""

from typing import List, Dict, Any, Optional
from loguru import logger
import json
from pathlib import Path

from src.db.chroma_client import chroma_client
from src.storage.artifact_store import artifact_store
from src.models.schemas import KnowledgeNote


class ResearchRetriever:
    def __init__(self, n_results: int = 6):
        self.chroma = chroma_client
        self.artifact = artifact_store
        self.n_results = n_results

    def search(self, query: str, topic: Optional[str] = None, n_results: Optional[int] = None) -> Dict[str, Any]:
        n = n_results or self.n_results
        where = {"topic": topic} if topic else None

        try:
            results = self.chroma.query(query_text=query, n_results=n, where=where)
        except Exception as e:
            logger.error(f"Chroma query failed: {e}")
            return {"papers": [], "graph_triplets": []}

        contexts = []
        if not results or not results.get("ids") or not results["ids"][0]:
            return {"papers": [], "graph_triplets": []}

        ids = results["ids"][0]
        documents = results["documents"][0]
        metadatas = results["metadatas"][0]
        distances = results.get("distances", [[]])[0] if results.get("distances") else [None] * len(ids)

        for i, paper_id in enumerate(ids):
            meta = metadatas[i] if i < len(metadatas) else {}
            doc = documents[i] if i < len(documents) else ""

            full_note = self._load_full_note(paper_id)
            paper_meta = self._load_metadata(paper_id)

            contexts.append({
                "paper_id": paper_id,
                "title": meta.get("title") or (full_note.title if full_note else "Unknown"),
                "content": doc,
                "full_note": full_note,
                "metadata": paper_meta or meta,
                "score": 1 - distances[i] if distances[i] is not None else None,
                "arxiv_url": f"https://arxiv.org/abs/{paper_id}",
                "pdf_path": str(self.artifact.base_dir / paper_id / "paper.pdf")
                            if (self.artifact.base_dir / paper_id / "paper.pdf").exists() else None,
            })

        # === GRAPH RAG RELATIONSHIPS EXTRACTION ===
        import re
        from src.db.neo4j_client import neo4j_client

        entities_to_query = []
        # 1. Pull concepts from retrieved KnowledgeNotes
        for ctx in contexts:
            note = ctx.get("full_note")
            if note and note.concepts:
                entities_to_query.extend(note.concepts)

        # 2. Extract capitalized terms or acronyms from the query
        proper_nouns = re.findall(r'\b[A-Z][a-zA-Z0-9\-\.]+\b', query)
        entities_to_query.extend(proper_nouns)

        # 3. Clean duplicates
        entities_to_query = list(set([e.strip() for e in entities_to_query if len(e.strip()) > 1]))

        graph_triplets = []
        if neo4j_client.is_connected() and entities_to_query:
            graph_triplets = neo4j_client.get_related_triplets(entities_to_query)

        return {
            "papers": contexts,
            "graph_triplets": graph_triplets
        }

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