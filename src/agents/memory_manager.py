"""
Memory Manager - Layered Storage (Artifact + Vector + Graph + Research Index)
"""

from loguru import logger

from src.storage.artifact_store import artifact_store
from src.db.chroma_client import chroma_client
from src.db.neo4j_client import neo4j_client
from src.tools.research_index import research_index
from src.models.schemas import PerPaperOutput


class MemoryManager:
    def __init__(self):
        self.artifact = artifact_store
        self.vector = chroma_client
        self.graph = neo4j_client
        self.index = research_index

    async def store_paper(self, output: PerPaperOutput, topic: str):
        """
        Main storage orchestration.
        Writes to:
          1. Artifact Store (source of truth)
          2. Vector DB (Chroma)
          3. Knowledge Graph (Neo4j) - optional
          4. Research Index (for deduplication + continuous monitoring)
        """
        if not output.knowledge_note:
            logger.warning(f"No KnowledgeNote for {output.paper_id} — skipping storage")
            return

        # 1. Artifact Store (files on disk)
        self.artifact.save_paper_artifacts(output, topic)

        # 2. Vector DB
        self._store_in_vector(output, topic)

        # 3. Knowledge Graph (if available)
        self._update_graph(output, topic)

        # 4. Research Index (CRITICAL for Continuous Monitor + deduplication)
        self.index.register_paper(
            arxiv_id=output.paper_id,
            title=output.metadata.title if output.metadata else output.knowledge_note.title,
            topic=topic
        )

        logger.success(f"Paper fully stored + indexed: {output.paper_id}")

    def _store_in_vector(self, output: PerPaperOutput, topic: str):
        note = output.knowledge_note
        document = f"{note.title}\n\n{note.detailed_summary}"

        metadata = {
            "paper_id": output.paper_id,
            "topic": topic,
            "title": note.title,
            "artifact_type": "knowledge_note"
        }

        try:
            self.vector.add_knowledge_note(
                note_id=output.paper_id,
                document=document,
                metadata=metadata
            )
        except Exception as e:
            logger.warning(f"Vector DB storage failed for {output.paper_id}: {e}")

    def _update_graph(self, output: PerPaperOutput, topic: str):
        if not hasattr(self.graph, "is_connected") or not self.graph.is_connected():
            return

        try:
            paper = output.metadata
            self.graph.create_paper_node({
                "arxiv_id": output.paper_id,
                "title": paper.title,
                "abstract": paper.abstract,
                "published_date": str(paper.published_date),
                "topic": topic
            })

            for author in paper.authors:
                self.graph.create_author_relationship(output.paper_id, author.name)
        except Exception as e:
            logger.warning(f"Neo4j update failed for {output.paper_id}: {e}")


# Global instance
memory_manager = MemoryManager()

