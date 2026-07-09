"""
Memory Manager - Layered Storage (Artifact + Vector + Graph)
"""

from loguru import logger

from src.storage.artifact_store import artifact_store
from src.db.chroma_client import chroma_client
from src.db.neo4j_client import neo4j_client
from src.models.schemas import PerPaperOutput


class MemoryManager:
    def __init__(self):
        self.artifact = artifact_store
        self.vector = chroma_client
        self.graph = neo4j_client

    async def store_paper(self, output: PerPaperOutput, topic: str):
        """Main storage orchestration"""
        if not output.knowledge_note:
            logger.warning(f"No KnowledgeNote for {output.paper_id}")
            return

        # 1. Save to Artifact Store (Source of Truth)
        self.artifact.save_paper_artifacts(output, topic)

        # 2. Store in Vector DB
        self._store_in_vector(output, topic)

        # 3. Update Knowledge Graph (if available)
        self._update_graph(output)

        logger.success(f"Paper stored: {output.paper_id}")

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
            logger.warning(f"Vector DB storage failed: {e}")

    def _update_graph(self, output: PerPaperOutput):
        if not self.graph.is_connected():
            return

        paper = output.metadata
        self.graph.create_paper_node({
            "arxiv_id": output.paper_id,
            "title": paper.title,
            "abstract": paper.abstract,
            "published_date": str(paper.published_date),
            "topic": "agentic_rag_memory_systems"
        })

        for author in paper.authors:
            self.graph.create_author_relationship(output.paper_id, author.name)


# === This line is critical ===
memory_manager = MemoryManager()