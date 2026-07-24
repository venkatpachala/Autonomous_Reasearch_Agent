"""
Memory Manager - Layered Storage (Artifact + Vector + Graph + Research Index)
Updated for full parsed content + chunk storage + full paper metadata.
"""
import re
from typing import List, Dict, Any, Optional
from loguru import logger

from src.storage.artifact_store import artifact_store
from src.db.pinecone_client import pinecone_client
from src.db.neo4j_client import neo4j_client
from src.tools.research_index import research_index
from src.models.schemas import PerPaperOutput, ExtractedContent


class MemoryManager:
    def __init__(self):
        self.artifact = artifact_store
        self.vector = pinecone_client
        self.graph = neo4j_client
        self.index = research_index

    async def store_paper(self, output: PerPaperOutput, topic: str):
        """
        Main storage orchestration.
        Stores FULL extracted content as chunks in Pinecone + metadata registry.
        """
        paper_id = output.paper_id
        extracted = output.extracted
        meta = output.metadata

        title = (getattr(meta, "title", None) if meta else None) or paper_id

        if not extracted or not getattr(extracted, "full_text", None):
            logger.warning(f"No extracted content for {paper_id} — skipping storage")
            return

        # 1. Artifact Store (source of truth on disk)
        try:
            self.artifact.save_paper_artifacts(output, topic)
        except Exception as e:
            logger.warning(f"Artifact store failed for {paper_id}: {e}")

        # 2. Vector DB — Chunk + Store full content
        await self._store_chunks_in_vector(output, topic, title)

        # 3. Knowledge Graph (optional)
        self._update_graph(output, topic)

        # 4. Research Index — full metadata for authors / dates / abstracts
        try:
            authors: List[str] = []
            if meta and getattr(meta, "authors", None):
                for a in meta.authors:
                    if hasattr(a, "name") and a.name:
                        authors.append(a.name)
                    elif isinstance(a, str) and a.strip():
                        authors.append(a.strip())

            published = ""
            if meta is not None:
                published = str(
                    getattr(meta, "published_date", None)
                    or getattr(meta, "published", None)
                    or ""
                )

            categories = None
            if meta is not None:
                categories = getattr(meta, "categories", None)

            abstract = getattr(meta, "abstract", None) if meta else None

            self.index.register_paper(
                arxiv_id=paper_id,
                title=title,
                topic=topic,
                authors=authors,
                abstract=abstract,
                published=published,
                categories=categories,
                pdf_path=getattr(output, "local_pdf_path", None),
                status="indexed",
            )
        except Exception as e:
            logger.warning(f"Research Index registration failed for {paper_id}: {e}")

        logger.success(
            f"Paper fully stored + indexed: {paper_id} "
            f"({len(extracted.full_text)} chars)"
        )

    async def _store_chunks_in_vector(
        self, output: PerPaperOutput, topic: str, title: str
    ):
        """Chunk the full extracted content and store in Pinecone."""
        extracted = output.extracted
        if not extracted:
            return

        chunks = self._create_chunks(
            extracted=extracted,
            paper_id=output.paper_id,
            topic=topic,
            title=title,
        )

        for chunk in chunks:
            try:
                await self.vector.add_knowledge_note(
                    note_id=chunk["chunk_id"],
                    document=chunk["text"],
                    metadata=chunk["metadata"],
                )
            except Exception as e:
                logger.warning(f"Failed to store chunk {chunk['chunk_id']}: {e}")

    def _create_chunks(
        self,
        extracted: ExtractedContent,
        paper_id: str,
        topic: str,
        title: str = "Untitled",
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> List[Dict[str, Any]]:
        """
        Create retrieval-friendly chunks from full extracted content.
        Prefers section-based chunks when available.
        """
        chunks: List[Dict[str, Any]] = []

        # 1. Prefer section-based chunks (best quality)
        if extracted.sections:
            for idx, (section_name, content) in enumerate(extracted.sections.items()):
                if content and content.strip():
                    safe_section = (
                        re.sub(r"[^\x00-\x7F]+", "", section_name)[:40]
                        .strip()
                        .replace(" ", "_")
                        or f"sec{idx}"
                    )
                    chunks.append({
                        "chunk_id": f"{paper_id}_section_{safe_section}",
                        "text": f"Section: {section_name}\n\n{content.strip()}",
                        "metadata": {
                            "paper_id": paper_id,
                            "title": title,
                            "topic": topic,
                            "chunk_type": "section",
                            "section": section_name,
                            "artifact_type": "chunk",
                        },
                    })

        # 2. Fallback: fixed-size paragraph chunks
        if not chunks and extracted.full_text:
            text = extracted.full_text.strip()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            current_chunk: List[str] = []
            current_length = 0
            chunk_idx = 0

            for para in paragraphs:
                if current_length + len(para) > chunk_size and current_chunk:
                    chunk_text = "\n\n".join(current_chunk)
                    chunks.append({
                        "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                        "text": f"[Document: {paper_id}]\n{chunk_text}",
                        "metadata": {
                            "paper_id": paper_id,
                            "title": title,
                            "topic": topic,
                            "chunk_index": chunk_idx,
                            "chunk_type": "text",
                            "artifact_type": "chunk",
                        },
                    })
                    chunk_idx += 1

                    # Keep overlap
                    overlap_chars = 0
                    new_chunk: List[str] = []
                    for prev in reversed(current_chunk):
                        if overlap_chars + len(prev) < overlap:
                            new_chunk.insert(0, prev)
                            overlap_chars += len(prev)
                        else:
                            break
                    current_chunk = new_chunk
                    current_length = sum(len(x) for x in current_chunk)

                current_chunk.append(para)
                current_length += len(para)

            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                chunks.append({
                    "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                    "text": f"[Document: {paper_id}]\n{chunk_text}",
                    "metadata": {
                        "paper_id": paper_id,
                        "title": title,
                        "topic": topic,
                        "chunk_index": chunk_idx,
                        "chunk_type": "text",
                        "artifact_type": "chunk",
                    },
                })

        return chunks

    def _update_graph(self, output: PerPaperOutput, topic: str):
        if not hasattr(self.graph, "is_connected") or not self.graph.is_connected():
            return

        try:
            paper = output.metadata
            if not paper:
                return

            self.graph.create_paper_node({
                "arxiv_id": output.paper_id,
                "title": getattr(paper, "title", output.paper_id),
                "abstract": getattr(paper, "abstract", "") or "",
                "published_date": str(
                    getattr(paper, "published_date", None)
                    or getattr(paper, "published", "")
                    or ""
                ),
                "topic": topic,
            })

            authors = getattr(paper, "authors", None) or []
            for author in authors:
                name = author.name if hasattr(author, "name") else str(author)
                if name:
                    self.graph.create_author_relationship(output.paper_id, name)
        except Exception as e:
            logger.warning(f"Neo4j update failed for {output.paper_id}: {e}")


memory_manager = MemoryManager()