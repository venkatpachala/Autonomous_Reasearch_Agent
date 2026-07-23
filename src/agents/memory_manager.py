"""
Memory Manager - Full Content + Parallel Chunk Storage
======================================================
A: Parallel embeddings + upserts
B: Semantic section + table-aware chunking
"""

from typing import List, Dict, Any
import asyncio
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

        # Tunables
        self.MAX_PARALLEL_CHUNKS = 8          # concurrent embeddings
        self.CHUNK_SIZE = 1100
        self.CHUNK_OVERLAP = 180

    async def store_paper(self, output: PerPaperOutput, topic: str):
        """
        Store full parsed content:
        1. Artifact Store
        2. Chunk → Parallel Embed → Pinecone
        3. Neo4j (paper node)
        4. Research Index
        """
        paper_id = output.paper_id
        extracted = output.extracted

        if not extracted or not getattr(extracted, "full_text", None):
            logger.warning(f"No extracted content for {paper_id} — skipping storage")
            return

        # 1. Artifact Store
        try:
            self.artifact.save_paper_artifacts(output, topic)
        except Exception as e:
            logger.warning(f"Artifact store failed for {paper_id}: {e}")

        # 2. Vector DB — Parallel chunk storage
        await self._store_chunks_in_vector(output, topic)

        # 3. Knowledge Graph (basic paper node)
        self._update_graph(output, topic)

        # 4. Research Index
        try:
            self.index.register_paper(
                arxiv_id=paper_id,
                title=output.metadata.title if output.metadata else paper_id,
                topic=topic
            )
        except Exception as e:
            logger.warning(f"Research Index failed for {paper_id}: {e}")

        logger.success(
            f"Paper fully stored + indexed: {paper_id} "
            f"({len(extracted.full_text)} chars)"
        )

    async def _store_chunks_in_vector(self, output: PerPaperOutput, topic: str):
        """Create high-quality chunks and store them in parallel."""
        extracted = output.extracted
        if not extracted:
            return

        chunks = self._create_chunks(extracted, output.paper_id, topic)
        if not chunks:
            logger.warning(f"No chunks created for {output.paper_id}")
            return

        logger.info(f"Storing {len(chunks)} chunks for {output.paper_id} (parallel={self.MAX_PARALLEL_CHUNKS})")

        # Parallel with concurrency limit
        semaphore = asyncio.Semaphore(self.MAX_PARALLEL_CHUNKS)

        async def _store_one(chunk: Dict[str, Any]):
            async with semaphore:
                try:
                    await self.vector.add_knowledge_note(
                        note_id=chunk["chunk_id"],
                        document=chunk["text"],
                        metadata=chunk["metadata"]
                    )
                except Exception as e:
                    logger.warning(f"Failed to store {chunk['chunk_id']}: {e}")

        await asyncio.gather(*[_store_one(c) for c in chunks])

    def _create_chunks(
        self,
        extracted: ExtractedContent,
        paper_id: str,
        topic: str
    ) -> List[Dict[str, Any]]:
        """
        High-quality chunking strategy:
        1. Prefer real sections if available
        2. Extract and keep tables as separate contiguous chunks
        3. Fall back to paragraph-based overlapping chunks
        """
        chunks = []
        chunk_idx = 0

        # ---------- 1. Section-based chunks (best quality) ----------
        if extracted.sections:
            for section_name, content in extracted.sections.items():
                content = (content or "").strip()
                if not content or len(content) < 80:
                    continue

                # If section is very long, further split it
                if len(content) > self.CHUNK_SIZE * 1.8:
                    sub_chunks = self._split_long_text(
                        content, paper_id, topic, section_name, chunk_idx
                    )
                    chunks.extend(sub_chunks)
                    chunk_idx += len(sub_chunks)
                else:
                    chunks.append({
                        "chunk_id": f"{paper_id}_sec_{chunk_idx}",
                        "text": f"Section: {section_name}\n\n{content}",
                        "metadata": {
                            "paper_id": paper_id,
                            "topic": topic,
                            "chunk_type": "section",
                            "section": section_name,
                            "chunk_index": chunk_idx,
                            "artifact_type": "chunk"
                        }
                    })
                    chunk_idx += 1

        # ---------- 2. Table chunks (keep intact) ----------
        if extracted.tables:
            for i, table in enumerate(extracted.tables):
                # table can be dict or string
                if isinstance(table, dict):
                    table_text = table.get("text") or table.get("content") or str(table)
                    caption = table.get("caption", f"Table {i+1}")
                else:
                    table_text = str(table)
                    caption = f"Table {i+1}"

                if table_text.strip():
                    chunks.append({
                        "chunk_id": f"{paper_id}_table_{i}",
                        "text": f"[Table] {caption}\n\n{table_text.strip()}",
                        "metadata": {
                            "paper_id": paper_id,
                            "topic": topic,
                            "chunk_type": "table",
                            "section": caption,
                            "chunk_index": chunk_idx,
                            "is_table": True,
                            "artifact_type": "chunk"
                        }
                    })
                    chunk_idx += 1

        # ---------- 3. Fallback: paragraph-based chunks ----------
        if not chunks and extracted.full_text:
            text = extracted.full_text.strip()
            paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

            current = []
            current_len = 0

            for para in paragraphs:
                if current_len + len(para) > self.CHUNK_SIZE and current:
                    chunk_text = "\n\n".join(current)
                    chunks.append({
                        "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                        "text": f"[Document: {paper_id}]\n{chunk_text}",
                        "metadata": {
                            "paper_id": paper_id,
                            "topic": topic,
                            "chunk_type": "text",
                            "chunk_index": chunk_idx,
                            "artifact_type": "chunk"
                        }
                    })
                    chunk_idx += 1

                    # Overlap
                    overlap_chars = 0
                    new_current = []
                    for prev in reversed(current):
                        if overlap_chars + len(prev) <= self.CHUNK_OVERLAP:
                            new_current.insert(0, prev)
                            overlap_chars += len(prev)
                        else:
                            break
                    current = new_current
                    current_len = sum(len(x) for x in current)

                current.append(para)
                current_len += len(para)

            if current:
                chunk_text = "\n\n".join(current)
                chunks.append({
                    "chunk_id": f"{paper_id}_chunk_{chunk_idx}",
                    "text": f"[Document: {paper_id}]\n{chunk_text}",
                    "metadata": {
                        "paper_id": paper_id,
                        "topic": topic,
                        "chunk_type": "text",
                        "chunk_index": chunk_idx,
                        "artifact_type": "chunk"
                    }
                })

        return chunks

    def _split_long_text(
        self,
        text: str,
        paper_id: str,
        topic: str,
        section_name: str,
        start_idx: int
    ) -> List[Dict[str, Any]]:
        """Split a long section into overlapping chunks."""
        chunks = []
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        current = []
        current_len = 0
        idx = start_idx

        for para in paragraphs:
            if current_len + len(para) > self.CHUNK_SIZE and current:
                chunk_text = "\n\n".join(current)
                chunks.append({
                    "chunk_id": f"{paper_id}_sec_{idx}",
                    "text": f"Section: {section_name}\n\n{chunk_text}",
                    "metadata": {
                        "paper_id": paper_id,
                        "topic": topic,
                        "chunk_type": "section",
                        "section": section_name,
                        "chunk_index": idx,
                        "artifact_type": "chunk"
                    }
                })
                idx += 1

                # Overlap
                overlap_chars = 0
                new_current = []
                for prev in reversed(current):
                    if overlap_chars + len(prev) <= self.CHUNK_OVERLAP:
                        new_current.insert(0, prev)
                        overlap_chars += len(prev)
                    else:
                        break
                current = new_current
                current_len = sum(len(x) for x in current)

            current.append(para)
            current_len += len(para)

        if current:
            chunk_text = "\n\n".join(current)
            chunks.append({
                "chunk_id": f"{paper_id}_sec_{idx}",
                "text": f"Section: {section_name}\n\n{chunk_text}",
                "metadata": {
                    "paper_id": paper_id,
                    "topic": topic,
                    "chunk_type": "section",
                    "section": section_name,
                    "chunk_index": idx,
                    "artifact_type": "chunk"
                }
            })

        return chunks

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
            logger.warning(f"Neo4j paper node update failed for {output.paper_id}: {e}")


# Global instance
memory_manager = MemoryManager()