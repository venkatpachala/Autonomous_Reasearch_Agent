"""
Memory Manager - Layered Storage (Artifact + Vector + Graph + Research Index + BM25)
Batch embeddings + batch Pinecone upsert + token-safe chunk sizes + lexical index.
"""
import re
import time
from typing import List, Dict, Any, Optional
from loguru import logger

from src.storage.artifact_store import artifact_store
from src.db.pinecone_client import pinecone_client
from src.db.neo4j_client import neo4j_client
from src.tools.research_index import research_index
from src.models.schemas import PerPaperOutput, ExtractedContent
from src.gateway.embeddings import embeddings_gateway

MAX_EMBED_CHARS = 6000


class MemoryManager:
    def __init__(self):
        self.artifact = artifact_store
        self.vector = pinecone_client
        self.graph = neo4j_client
        self.index = research_index

    async def store_paper(self, output: PerPaperOutput, topic: str):
        paper_id = output.paper_id
        extracted = output.extracted
        meta = output.metadata

        title = (getattr(meta, "title", None) if meta else None) or paper_id

        if not extracted or not getattr(extracted, "full_text", None):
            logger.warning(f"No extracted content for {paper_id} — skipping storage")
            return

        try:
            self.artifact.save_paper_artifacts(output, topic)
        except Exception as e:
            logger.warning(f"Artifact store failed for {paper_id}: {e}")

        await self._store_chunks_in_vector(output, topic, title)
        self._update_graph(output, topic)

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

            categories = getattr(meta, "categories", None) if meta else None
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
        extracted = output.extracted
        paper_id = output.paper_id

        if not extracted:
            return

        chunks = self._create_chunks(
            extracted=extracted,
            paper_id=paper_id,
            topic=topic,
            title=title,
        )
        if not chunks:
            logger.warning(f"No chunks created for {paper_id}")
            return

        # Always update lexical index (even when Pinecone is skipped)
        try:
            from src.tools.bm25_store import bm25_store
            bm25_store.add_chunks(topic, chunks)
        except Exception as e:
            logger.warning(f"BM25 index update failed for {paper_id}: {e}")

        # Stage 5: skip dense embed/upsert if already present
        if hasattr(self.vector, "paper_has_vectors") and self.vector.paper_has_vectors(
            paper_id, topic
        ):
            logger.info(f"Skip embed/upsert — vectors already exist for {paper_id}")
            return

        for c in chunks:
            if len(c["text"]) > MAX_EMBED_CHARS:
                logger.warning(
                    f"Chunk still over limit after split: {c['chunk_id']} "
                    f"({len(c['text'])} chars)"
                )

        texts = [c["text"] for c in chunks]

        t0 = time.perf_counter()
        vectors = await embeddings_gateway.embed_batch(texts)
        t_embed = time.perf_counter() - t0

        items: List[Dict[str, Any]] = []
        for chunk, vec in zip(chunks, vectors):
            meta = dict(chunk["metadata"])
            meta["_document"] = chunk["text"][:35000]
            items.append({
                "id": chunk["chunk_id"],
                "values": vec,
                "metadata": meta,
            })

        t1 = time.perf_counter()
        n = await self.vector.upsert_vectors(items)
        t_upsert = time.perf_counter() - t1

        logger.info(
            f"Vector store {paper_id}: {n}/{len(chunks)} chunks | "
            f"embed={t_embed:.1f}s upsert={t_upsert:.1f}s"
        )

        if n == 0:
            raise RuntimeError(
                f"No vectors stored for {paper_id} "
                "(all zero embeddings or Pinecone upsert failed)"
            )

    def _split_oversized(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []

        for c in chunks:
            text = c.get("text") or ""
            if len(text) <= MAX_EMBED_CHARS:
                out.append(c)
                continue

            meta = dict(c.get("metadata") or {})
            base_id = c["chunk_id"]
            parts = [p for p in text.split("\n\n") if p.strip()]
            buf: List[str] = []
            buf_len = 0
            part_i = 0

            def flush():
                nonlocal buf, buf_len, part_i
                if not buf:
                    return
                piece = "\n\n".join(buf)
                out.append({
                    "chunk_id": f"{base_id}_p{part_i}",
                    "text": piece,
                    "metadata": {
                        **meta,
                        "part": part_i,
                        "oversized_split": True,
                    },
                })
                part_i += 1
                buf, buf_len = [], 0

            for p in parts:
                if len(p) > MAX_EMBED_CHARS:
                    flush()
                    for j in range(0, len(p), MAX_EMBED_CHARS):
                        out.append({
                            "chunk_id": f"{base_id}_p{part_i}",
                            "text": p[j : j + MAX_EMBED_CHARS],
                            "metadata": {
                                **meta,
                                "part": part_i,
                                "oversized_split": True,
                            },
                        })
                        part_i += 1
                    continue

                if buf_len + len(p) + 2 > MAX_EMBED_CHARS and buf:
                    flush()
                buf.append(p)
                buf_len += len(p) + 2

            flush()
            logger.info(
                f"Split oversized chunk {base_id} ({len(text)} chars) → {part_i} parts"
            )

        return out

    def _create_chunks(
        self,
        extracted: ExtractedContent,
        paper_id: str,
        topic: str,
        title: str = "Untitled",
        chunk_size: int = 1200,
        overlap: int = 200,
    ) -> List[Dict[str, Any]]:
        chunks: List[Dict[str, Any]] = []

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

        return self._split_oversized(chunks)

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