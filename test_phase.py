"""
Individual Phase Tests
"""

import asyncio
from src.agents.pdf_extractor import pdf_extractor_node
from src.agents.memory_manager import memory_manager
from src.agents.extractor_agent import extractor_agent
from src.models.schemas import PaperMetadata, PerPaperInput
from loguru import logger


async def test_phases():
    paper = PaperMetadata(
        arxiv_id="2404.14464v1",
        title="Tree of Reviews: A Tree-based Dynamic Iterative Retrieval-Augmented Generation",
        authors=[],
        abstract="Test paper",
        published_date="2024-04-01",
        pdf_url="https://arxiv.org/pdf/2404.14464v1.pdf",
        arxiv_url="https://arxiv.org/abs/2404.14464v1",
        categories=["cs.AI"]
    )

    input_data = PerPaperInput(paper=paper, topic="Retrieval frameworks for AI Agents")

    print("=== PHASE 1: PDF Extraction ===")
    output = await pdf_extractor_node(input_data)
    print(f"Status: {output.status}")
    print(f"Text length: {len(output.extracted.full_text) if output.extracted else 0}")

    print("\n=== PHASE 2: Memory Storage (Chunks) ===")
    await memory_manager.store_paper(output, "Retrieval frameworks for AI Agents")

    print("\n=== PHASE 3: Graph Extraction ===")
    graph = await extractor_agent.extract(
        paper_id=paper.arxiv_id,
        title=paper.title,
        full_text=output.extracted.full_text[:5000]
    )
    print(f"Entities: {len(graph.entities)} | Relationships: {len(graph.relationships)}")


if __name__ == "__main__":
    asyncio.run(test_phases())