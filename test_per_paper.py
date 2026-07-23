"""
Test Per-Paper Pipeline (PDF → ExtractedContent → Storage)
"""

import asyncio
from src.models.schemas import PaperMetadata, PerPaperInput
from src.graphs.ingestion_graph import per_paper_pipeline
from loguru import logger


async def test_per_paper():
    # Use a real paper that exists in your index
    paper = PaperMetadata(
        arxiv_id="2412.12881v1",
        title="Memory Architectures for AI Agents",
        authors=[],
        abstract="Test abstract for pipeline validation.",
        published_date="2024-12-01",
        pdf_url="https://arxiv.org/pdf/2412.12881v1.pdf",
        arxiv_url="https://arxiv.org/abs/2412.12881v1",
        categories=["cs.AI"]
    )

    input_data = PerPaperInput(paper=paper, topic="Graph RAG for retrievals")

    logger.info(f"Testing per-paper pipeline for {paper.arxiv_id}")

    try:
        output = await per_paper_pipeline({"paper": paper, "topic": "Graph RAG for retrievals"})
        print("✅ Per-paper pipeline SUCCESS")
        print(f"   Paper ID: {output['processed_papers'][0].get('paper_id')}")
        print(f"   Extracted text length: {len(output['processed_papers'][0].get('extracted', {}).get('full_text', ''))}")
    except Exception as e:
        logger.error(f"Per-paper pipeline FAILED: {e}")


if __name__ == "__main__":
    asyncio.run(test_per_paper())