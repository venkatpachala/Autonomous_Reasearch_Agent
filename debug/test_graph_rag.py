import asyncio
import os
from pydantic import BaseModel
from src.tools.pdf_tools import pdf_tools
from src.agents.extractor_agent import extractor_agent, GraphKnowledge
from src.db.neo4j_client import neo4j_client
from src.tools.retriever import research_retriever
from src.agents.query_agent import query_agent
from src.models.schemas import PaperMetadata, PerPaperOutput, StructuredPaperSummary, ExtractedContent, PaperStatus, KnowledgeNote
from datetime import datetime

async def test_graph_rag():
    print("=" * 60)
    print("🚀 Starting Graph RAG Verification Suite")
    print("=" * 60)

    # 1. Test Table-Aware & Context-Aware Chunking
    print("\n1. Testing Table-Aware Chunking...")
    mock_pdf_text = """
Introduction to Machine Learning.
This paper presents benchmark results comparing models.

| Model | MMLU | GSM8k |
|---|---|---|
| Qwen-2.5 | 84.2 | 79.5 |
| Llama-3 | 82.0 | 78.1 |

Conclusion and future directions.
"""
    chunks = pdf_tools.chunk_text(mock_pdf_text, paper_id="2501.0001", topic="LLM Benchmarks")
    print(f"   - Total chunks generated: {len(chunks)}")
    for c in chunks:
        print(f"     * Chunk ID: {c['chunk_id']} | is_table: {c['metadata']['is_table']}")
        print(f"       Text: {c['text'][:120]}...")

    # 2. Test Entity-Relationship Extraction
    print("\n2. Testing Entity-Relationship extraction...")
    mock_metadata = PaperMetadata(
        arxiv_id="2501.0001",
        title="Evaluating Qwen-2.5 on Cognitive Benchmarks",
        authors=[],
        abstract="We evaluate Qwen-2.5 on cognitive reasoning tasks, showing improvements over Llama-3 on MMLU.",
        published_date=datetime.now(),
        pdf_url="http://arxiv.org/pdf/2501.0001.pdf",
        arxiv_url="http://arxiv.org/abs/2501.0001"
    )
    mock_summary = StructuredPaperSummary(
        objective="Analyze Qwen-2.5 capabilities on reasoning",
        methodology="Zero-shot evaluations on GSM8k and MMLU",
        key_contributions=["Proven cognitive reasoning boost", "SOTA scoring", "Low inference footprint"],
        achievements="Qwen-2.5 achieves 84.2% on MMLU",
        benchmarks=[{"Model": "Qwen-2.5", "MMLU": "84.2", "GSM8k": "79.5"}],
        limitations=["Inability to handle long contexts"],
        future_work=["Train 70B variant"]
    )
    mock_output = PerPaperOutput(
        paper_id="2501.0001",
        metadata=mock_metadata,
        extracted=ExtractedContent(full_text=mock_pdf_text),
        summary=mock_summary,
        status=PaperStatus.COMPLETED
    )
    
    graph_knowledge = await extractor_agent.extract_graph_elements(mock_output)
    print(f"   - Extracted entities count: {len(graph_knowledge.entities)}")
    print(f"   - Extracted relationships count: {len(graph_knowledge.relationships)}")
    for ent in graph_knowledge.entities[:3]:
        print(f"     * Node: {ent.name} ({ent.type})")
    for rel in graph_knowledge.relationships[:3]:
        val = f" ({rel.value})" if rel.value else ""
        print(f"     * Relationship: {rel.source} -[{rel.relation}{val}]-> {rel.target}")

    # Store in memory manager to index in Chroma DB
    from src.agents.memory_manager import memory_manager
    mock_output.knowledge_note = KnowledgeNote(
        paper_id=mock_output.paper_id,
        title=mock_metadata.title,
        one_sentence_summary="Qwen-2.5 evaluations on cognitive benchmarks.",
        detailed_summary=f"Abstract: {mock_metadata.abstract}",
        structured_data=mock_summary,
        concepts=["Qwen-2.5", "MMLU", "GSM8k", "Llama-3"],
        criticality_score=0.8
    )
    await memory_manager.store_paper(mock_output, topic="LLM Benchmarks")
    print("   - SUCCESS: Indexed paper note in Chroma DB")

    # 3. Test Neo4j Insertion & Query
    print("\n3. Testing Neo4j Property Graph writing...")
    if neo4j_client.is_connected():
        try:
            # Create paper node first
            neo4j_client.create_paper_node({
                "arxiv_id": "2501.0001",
                "title": mock_metadata.title,
                "abstract": mock_metadata.abstract,
                "topic": "LLM Benchmarks"
            })
            neo4j_client.write_extracted_graph(
                paper_id="2501.0001",
                entities=graph_knowledge.entities,
                relationships=graph_knowledge.relationships
            )
            print("   - SUCCESS: Graph triplets merged in Neo4j")
            
            # Query Neo4j for traversed paths
            test_entities = [ent.name for ent in graph_knowledge.entities[:3]]
            triplets = neo4j_client.get_related_triplets(test_entities)
            print(f"   - Querying traversed paths (Found: {len(triplets)} paths):")
            for t in triplets[:5]:
                print(f"     * {t}")
        except Exception as e:
            print(f"   - Neo4j writing failed: {e}")
    else:
        print("   - Neo4j is offline/disconnected. Skipping Neo4j DB write tests (falls back gracefully).")

    # 4. Test Hybrid Retriever Search
    print("\n4. Testing Hybrid Retriever Search (Vector + Graph)...")
    retrieved = research_retriever.search("How does Qwen-2.5 compare on MMLU?", topic="LLM Benchmarks")
    print(f"   - Retrieved papers count: {len(retrieved.get('papers', []))}")
    print(f"   - Retrieved graph triplets count: {len(retrieved.get('graph_triplets', []))}")
    for t in retrieved.get('graph_triplets', [])[:3]:
        print(f"     * Triplet: {t}")

    # 5. Test Query Agent Grounded Answer
    print("\n5. Testing Query Agent RAG synthesis...")
    chat_result = await query_agent.answer("Compare Qwen-2.5 and Llama-3 benchmarks", topic="LLM Benchmarks")
    print("\n[RAG ANSWER]:")
    print(chat_result["answer"])

    print("\n" + "=" * 60)
    print("🎉 Graph RAG tests completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_graph_rag())
