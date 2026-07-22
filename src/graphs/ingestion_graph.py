"""
Main Ingestion Graph with Layered Storage (Artifact + Vector + Graph)
"""

from typing import Dict, Any
import asyncio
import re
from loguru import logger

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send

from src.agents.decomposer import decomposer_agent
from src.agents.pdf_extractor import pdf_extractor_node
from src.agents.summarizer import summarizer_agent
from src.agents.critic_note import critic_agent
from src.agents.memory_manager import memory_manager
from src.models.schemas import ResearchState
from src.tools.arxiv_tool import arxiv_tool


# ==================== NODES ====================

async def decomposer_node(state: ResearchState) -> ResearchState:
    return await decomposer_agent.run(state)


async def retriever_node(state: ResearchState) -> ResearchState:
    """Search arXiv using cleaned keywords (rate-limit friendly)."""
    all_papers = []
    raw_keywords = state.get("keywords", [state["topic"]])

    # Clean keywords
    clean_keywords = []
    for kw in raw_keywords:
        kw = re.sub(r'^\d+[\.\)]\s*', '', kw).strip()
        kw = kw.replace('**', '').replace('*', '').strip()
        kw = kw.strip('"\'')
        if kw:
            clean_keywords.append(kw)

    clean_keywords = clean_keywords[:3]  # Limit during development

    for i, kw in enumerate(clean_keywords):
        logger.info(f"[{i+1}/{len(clean_keywords)}] Searching: {kw}")
        try:
            papers = await arxiv_tool.search(kw, state["topic"], max_results=4)
            all_papers.extend(papers)
        except Exception as e:
            logger.warning(f"Failed to search '{kw}': {e}")
            continue
        if i < len(clean_keywords) - 1:
            await asyncio.sleep(7)

    # Deduplication
    seen = {}
    unique_papers = []
    for p in all_papers:
        if p.arxiv_id not in seen:
            seen[p.arxiv_id] = p
            unique_papers.append(p)

    state["papers"] = unique_papers[:8]
    state["papers_to_process"] = unique_papers[:8]
    state["current_stage"] = "retrieve"
    return state


def route_to_parallel(state: ResearchState):
    return [
        Send("per_paper_pipeline", {"paper": paper, "topic": state["topic"]})
        for paper in state.get("papers_to_process", [])
    ]


async def per_paper_pipeline(state_input: dict) -> dict:
    """Full per-paper pipeline + Layered Storage + Property Graph Extractor"""
    topic = state_input.get("topic", "unknown")

    # Run extraction → summarization → critic
    output = await pdf_extractor_node(state_input)
    output = await summarizer_agent.run(output)
    output = await critic_agent.run(output)

    # === Store using new layered Memory Manager ===
    try:
        await memory_manager.store_paper(output, topic)
    except Exception as e:
        logger.error(f"Memory Manager failed for {output.paper_id}: {e}")

    # === Extract Property Graph Entities & Write to Neo4j ===
    try:
        from src.agents.extractor_agent import extractor_agent
        from src.db.neo4j_client import neo4j_client
        if neo4j_client.is_connected():
            graph_data = await extractor_agent.extract_graph_elements(output)
            neo4j_client.write_extracted_graph(output.paper_id, graph_data.entities, graph_data.relationships)
    except Exception as e:
        logger.error(f"Failed to compile property graph for {output.paper_id}: {e}")

    return {
        "processed_papers": [output.dict() if hasattr(output, "dict") else output]
    }


# ==================== GRAPH ====================

def build_ingestion_graph():
    workflow = StateGraph(ResearchState)

    workflow.add_node("decomposer", decomposer_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("per_paper_pipeline", per_paper_pipeline)

    workflow.add_edge(START, "decomposer")
    workflow.add_edge("decomposer", "retriever")
    workflow.add_conditional_edges(
        "retriever",
        route_to_parallel,
        ["per_paper_pipeline"]
    )
    workflow.add_edge("per_paper_pipeline", END)

    return workflow.compile()


ingestion_graph = build_ingestion_graph()