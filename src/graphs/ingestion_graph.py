"""
Main Ingestion Graph with parallel per-paper pipeline using Send.
"""

from typing import Dict, Any

from langgraph.graph import StateGraph, END, START
from langgraph.types import Send
import re, asyncio
from loguru import logger

from src.agents.decomposer import decomposer_agent
from src.agents.pdf_extractor import pdf_extractor_node
from src.agents.summarizer import summarizer_agent
from src.agents.critic_note import critic_agent
from src.models.schemas import ResearchState
from src.tools.arxiv_tool import arxiv_tool   # ← ADD THIS LINE


async def decomposer_node(state: ResearchState) -> ResearchState:
    """Decompose topic into keywords."""
    return await decomposer_agent.run(state)


async def retriever_node(state: ResearchState) -> ResearchState:
    """Search arXiv using decomposed keywords (rate-limit friendly version)."""
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

    # Limit number of searches during development/testing
    clean_keywords = clean_keywords[:3]   # ← Only use first 3 keywords

    logger.info(f"Searching with {len(clean_keywords)} keywords: {clean_keywords}")

    for i, kw in enumerate(clean_keywords):
        logger.info(f"[{i+1}/{len(clean_keywords)}] Searching: {kw}")
        
        try:
            papers = await arxiv_tool.search(kw, state["topic"], max_results=4)
            all_papers.extend(papers)
        except Exception as e:
            logger.warning(f"Failed to search keyword '{kw}': {e}")
            continue
        
        # Important: Delay between different keyword searches
        if i < len(clean_keywords) - 1:
            await asyncio.sleep(7)   # 7 seconds between keyword searches

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
    """Dynamic parallel routing to per-paper pipeline."""
    return [
        Send("per_paper_pipeline", {"paper": paper, "topic": state["topic"]})
        for paper in state.get("papers_to_process", [])
    ]


async def per_paper_pipeline(state_input: dict) -> dict:
    """Full per-paper pipeline"""
    output = await pdf_extractor_node(state_input)
    output = await summarizer_agent.run(output)
    output = await critic_agent.run(output)

    return {
        "processed_papers": [output.dict() if hasattr(output, "dict") else output]
    }


# Build the graph
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