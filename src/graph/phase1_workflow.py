from langgraph.graph import StateGraph, END, START
from typing import TypedDict, Annotated, List
from operator import add

from src.agents.decomposer import decompose_topic
from src.agents.retriever import retrieve_papers
from src.agents.pdf_extractor import pdf_extractor_agent
from src.agents.summarizer import summarize_papers
from src.agents.critic import critique_papers
from src.agents.synthesizer import synthesize_review

class ResearchState(TypedDict):
    topic: str
    sub_questions: Annotated[List[str], add]
    retrieved_papers: Annotated[List[dict], add]
    extracted_docs: Annotated[List[dict], add]
    summaries: Annotated[List[str], add]
    critiques: Annotated[List[str], add]
    final_literature_review: str
    memory_notes: Annotated[List[str], add]

def build_phase1_workflow():
    workflow = StateGraph(ResearchState)
    
    workflow.add_node("decompose", decompose_topic)
    workflow.add_node("retrieve", retrieve_papers)
    workflow.add_node("pdf_extract", pdf_extractor_agent)
    workflow.add_node("summarize", summarize_papers)
    workflow.add_node("critique", critique_papers)
    workflow.add_node("synthesize", synthesize_review)
    
    workflow.set_entry_point("decompose")
    workflow.add_edge("decompose", "retrieve")
    workflow.add_edge("retrieve", "pdf_extract")
    workflow.add_edge("pdf_extract", "summarize")
    workflow.add_edge("summarize", "critique")
    workflow.add_edge("critique", "synthesize")
    workflow.add_edge("synthesize", END)
    
    return workflow.compile()