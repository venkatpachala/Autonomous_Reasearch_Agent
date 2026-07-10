"""
Query Graph - Simple RAG over Research Knowledge Base
"""

from typing import TypedDict, Annotated, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END, START
from langgraph.graph.message import add_messages
from loguru import logger

from src.agents.query_agent import query_agent


class QueryState(TypedDict):
    question: str
    topic: Optional[str]
    messages: Annotated[list, add_messages]
    answer: Optional[str]
    sources: List[Dict[str, Any]]
    contexts_used: int


async def retrieve_and_answer(state: QueryState) -> QueryState:
    result = await query_agent.answer(state["question"], topic=state.get("topic"))

    state["answer"] = result["answer"]
    state["sources"] = result["sources"]
    state["contexts_used"] = result["contexts_used"]

    from langchain_core.messages import HumanMessage, AIMessage
    state["messages"] = state.get("messages", []) + [
        HumanMessage(content=state["question"]),
        AIMessage(content=result["answer"])
    ]
    return state


def build_query_graph():
    workflow = StateGraph(QueryState)
    workflow.add_node("answer", retrieve_and_answer)
    workflow.add_edge(START, "answer")
    workflow.add_edge("answer", END)
    return workflow.compile()


query_graph = build_query_graph()