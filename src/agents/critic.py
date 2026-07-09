from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.config import settings

def critique_papers(state):
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL, 
        temperature=0.4
    )
    
    critiques = []
    
    for summary in state.get("summaries", []):
        prompt = f"""Critically evaluate this paper summary. Be specific and academic.

Summary: {summary}

Provide critique in this format:
- **Strengths**: ...
- **Weaknesses**: ...
- **Potential Biases**: ...
- **Open Questions**: ..."""

        response = llm.invoke([HumanMessage(content=prompt)])
        critiques.append(response.content)
    
    print(f"Generated {len(critiques)} detailed critiques.")
    return {
    "critiques": critiques}