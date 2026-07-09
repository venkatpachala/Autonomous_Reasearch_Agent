from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.config import settings

def summarize_papers(state):
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL, 
        temperature=0.3   # Lower for more consistent summaries
    )
    
    summaries = []
    
    for doc in state.get("extracted_docs", []):
        prompt = f"""Summarize the following research paper in a structured way.

Title: {doc.get('paper_title', 'Unknown')}

Content: {doc.get('extracted_text', '')[:4000]}

Provide the summary in this exact format:
- **Key Methods**: ...
- **Main Findings**: ...
- **Limitations**: ...
- **Contribution**: ..."""

        response = llm.invoke([HumanMessage(content=prompt)])
        summaries.append(response.content)
    
    state["summaries"] = summaries
    print(f"Generated {len(summaries)} structured summaries.")
    return {
    "summaries": summaries}