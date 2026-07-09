from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.config import settings

def decompose_topic(state):
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL, 
        temperature=0.4
    )
    
    prompt = f"""You are an expert research strategist.

User Topic: {state['topic']}

Your task:
1. Understand the core research intent.
2. Generate 5-7 high-quality keyword combinations or search phrases that would yield the most relevant academic papers.
3. Prioritize technical depth and recent advancements.

Return the output in this exact JSON format:
{{
  "intent": "Short description of research intent",
  "keywords": ["keyword1", "keyword2", ...],
  "search_strategy": "Brief explanation of how to search"
}}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Simple parsing (we'll improve later)
    try:
        import json
        data = json.loads(response.content.strip())
        state["intent"] = data.get("intent", state["topic"])
        state["keywords"] = data.get("keywords", [])
        state["search_strategy"] = data.get("search_strategy", "")
    except:
        state["intent"] = state["topic"]
        state["keywords"] = [state["topic"]]
        state["search_strategy"] = "General search"
    
    print(f"Decomposed topic. Intent: {state.get('intent')}")
    print(f"Generated {len(state.get('keywords', []))} keywords.")
    
    return state