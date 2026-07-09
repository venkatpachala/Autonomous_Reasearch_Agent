from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.config import settings
from src.utils.prompts import SYNTHESIZER_PROMPT

def synthesize_review(state):
    """Synthesizer Agent - merges everything into final literature review"""
    llm = ChatOllama(
        model=settings.OLLAMA_MODEL, 
        temperature=0.6
    )
    
    # Prepare inputs safely
    sub_questions_str = "\n".join([f"- {q}" for q in state.get("sub_questions", [])])
    summaries_str = "\n\n".join(state.get("summaries", []))
    critiques_str = "\n\n".join(state.get("critiques", []))
    
    prompt = SYNTHESIZER_PROMPT.format(
        topic=state.get("topic", "Unknown Topic"),
        sub_questions=sub_questions_str,
        summaries=summaries_str,
        critiques=critiques_str
    )
    
    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Update basic memory (safe for reducer)
    new_note = f"Completed research on topic: {state.get('topic', 'Unknown')}"
    current_notes = state.get("memory_notes", [])
    
    print("Literature review synthesis completed.")

    return {
    "final_literature_review": response.content,
    "memory_notes": [new_note],}