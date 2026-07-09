from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from src.config import settings
from src.utils.prompts import DECOMPOSER_PROMPT

def decompose_topic(state):
    llm=ChatOllama(
        model=settings.OLLAMA_MODEL,
        temperature=settings.TEMPERATURE
    )
    prompt=DECOMPOSER_PROMPT.format(topic=state['topic'])
    response=llm.invoke([HumanMessage(content=prompt)])

    sub_questions = [q.strip() for q in response.content.strip().split("\n") if q.strip()]
    state["sub_questions"] = sub_questions
    
    print(f"Decomposed into {len(sub_questions)} sub-questions.")

    return {
    "sub_questions": sub_questions}