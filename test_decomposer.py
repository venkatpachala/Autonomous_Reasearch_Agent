from src.agents.decomposer import decompose_topic

initial_state = {
    "topic": "Efficient on-device agentic AI with long-term memory and thermal optimization"
}

result = decompose_topic(initial_state)

print("Intent:", result.get("intent"))
print("Keywords:", result.get("keywords"))
print("Search Strategy:", result.get("search_strategy"))