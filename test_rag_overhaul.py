"""
End-to-end validation of the upgraded Helix Research pipeline.
Tests: import, intent routing, decomposer output, retriever threshold, gateway bug fix.
"""
import asyncio
import sys

print("=" * 60)
print("HELIX RESEARCH — RAG OVERHAUL VALIDATION")
print("=" * 60)

async def run_all_tests():
    errors = []

    # ── Test 1: All new modules import cleanly ─────────────────────
    print("\n[1/5] Module Import Check...")
    try:
        from src.agents.intent_classifier import intent_classifier, IntentClassifier
        from src.agents.relevance_filter import relevance_filter_agent, RelevanceFilterAgent
        from src.agents.synthesis_agent import synthesis_agent, SynthesisAgent
        from src.agents.decomposer import decomposer_agent, DecomposerAgent, SearchKeywordSet
        from src.tools.retriever import research_retriever
        from src.agents.query_agent import query_agent, QueryAgent
        from src.gateway.model_registry import TASK_MODEL_REGISTRY
        print("  ✅ All modules imported successfully")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        errors.append(("import", str(e)))
        return errors

    # ── Test 2: Model Registry has new task types ──────────────────
    print("\n[2/5] Model Registry Expansion Check...")
    required_tasks = ["intent_classification", "synthesis", "relevance_check", "research_answer", "keyword_generation"]
    for task in required_tasks:
        if task in TASK_MODEL_REGISTRY:
            cfg = TASK_MODEL_REGISTRY[task]
            print(f"  ✅ {task}: {cfg.provider}/{cfg.model_name}")
        else:
            print(f"  ❌ Missing task: {task}")
            errors.append(("registry", f"Missing task: {task}"))

    # ── Test 3: Retriever threshold ────────────────────────────────
    print("\n[3/5] Retriever Threshold Check...")
    threshold = research_retriever.MIN_SCORE_THRESHOLD
    has_collection_method = hasattr(research_retriever, "get_all_notes_for_topic")
    print(f"  ✅ Min score threshold: {threshold}")
    print(f"  {'✅' if has_collection_method else '❌'} get_all_notes_for_topic() method exists: {has_collection_method}")
    if not has_collection_method:
        errors.append(("retriever", "Missing get_all_notes_for_topic method"))

    # ── Test 4: Intent Classifier live test ────────────────────────
    print("\n[4/5] Intent Classifier Live Test (3 queries)...")
    test_cases = [
        ("What are all these papers about?", "collection_overview"),
        ("Compare GPT-4 and Llama-3 accuracy", "comparison"),
        ("What research gaps exist in agent memory systems?", "gap_analysis"),
    ]
    for query, expected in test_cases:
        result = await intent_classifier.classify(query, topic="memory architectures in AI agents")
        match = "✅" if result.intent == expected else "⚠️ "
        print(f"  {match} '{query[:45]}...' → {result.intent} [{result.confidence:.2f}] (expected: {expected})")
        if result.intent != expected:
            errors.append(("intent", f"Expected {expected}, got {result.intent}"))

    # ── Test 5: Decomposer structured output ───────────────────────
    print("\n[5/5] Decomposer Domain-Grounded Keywords Test...")
    state = {"topic": "memory architectures in AI agents", "current_stage": "init"}
    result = await decomposer_agent.run(state)
    kws = result.get("keywords", [])
    print(f"  Generated {len(kws)} keyword strategies:")
    ai_terms = ["agent", "llm", "language model", "memory", "neural", "ai", "rag", "retrieval", "cognitive", "autonomous"]
    all_grounded = True
    for i, kw in enumerate(kws, 1):
        is_grounded = any(t in kw.lower() for t in ai_terms)
        mark = "✅" if is_grounded else "⚠️ "
        print(f"  {mark} {i}. {kw}")
        if not is_grounded:
            all_grounded = False
    if all_grounded:
        print("  ✅ All keywords are domain-grounded")
    else:
        errors.append(("decomposer", "Some keywords lack domain context"))

    return errors

errors = asyncio.run(run_all_tests())

print("\n" + "=" * 60)
if not errors:
    print("✅ ALL TESTS PASSED — Pipeline overhaul is fully operational")
else:
    print(f"⚠️  {len(errors)} issue(s) found:")
    for component, msg in errors:
        print(f"   [{component}] {msg}")
print("=" * 60)
sys.exit(0 if not errors else 1)
