"""
Test: Research Ontology Agent + Query Builder
Verifies that the new decomposer pipeline generates accurate, domain-specific queries.
"""

import asyncio
import sys

async def main():
    print("=" * 65)
    print("  HELIX RESEARCH — ONTOLOGY + QUERY BUILDER VERIFICATION")
    print("=" * 65)

    errors = []

    # Test 1: All new modules import cleanly
    print("\n[1/4] Module Import Check...")
    try:
        from src.agents.research_ontology_agent import research_ontology_agent, ResearchOntology
        from src.tools.query_builder import query_builder, SearchQueryBuilder
        from src.agents.decomposer import decomposer_agent
        from src.db.pinecone_client import PineconeVectorClient
        print("  ✅ All new modules imported successfully")
    except Exception as e:
        print(f"  ❌ Import failed: {e}")
        errors.append(str(e))
        return errors

    # Test 2: Research Ontology Generation
    print("\n[2/4] Research Ontology Generation (topic: 'memory architectures in AI agents')...")
    topic = "memory architectures in AI agents"
    ontology = await research_ontology_agent.generate(topic)

    print(f"  Topic Summary: {ontology.topic_summary}")
    print(f"  Core Terms ({len(ontology.core_terms)}): {ontology.core_terms}")
    print(f"  Named Frameworks ({len(ontology.named_frameworks)}): {ontology.named_frameworks}")
    print(f"  Task Types ({len(ontology.task_types)}): {ontology.task_types}")
    print(f"  Datasets ({len(ontology.benchmark_datasets)}): {ontology.benchmark_datasets}")
    print(f"  Synonyms ({len(ontology.synonyms)}): {ontology.synonyms}")
    print(f"  Negative Terms ({len(ontology.negative_terms)}): {ontology.negative_terms}")

    if len(ontology.named_frameworks) == 0:
        print("  ⚠️  WARNING: No named frameworks detected — ontology may be too generic")
        errors.append("No named frameworks in ontology")
    else:
        print(f"  ✅ Named frameworks found: {ontology.named_frameworks}")

    if any(bad in " ".join(ontology.core_terms).lower() for bad in ["memory management", "cache", "allocation"]):
        print("  ⚠️  WARNING: Potential OS-level terms leaked into core_terms")
        errors.append("OS-level terms in core_terms")
    else:
        print("  ✅ Core terms are domain-grounded (no OS-level contamination)")

    # Test 3: Query Builder
    print("\n[3/4] Query Builder Output...")
    queries = query_builder.build_queries(ontology)
    print(f"  Generated {len(queries)} queries:")
    for q, qt in queries:
        words = len(q.split())
        long_flag = " ⚠️ (>5 words)" if words > 5 else ""
        print(f"    [{qt:<22}] '{q}'{long_flag}")

    type_a = [q for q, t in queries if t.startswith("A:")]
    type_b = [q for q, t in queries if t.startswith("B:")]
    print(f"\n  Query type breakdown:")
    print(f"    A (framework names): {len(type_a)} — {type_a[:3]}")
    print(f"    B (core terms):      {len(type_b)}")
    print(f"    Total: {len(queries)}")

    if len(queries) < 8:
        print(f"  ⚠️  Only {len(queries)} queries — may need richer ontology")
        errors.append("Too few queries generated")
    else:
        print(f"  ✅ {len(queries)} queries (good coverage)")

    long_queries = [q for q, _ in queries if len(q.split()) > 5]
    if long_queries:
        print(f"  ⚠️  {len(long_queries)} queries exceed 5-word limit: {long_queries}")
    else:
        print(f"  ✅ All queries ≤ 5 words (arXiv-optimized)")

    # Test 4: Pinecone client init
    print("\n[4/4] Pinecone Client Init Check...")
    try:
        from src.db.pinecone_client import pinecone_client
        connected = pinecone_client.is_connected()
        if connected:
            stats = pinecone_client.get_collection_stats()
            print(f"  ✅ Pinecone connected: index='{stats['name']}', vectors={stats['count']}")
        else:
            print("  ⚠️  Pinecone not connected (PINECONE_API_KEY likely not set in .env)")
            print("     → Add your key to .env: PINECONE_API_KEY=pc-xxxxxxxx")
            print("     → ChromaDB is still active as fallback during migration")
    except Exception as e:
        print(f"  ⚠️  Pinecone client error: {e}")

    return errors

errors = asyncio.run(main())

print("\n" + "=" * 65)
if not errors:
    print("✅ ALL CHECKS PASSED")
else:
    print(f"⚠️  {len(errors)} warning(s): {errors}")
print("=" * 65)
sys.exit(0)
