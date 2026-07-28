"""
Test Research Ontology Agent — see core / related / negative terms for a topic.
Usage:
  python test_ontology.py
  python test_ontology.py "Edge devices for running local models"
"""

import asyncio
import json
import sys

from loguru import logger


async def main():
    topic = (
        " ".join(sys.argv[1:]).strip()
        if len(sys.argv) > 1
        else "Large Language Models for Health Care Industry"
    )

    print("=" * 60)
    print("Research Ontology Agent — test")
    print(f"Topic: {topic}")
    print("=" * 60)

    from src.agents.research_ontology_agent import research_ontology_agent

    ontology = await research_ontology_agent.generate(topic)

    # Pretty print (works for Pydantic v2)
    if hasattr(ontology, "model_dump"):
        data = ontology.model_dump()
    else:
        data = ontology.dict()

    print("\n--- Raw JSON ---")
    print(json.dumps(data, indent=2, ensure_ascii=False))

    print("\n--- Fields ---")
    core = getattr(ontology, "core_terms", None) or data.get("core_terms") or []
    related = getattr(ontology, "related_terms", None) or data.get("related_terms") or []
    negative = getattr(ontology, "negative_terms", None) or data.get("negative_terms") or []

    print(f"\ncore_terms ({len(core)}):")
    for t in core:
        print(f"  • {t}")

    print(f"\nrelated_terms ({len(related)}):")
    for t in related:
        print(f"  • {t}")

    print(f"\nnegative_terms ({len(negative)}):")
    for t in negative:
        print(f"  • {t}")

    # Optional: show what Query Builder would emit
    try:
        from src.tools.query_builder import query_builder

        queries = query_builder.build_queries(ontology)
        print(f"\n--- Query Builder ({len(queries)} queries) ---")
        for item in queries:
            if isinstance(item, tuple) and len(item) >= 2:
                q, qt = item[0], item[1]
                print(f"  [{qt}] {q}")
            else:
                print(f"  {item}")
    except Exception as e:
        logger.warning(f"Query builder skip: {e}")

    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())