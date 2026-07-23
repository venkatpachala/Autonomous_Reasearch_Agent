"""
Retrieval + Answer Debug
"""

import asyncio
from src.agents.query_agent import query_agent
from loguru import logger


async def test_retrieval():
    topic = "Retrieval frameworks for AI Agents"
    question = "What are all these papers about?"

    logger.info(f"Testing retrieval for question: {question}")

    try:
        result = await query_agent.answer(question, topic=topic)

        print("\n" + "="*60)
        print("RETRIEVAL + ANSWER TEST")
        print("="*60)
        print(f"Confidence: {result.get('retrieval_confidence', 0):.3f}")
        print(f"Contexts used: {result.get('contexts_used', 0)}")
        print(f"\nAnswer:\n{result.get('answer', 'No answer')[:800]}...")

        if result.get("sources"):
            print("\nSources:")
            for s in result["sources"][:5]:
                print(f"  • {s.get('paper_id')} | Score: {s.get('score')}")

    except Exception as e:
        logger.error(f"Retrieval test failed: {e}")


if __name__ == "__main__":
    asyncio.run(test_retrieval())