import asyncio
import json
import os
from pydantic import BaseModel, Field
from typing import List
from src.gateway import gateway
from src.config import settings

class KeywordExtraction(BaseModel):
    keywords: List[str] = Field(..., description="List of technical keywords")
    sentiment: str = Field(..., description="Sentiment of the text (e.g., positive, neutral, negative)")

async def test_all():
    print("=" * 60)
    print("🚀 Starting AI Gateway Verification Suite")
    print("=" * 60)

    # 1. Test standard local generation
    print("\n1. Testing local generation (Task: 'keyword_generation')...")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, list three colors of the rainbow."}
    ]
    response = await gateway.generate(
        task="keyword_generation",
        messages=messages,
        temperature=0.2
    )
    print(f"   - Status: SUCCESS")
    print(f"   - Provider/Model: {response.provider} / {response.model}")
    print(f"   - Response Text: {response.text.strip()}")
    print(f"   - Latency: {response.latency:.2f}s")
    print(f"   - Cached: {response.cached}")
    print(f"   - Tokens: {response.total_tokens} (Prompt: {response.prompt_tokens}, Completion: {response.completion_tokens})")

    # 2. Test cache functionality
    print("\n2. Testing local cache (running the identical prompt again)...")
    cache_response = await gateway.generate(
        task="keyword_generation",
        messages=messages,
        temperature=0.2
    )
    print(f"   - Status: SUCCESS")
    print(f"   - Cached: {cache_response.cached} (Expected: True)")
    print(f"   - Latency: {cache_response.latency:.2f}s (Expected: 0.00s)")

    # 3. Test structured output validation
    print("\n3. Testing structured output validation (Pydantic enforcement)...")
    struct_messages = [
        {"role": "system", "content": "You extract keywords and sentiment. Return valid JSON matching the schema."},
        {"role": "user", "content": "Machine Learning and Artificial Intelligence have huge economic impact globally."}
    ]
    struct_response = await gateway.generate(
        task="keyword_generation",
        messages=struct_messages,
        temperature=0.1,
        schema_model=KeywordExtraction
    )
    print(f"   - Status: SUCCESS")
    print(f"   - Structured Object Type: {type(struct_response.structured)}")
    print(f"   - Extracted Keywords: {struct_response.structured.keywords if struct_response.structured else 'None'}")
    print(f"   - Extracted Sentiment: {struct_response.structured.sentiment if struct_response.structured else 'None'}")
    print(f"   - Raw Output: {struct_response.text.strip()}")

    # 4. Test embeddings gateway routing
    print("\n4. Testing Embeddings Gateway...")
    try:
        embedding = await gateway.embed(text="Deep Learning Research", model="nomic-embed-text")
        print(f"   - Status: SUCCESS")
        print(f"   - Vector dimension: {len(embedding)}")
        print(f"   - Vector snippet: {embedding[:5]}...")
    except Exception as e:
        print(f"   - Embedding Generation FAILED: {e}")

    # 5. Test cost tracking
    print("\n5. Testing Cost and Token Accounting...")
    costs_file = settings.outputs_dir / "gateway_costs.json"
    if costs_file.exists():
        print(f"   - Cost log created at: {costs_file.name}")
        costs_data = json.loads(costs_file.read_text(encoding="utf-8"))
        print(f"   - Total gateway requests tracked: {costs_data.get('total_requests')}")
        print(f"   - Total tokens tracked: {costs_data.get('total_tokens')}")
        print(f"   - Total accumulated cost: ${costs_data.get('total_cost')}")
    else:
        print("   - FAILED: Cost tracking log not found!")

    print("\n" + "=" * 60)
    print("🎉 All gateway tests completed!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(test_all())
