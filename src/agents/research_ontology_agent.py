# src/agents/research_ontology_agent.py

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger
import json as _json
import re
import asyncio

from src.gateway import gateway


class ResearchOntology(BaseModel):
    topic_summary: str = Field(..., description="One precise sentence describing the topic.")
    core_terms: List[str] = Field(..., min_length=3, max_length=8)

    task_types: List[str] = Field(
        default_factory=list,
        description=(
            "Specific tasks, problems, or challenges this research addresses. "
            "Example: ['memory compression', 'context retrieval', 'continual learning', 'catastrophic forgetting']"
        )
    )
    benchmark_datasets: List[str] = Field(
        default_factory=list,
        description=(
            "Known benchmark datasets or evaluation suites used in this research area. "
            "Example: ['LoCoMo', 'LOCRET', 'MemGPT-eval', 'LongBench']"
        )
    )
    evaluation_metrics: List[str] = Field(
        default_factory=list,
        description=(
            "Specific metrics used to evaluate models in this space. "
            "Example: ['Recall@k', 'NDCG', 'Needle In A Haystack', 'BLEU']"
        )
    )
    methods: List[str] = Field(
        default_factory=list,
        description=(
            "Specific methods, algorithms, or architectural patterns. "
            "Example: ['vector retrieval', 'key-value memory', 'hierarchical memory', 'RAG']"
        )
    )
    synonyms: List[str] = Field(
        default_factory=list,
        description=(
            "Alternative phrasings, related terms, and broader concepts. "
            "Example: ['long-term memory LLM', 'persistent memory agent', 'external memory store']"
        )
    )
    key_authors: List[str] = Field(
        default_factory=list,
        description=(
            "Last names of prominent researchers known for this specific topic. "
            "Example: ['Lewis', 'Karpukhin', 'Izacard', 'Gao']"
        )
    )
    venues: List[str] = Field(
        default_factory=list,
        description=(
            "Top conferences or journals where this research is published. "
            "Example: ['ACL', 'EMNLP', 'NeurIPS', 'ICLR', 'SIGIR']"
        )
    )
    negative_terms: List[str] = Field(
        default_factory=list,
        description=(
            "Terms that look relevant but would return wrong papers (OS/hardware/generic CS). "
            "Example: ['memory management', 'cache eviction', 'RAM allocation', 'memory leak']"
        )
    )


class ResearchOntologyAgent:
    async def generate(self, topic: str) -> ResearchOntology:
        example_json = '''{
  "topic_summary": "Research on running large language models efficiently on edge devices and mobile hardware.",
  "core_terms": ["on-device LLM", "edge inference", "model quantization", "efficient LLM"],
  "named_frameworks": ["TinyChat", "MLC-LLM", "llama.cpp", "TensorRT-LLM", "ExecuTorch", "ONNX Runtime"],
  "task_types": ["4-bit quantization", "weight pruning", "speculative decoding", "memory-efficient inference"],
  "benchmark_datasets": ["MMLU", "WikiText-2", "C4", "HellaSwag", "LMBench"],
  "evaluation_metrics": ["tokens per second", "perplexity", "peak memory usage", "latency (ms)"],
  "methods": ["AWQ", "GPTQ", "SmoothQuant", "knowledge distillation", "structured pruning"],
  "synonyms": ["mobile LLM", "embedded AI inference", "tinyML language model", "resource-constrained LLM"],
  "key_authors": ["Lin", "Han", "Zhu", "Guo"],
  "venues": ["MLSys", "NeurIPS", "ICLR", "ASPLOS", "MobiSys"],
  "negative_terms": ["memory management", "cache eviction", "RAM allocation", "operating system kernel", "network routing"]
}'''

        system = f"""You are a senior AI research librarian.

Your ONLY job is to produce a grounded research ontology for arXiv search.

STRICT RULES (must follow):
1. Output ONLY valid JSON. No markdown, no explanations, no extra text.
2. named_frameworks: Include well-known frameworks, libraries, or systems that are commonly published
   about in this research area. Be generous but accurate — include names you have strong knowledge of
   (e.g. TinyChat, MLC-LLM, llama.cpp, TensorRT-LLM, ExecuTorch for edge inference).
   If truly uncertain, return an empty list.
3. benchmark_datasets: Include real evaluation benchmarks or datasets used in the field.
   Examples for edge AI: MMLU, WikiText-2, C4, HellaSwag, LMBench.
4. ANTI-HALLUCINATION: Do not invent acronyms or made-up system names. Only list things you know exist.
5. core_terms must be short (1-4 words) and likely to appear in paper titles/abstracts.
6. Prefer specific, targeted terms over broad generic ones (e.g. 'on-device LLM' > 'LLM').
7. negative_terms = generic CS/OS terms that would pollute results.

Example for an edge AI inference topic:
{example_json}
"""

        messages = [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": (
                    f"Research Topic: {topic}\n\n"
                    "Output ONLY the JSON object. Start with {{ and end with }}."
                )
            }
        ]

        for attempt in range(3):
            try:
                response = await gateway.generate(
                    task="keyword_generation",
                    messages=messages,
                    temperature=0.05,   # lower temperature for more grounded output
                )
                raw = response.text.strip()

                # Clean markdown if present
                if "```" in raw:
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
                    if match:
                        raw = match.group(1)

                start, end = raw.find("{"), raw.rfind("}") + 1
                if start != -1 and end > start:
                    raw = raw[start:end]

                parsed = _json.loads(raw)
                ontology = ResearchOntology.model_validate(parsed)

                # Extra safety: strip parentheticals and drop any framework that looks invented
                ontology.named_frameworks = [
                    re.sub(r"\s*\(.*?\)\s*", "", fw).strip()
                    for fw in ontology.named_frameworks
                ]
                ontology.named_frameworks = [
                    fw for fw in ontology.named_frameworks
                    if len(fw) > 2 and (not fw.isupper() or fw in {"RAG", "DPR", "ANCE", "ColBERT", "MemGPT", "BERT", "GPT"})
                ]

                logger.success(
                    f"Ontology for '{topic}':\n"
                    f"  Core: {ontology.core_terms}\n"
                    f"  Frameworks: {ontology.named_frameworks}\n"
                    f"  Datasets: {ontology.benchmark_datasets}"
                )
                return ontology

            except Exception as e:
                logger.warning(f"Ontology attempt {attempt+1}/3 failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.2)

        # Safe minimal fallback
        logger.error(f"Ontology failed for '{topic}'. Using minimal fallback.")
        return ResearchOntology(
            topic_summary=f"Research on {topic}",
            core_terms=[topic, f"{topic} survey", f"{topic} deep learning"][:3],
            named_frameworks=[],
            task_types=[],
            benchmark_datasets=[],
            evaluation_metrics=[],
            methods=[],
            synonyms=[f"{topic} neural network", f"{topic} language model"],
            key_authors=[],
            venues=[],
            negative_terms=["memory management", "cache", "operating system"]
        )


research_ontology_agent = ResearchOntologyAgent()