"""
Research Ontology Agent
========================
Transforms a free-form research topic into a structured domain ontology.
The ontology captures the terminology that researchers actually use —
named frameworks, task types, benchmark datasets, synonyms, and negative terms.

This is the critical upstream step before query building. The LLM's job here is
domain understanding, NOT search query generation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field
from loguru import logger

from src.gateway import gateway


class ResearchOntology(BaseModel):
    """
    A structured domain map of a research topic.
    Captures the exact vocabulary researchers use when writing about this topic.
    """
    topic_summary: str = Field(
        ...,
        description="One precise sentence describing what this research topic is about."
    )
    core_terms: List[str] = Field(
        ...,
        min_length=3,
        max_length=8,
        description=(
            "Essential 1-3 word research terms. These are the backbone of the search. "
            "Example: ['episodic memory', 'working memory', 'memory-augmented agent']"
        )
    )
    named_frameworks: List[str] = Field(
        default_factory=list,
        description=(
            "Specific named systems, frameworks, models, or tools in this space. "
            "These often appear verbatim in paper titles. "
            "Example: ['MemGPT', 'MemoryOS', 'MemoryBank', 'LightMem', 'Reflexion']"
        )
    )
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
    negative_terms: List[str] = Field(
        default_factory=list,
        description=(
            "Terms that look relevant but would return wrong papers (OS/hardware/generic CS). "
            "Example: ['memory management', 'cache eviction', 'RAM allocation', 'memory leak']"
        )
    )


class ResearchOntologyAgent:
    """
    LLM-powered domain understanding agent.
    Produces a structured ResearchOntology from a free-form topic string.
    """

    async def generate(self, topic: str) -> ResearchOntology:
        """Generate a structured research ontology for the given topic."""

        # NOTE: We do NOT use schema_model= here because qwen2.5:7b confuses the
        # injected JSON Schema spec with "what to return" and echoes it back.
        # Instead we provide an explicit example-driven prompt and parse manually.

        example_json = '''{
  "topic_summary": "Research on episodic and working memory mechanisms in LLM-powered autonomous agents.",
  "core_terms": ["episodic memory", "working memory", "memory-augmented agent"],
  "named_frameworks": ["MemGPT", "MemoryOS", "MemoryBank", "Reflexion"],
  "task_types": ["context retrieval", "continual learning", "memory compression", "catastrophic forgetting"],
  "benchmark_datasets": ["LoCoMo", "LOCRET", "MemGPT-eval", "LongBench"],
  "methods": ["vector retrieval", "key-value memory", "hierarchical memory", "RAG"],
  "synonyms": ["long-term memory LLM", "persistent memory agent", "external memory store"],
  "negative_terms": ["memory management", "cache eviction", "RAM allocation", "memory leak"]
}'''

        messages = [
            {
                "role": "system",
                "content": (
                    "You are a senior AI research librarian with deep expertise across all subfields of "
                    "machine learning, NLP, and systems AI.\n\n"
                    "Your task: Given a research topic, output a JSON object that maps the exact vocabulary "
                    "researchers use when publishing papers on this topic.\n\n"
                    "RULES:\n"
                    "1. Output ONLY valid JSON. No explanation, no markdown fences, no extra text.\n"
                    "2. The JSON must have EXACTLY these keys:\n"
                    "   - topic_summary (string): One precise sentence describing the topic.\n"
                    "   - core_terms (list of strings): 3-8 essential 1-3 word research concepts.\n"
                    "   - named_frameworks (list of strings): Specific named systems/models (e.g. 'MemGPT', 'AutoGen'). "
                    "These appear verbatim in paper titles. Include at least 3 if they exist.\n"
                    "   - task_types (list of strings): Concrete problems this research solves.\n"
                    "   - benchmark_datasets (list of strings): Known benchmark datasets or evals.\n"
                    "   - methods (list of strings): Specific algorithms or architectural patterns.\n"
                    "   - synonyms (list of strings): Alternative phrasings for the topic.\n"
                    "   - negative_terms (list of strings): Generic CS/OS terms that would pollute search "
                    "results (e.g. 'memory management', 'cache eviction', 'scheduling').\n\n"
                    f"Example output for topic 'memory architectures in AI agents':\n{example_json}"
                )
            },
            {
                "role": "user",
                "content": (
                    f"Research Topic: {topic}\n\n"
                    "Output the JSON ontology for this topic now. Start your response with {{ and end with }}."
                )
            }
        ]

        import json as _json

        for attempt in range(3):
            try:
                # Plain text response — no schema_model to avoid confusion
                response = await gateway.generate(
                    task="keyword_generation",
                    messages=messages,
                    temperature=0.1,
                )

                raw_text = response.text.strip()

                # Extract JSON block if wrapped in markdown fences
                if "```" in raw_text:
                    import re
                    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw_text, re.DOTALL)
                    if match:
                        raw_text = match.group(1)

                # Find the outermost JSON object
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                if start != -1 and end > start:
                    raw_text = raw_text[start:end]

                parsed = _json.loads(raw_text)
                ontology = ResearchOntology.model_validate(parsed)

                logger.success(
                    f"Research Ontology generated for '{topic}':\n"
                    f"  Core terms ({len(ontology.core_terms)}): {ontology.core_terms}\n"
                    f"  Named frameworks ({len(ontology.named_frameworks)}): {ontology.named_frameworks}\n"
                    f"  Task types ({len(ontology.task_types)}): {ontology.task_types}\n"
                    f"  Datasets ({len(ontology.benchmark_datasets)}): {ontology.benchmark_datasets}\n"
                    f"  Synonyms ({len(ontology.synonyms)}): {ontology.synonyms}\n"
                    f"  Negative terms ({len(ontology.negative_terms)}): {ontology.negative_terms}"
                )
                return ontology

            except Exception as e:
                logger.warning(f"Ontology generation attempt {attempt + 1}/3 failed: {e}")
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(1.5)

        logger.error(f"All 3 ontology generation attempts failed for '{topic}'. Using minimal fallback.")

        # Minimal fallback — ensures pipeline always continues
        topic_words = topic.lower().split()
        return ResearchOntology(
            topic_summary=f"Research on {topic}",
            core_terms=[topic, f"{topic} survey", f"{topic} deep learning"][:3],
            named_frameworks=[],
            task_types=[],
            benchmark_datasets=[],
            methods=[],
            synonyms=[f"{topic} neural network", f"{topic} language model"],
            negative_terms=[]
        )



research_ontology_agent = ResearchOntologyAgent()
