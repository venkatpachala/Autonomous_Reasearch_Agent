DECOMPOSER_PROMPT = """Break down the following research topic into 3-5 clear, specific sub-questions that need to be answered for a comprehensive literature review.

Topic: {topic}

Return only the sub-questions, one per line, numbered 1 to 5."""

SUMMARIZER_PROMPT = """Summarize the following paper in 3-4 concise sentences. Focus on:
- Key methods
- Main findings
- Limitations

Title: {title}
Abstract: {abstract}"""

CRITIC_PROMPT = """Critically analyze this paper summary. Point out:
- Potential limitations
- Open questions
- Methodological weaknesses
- Possible biases

Summary: {summary}"""

SYNTHESIZER_PROMPT = """Write a high-quality, well-structured literature review based on the information below.

Topic: {topic}

Sub-questions:
{sub_questions}

Paper Summaries:
{summaries}

Critiques:
{critiques}

Structure the review with clear sections and proper academic tone."""