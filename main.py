from src.graph.phase1_workflow import build_phase1_workflow
from dotenv import load_dotenv
import json
from pathlib import Path

load_dotenv()

workflow = build_phase1_workflow()

initial_state = {
    "topic": "Efficient on-device agentic AI with long-term memory and thermal optimization",
    "sub_questions": [],
    "retrieved_papers": [],
    "extracted_docs": [],
    "summaries": [],
    "critiques": [],
    "final_literature_review": "",
    "memory_notes": []
}

print("Starting ResearchForge Phase 1...\n")
result = workflow.invoke(initial_state)
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

output_file = output_dir / f"review_{hash(initial_state['topic'])}.md"
output_file.write_text(result["final_literature_review"], encoding="utf-8")

print("\n" + "="*80)
print("FINAL LITERATURE REVIEW")
print("="*80)
print(result["final_literature_review"])

print("\nMemory Notes:")
for note in result["memory_notes"]:
    print(f"- {note}")

print(f"\nReview saved to: {output_file}")