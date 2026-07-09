import requests
from pathlib import Path
import tempfile
from llama_parse import LlamaParse
from src.config import settings

parser = LlamaParse(
    api_key=settings.LLAMA_CLOUD_API_KEY,
    result_type="markdown",
    num_workers=2,
    verbose=True
)

def pdf_extractor_agent(state):
    """PDF Extractor with LlamaParse"""
    extracted_docs = []
    
    for paper in state.get("retrieved_papers", []):
        pdf_url = paper.get("pdf_url")
        if not pdf_url:
            continue
            
        try:
            # Download PDF
            response = requests.get(pdf_url, timeout=30)
            response.raise_for_status()
            
            # Save to temp file
            temp_path = Path(tempfile.gettempdir()) / f"paper_{hash(pdf_url)}.pdf"
            temp_path.write_bytes(response.content)
            
            # Extract with LlamaParse
            documents = parser.load_data(str(temp_path))
            text = "\n".join([doc.text for doc in documents])
            
            extracted_docs.append({
                "paper_title": paper.get("title"),
                "extracted_text": text[:4000],
                "metadata": documents[0].metadata if documents else {}
            })
            
            print(f"✓ Extracted: {paper.get('title')[:60]}...")
            
        except Exception as e:
            print(f"✗ Failed to extract {paper.get('title')}: {e}")
    
    return {
    "extracted_docs": extracted_docs
}