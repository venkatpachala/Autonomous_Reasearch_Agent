"""
Stage 3: PyMuPDF-first vs LlamaParse path timing.
Run:  python test_parse_fastpath.py
      python test_parse_fastpath.py path/to/paper.pdf
"""

import asyncio
import sys
import time
from pathlib import Path

from loguru import logger


async def time_extract(pdf_path: Path):
    from src.tools.pdf_tools import pdf_tools

    print(f"\n{'='*60}")
    print(f"PDF: {pdf_path}")
    print(f"Size: {pdf_path.stat().st_size / 1024:.1f} KB")
    print(f"{'='*60}")

    t0 = time.perf_counter()
    extracted = await pdf_tools.extract_content(pdf_path)
    elapsed = time.perf_counter() - t0

    text = getattr(extracted, "full_text", "") or ""
    sections = getattr(extracted, "sections", None) or {}
    tables = getattr(extracted, "tables", None) or []

    print(f"\nTime           : {elapsed:.2f}s")
    print(f"Text length    : {len(text)} chars")
    print(f"Sections       : {len(sections)}")
    print(f"Tables         : {len(tables) if isinstance(tables, list) else 'n/a'}")
    print(f"Preview        : {text[:200].replace(chr(10), ' ')!r}...")

    # Heuristic: < 8s and long text ≈ PyMuPDF path
    if elapsed < 8 and len(text) > 1500:
        print("Likely path    : PyMuPDF (fast)")
    elif elapsed >= 15:
        print("Likely path    : LlamaParse (slow) or network")
    else:
        print("Likely path    : mixed / borderline")

    return elapsed, len(text)


async def main():
    # Explicit path argument
    if len(sys.argv) > 1:
        pdfs = [Path(sys.argv[1])]
    else:
        # Auto-find a few existing papers under papers/
        root = Path("papers")
        pdfs = sorted(root.rglob("*.pdf"))[:3] if root.exists() else []

    if not pdfs:
        print(
            "No PDFs found. Usage:\n"
            "  python test_parse_fastpath.py path/to/file.pdf\n"
            "Or put PDFs under ./papers/"
        )
        return

    results = []
    for p in pdfs:
        if not p.exists():
            print(f"Missing: {p}")
            continue
        try:
            elapsed, nchars = await time_extract(p)
            results.append((p.name, elapsed, nchars))
        except Exception as e:
            logger.exception(f"Failed on {p}: {e}")

    if results:
        print(f"\n{'='*60}")
        print("SUMMARY")
        print(f"{'='*60}")
        for name, elapsed, nchars in results:
            tag = "FAST" if elapsed < 8 and nchars > 1500 else "SLOW/CHECK"
            print(f"  [{tag}] {elapsed:6.2f}s | {nchars:6d} chars | {name}")
        avg = sum(r[1] for r in results) / len(results)
        print(f"\nAvg extract time: {avg:.2f}s over {len(results)} file(s)")


if __name__ == "__main__":
    asyncio.run(main())