"""
Inspect Neo4j knowledge graph in detail.

Usage:
  python -u test_knowledge_graph.py
  python -u test_knowledge_graph.py 2412.12881v1
  python -u test_knowledge_graph.py --extract 2412.12881v1 "Graph RAG for retrievals"
  python -u test_knowledge_graph.py --topic "Graph RAG for retrievals"
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def _print(title: str) -> None:
    print("\n" + "=" * 64)
    print(title)
    print("=" * 64)


def _neo4j():
    from src.db.neo4j_client import neo4j_client
    return neo4j_client


def run_cypher(query: str, **params) -> List[Dict[str, Any]]:
    client = _neo4j()
    if not getattr(client, "driver", None) and not getattr(client, "is_connected", lambda: False)():
        raise RuntimeError("Neo4j not connected")

    rows = []
    with client.driver.session() as session:
        result = session.run(query, **params)
        for rec in result:
            rows.append(dict(rec))
    return rows


def show_connection() -> None:
    _print("1. Connection")
    client = _neo4j()
    ok = False
    try:
        if hasattr(client, "is_connected"):
            ok = client.is_connected()
        else:
            ok = client.driver is not None
        print(f"  connected: {ok}")
        if not ok:
            print("  Start Neo4j Desktop / local DB (bolt://localhost:7687) and check .env")
            return
        # server info if available
        with client.driver.session() as session:
            rec = session.run("CALL dbms.components() YIELD name, versions, edition RETURN name, versions, edition").single()
            if rec:
                print(f"  server: {rec['name']} {rec['versions']} ({rec['edition']})")
    except Exception as e:
        print(f"  error: {e}")


def show_counts() -> None:
    _print("2. Global counts")
    try:
        rows = run_cypher(
            """
            MATCH (n)
            WITH labels(n) AS labs, count(*) AS c
            UNWIND labs AS lab
            RETURN lab AS label, sum(c) AS count
            ORDER BY count DESC
            """
        )
        if not rows:
            # fallback simpler
            total = run_cypher("MATCH (n) RETURN count(n) AS nodes")
            rels = run_cypher("MATCH ()-[r]->() RETURN count(r) AS rels")
            print(f"  nodes: {total[0].get('nodes') if total else 0}")
            print(f"  rels:  {rels[0].get('rels') if rels else 0}")
        else:
            for r in rows:
                print(f"  {r.get('label')}: {r.get('count')}")
        rel_types = run_cypher(
            """
            MATCH ()-[r]->()
            RETURN type(r) AS rel, count(*) AS c
            ORDER BY c DESC
            LIMIT 20
            """
        )
        print("\n  relationship types:")
        for r in rel_types:
            print(f"    -[{r.get('rel')}]->  {r.get('c')}")
    except Exception as e:
        print(f"  failed: {e}")


def show_paper(paper_id: str) -> None:
    _print(f"3. Paper node: {paper_id}")
    try:
        rows = run_cypher(
            """
            MATCH (p)
            WHERE p.arxiv_id = $pid OR p.paper_id = $pid OR p.id = $pid
            RETURN labels(p) AS labels, properties(p) AS props
            LIMIT 5
            """,
            pid=paper_id,
        )
        if not rows:
            print("  (no paper node found — graph extract may not have run yet)")
            return
        for r in rows:
            print(f"  labels: {r.get('labels')}")
            print(f"  props:  {json.dumps(r.get('props') or {}, indent=2, default=str)}")
    except Exception as e:
        print(f"  failed: {e}")


def show_entities_for_paper(paper_id: str) -> None:
    _print(f"4. Entities linked to paper {paper_id}")
    queries = [
        """
        MATCH (p)-[r]->(e)
        WHERE p.arxiv_id = $pid OR p.paper_id = $pid
        RETURN type(r) AS rel, labels(e) AS labels, e.name AS name,
               e.type AS type, properties(e) AS props
        LIMIT 50
        """,
        """
        MATCH (e)-[r]->(p)
        WHERE p.arxiv_id = $pid OR p.paper_id = $pid
        RETURN type(r) AS rel, labels(e) AS labels, e.name AS name,
               e.type AS type, properties(e) AS props
        LIMIT 50
        """,
    ]
    found = False
    for q in queries:
        try:
            rows = run_cypher(q, pid=paper_id)
            for r in rows:
                found = True
                print(
                    f"  ({r.get('name')}) type={r.get('type') or r.get('labels')} "
                    f"via -[{r.get('rel')}]->"
                )
                props = r.get("props") or {}
                if props.get("description"):
                    print(f"      desc: {str(props['description'])[:120]}")
        except Exception as e:
            print(f"  query note: {e}")
    if not found:
        print("  (no MENTIONS / paper–entity edges — listing sample entities globally)")
        try:
            rows = run_cypher(
                """
                MATCH (e)
                WHERE e.name IS NOT NULL
                RETURN labels(e) AS labels, e.name AS name, e.type AS type
                LIMIT 30
                """
            )
            for r in rows:
                print(f"  • {r.get('name')}  [{r.get('type') or r.get('labels')}]")
        except Exception as e:
            print(f"  failed: {e}")


def show_relationships(limit: int = 40) -> None:
    _print(f"5. Sample relationships (limit {limit})")
    try:
        rows = run_cypher(
            """
            MATCH (s)-[r]->(t)
            RETURN s.name AS source, type(r) AS rel, t.name AS target,
                   r.value AS value, labels(s)[0] AS s_label, labels(t)[0] AS t_label
            LIMIT $limit
            """,
            limit=limit,
        )
        if not rows:
            print("  (graph empty)")
            return
        for r in rows:
            val = f" value={r.get('value')}" if r.get("value") is not None else ""
            print(
                f"  ({r.get('source')}:{r.get('s_label')}) "
                f"-[{r.get('rel')}{val}]-> "
                f"({r.get('target')}:{r.get('t_label')})"
            )
    except Exception as e:
        print(f"  failed: {e}")


def show_triplets_api(entity_names: Optional[List[str]] = None) -> None:
    _print("6. get_related_triplets() (chat Graph-RAG helper)")
    client = _neo4j()
    if not hasattr(client, "get_related_triplets"):
        print("  neo4j_client has no get_related_triplets")
        return
    names = entity_names or []
    if not names:
        # pick a few names from DB
        try:
            rows = run_cypher(
                """
                MATCH (e)
                WHERE e.name IS NOT NULL
                RETURN e.name AS name
                LIMIT 8
                """
            )
            names = [r["name"] for r in rows if r.get("name")]
        except Exception:
            names = []
    print(f"  seed entities: {names}")
    try:
        triplets = client.get_related_triplets(names)
        print(f"  triplet count: {len(triplets)}")
        for t in triplets[:25]:
            print(f"    {t}")
    except Exception as e:
        print(f"  failed: {e}")


async def run_live_extract(paper_id: str, topic: str) -> None:
    """Optional: run ExtractorAgent on local PDF/text and show structured output."""
    _print(f"7. Live ExtractorAgent → {paper_id}")
    from src.agents.extractor_agent import ExtractorAgent

    # Try artifact full text first
    full_text = ""
    title = paper_id
    try:
        from src.config import settings

        # common artifact layouts
        candidates = list(Path(getattr(settings, "base_dir", Path("."))).rglob(f"*{paper_id}*"))
        for p in candidates:
            if p.suffix == ".json" and "extract" in p.name.lower():
                data = json.loads(p.read_text(encoding="utf-8"))
                full_text = data.get("full_text") or data.get("markdown") or ""
                if full_text:
                    print(f"  loaded text from {p} ({len(full_text)} chars)")
                    break
    except Exception as e:
        print(f"  artifact load skip: {e}")

    if len(full_text) < 200:
        print("  No local extracted text — pass a paper you already ingested, or extend this to call pdf_tools.")
        print("  Skipping live extract.")
        return

    agent = ExtractorAgent()
    gk = await agent.extract(paper_id=paper_id, title=title, full_text=full_text[:12000])
    print(f"\n  entities ({len(gk.entities)}):")
    for e in gk.entities:
        print(f"    - {e.name:40}  type={e.type}  desc={getattr(e, 'description', None)}")
    print(f"\n  relationships ({len(gk.relationships)}):")
    for rel in gk.relationships:
        print(
            f"    - ({rel.source}) -[{rel.relation}"
            f"{(' ' + str(rel.value)) if rel.value else ''}]-> ({rel.target})"
        )

    # write if connected
    client = _neo4j()
    if hasattr(client, "write_extracted_graph"):
        try:
            client.write_extracted_graph(paper_id, title, gk)
            print("\n  wrote graph to Neo4j via write_extracted_graph")
        except Exception as e:
            print(f"\n  write failed: {e}")
    elif hasattr(client, "is_connected") and client.is_connected():
        print("\n  (no write_extracted_graph helper — inspect extractor return only)")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    flags = {a for a in sys.argv[1:] if a.startswith("--")}

    paper_id = args[0] if args else None
    topic = " ".join(args[1:]) if len(args) > 1 else "research"

    show_connection()
    show_counts()
    show_relationships(limit=40)

    if paper_id and paper_id != "--topic":
        show_paper(paper_id)
        show_entities_for_paper(paper_id)

    show_triplets_api()

    if "--extract" in flags and paper_id:
        asyncio.run(run_live_extract(paper_id, topic))

    _print("Done")
    print(
        "Tip: open Neo4j Browser →\n"
        "  MATCH (s)-[r]->(t) RETURN s,r,t LIMIT 50\n"
        "for an interactive view."
    )


if __name__ == "__main__":
    main()