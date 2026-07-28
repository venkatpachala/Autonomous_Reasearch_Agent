"""
Session-Aware Helix Research Chat
==================================
Flow:
1. Show existing sessions or create a new one
2. For a new / empty topic → run ingestion automatically
3. Enter chat mode scoped to that topic
"""

import asyncio
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from loguru import logger
from typing import List,Dict,Any

from src.agents.session_manager import session_manager
from src.agents.query_agent import query_agent
from src.models.session import ResearchSession

from src.observability.startup import init_observability

init_observability(project_name="research-agent")

console = Console()


def _warm_reranker() -> None:
    """Load cross-encoder once so the first chat query is not cold."""
    try:
        from src.tools.reranker import _get_model

        _get_model()
        logger.info("Reranker warmed")
    except Exception as e:
        logger.warning(f"Reranker warmup skipped: {e}")


def print_sessions(sessions: list[ResearchSession]):
    if not sessions:
        console.print("[dim]No previous research sessions found.[/dim]")
        return

    table = Table(
        title="Your Research Sessions",
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("#", style="cyan", width=4)
    table.add_column("Session ID", style="green")
    table.add_column("Topic")
    table.add_column("Papers", justify="right")
    table.add_column("Last Active")
    table.add_column("Messages", justify="right")

    for i, s in enumerate(sessions, 1):
        table.add_row(
            str(i),
            s.session_id,
            s.topic[:50] + ("..." if len(s.topic) > 50 else ""),
            str(len(s.papers_ingested)),
            s.last_active.strftime("%Y-%m-%d %H:%M"),
            str(len(s.conversation)),
        )
    console.print(table)


async def start_session_flow() -> ResearchSession:
    """Interactive session selection / creation."""
    sessions = session_manager.list_sessions()

    console.print(
        Panel.fit(
            "[bold cyan]Helix Research — Session Manager[/bold cyan]\n"
            "Each session is scoped to one research topic.",
            title="Helix Research",
        )
    )

    print_sessions(sessions)

    console.print("\n[bold]Options:[/bold]")
    console.print("  • Type a number to resume an existing session")
    console.print("  • Type a new topic name to create a fresh session")
    console.print("  • Type 'q' to quit")

    choice = Prompt.ask("\n[bold green]Your choice[/bold green]").strip()

    if choice.lower() in {"q", "quit", "exit"}:
        console.print("Goodbye!")
        raise SystemExit(0)

    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            session = sessions[idx]
            session_manager.current_session = session
            console.print(
                f"\n[green]✓ Resumed session[/green] {session.session_id} — "
                f"[bold]{session.topic}[/bold]"
            )
            return session
        console.print("[red]Invalid number[/red]")
        return await start_session_flow()

    topic = choice
    console.print(
        f"\nCreating new research session for topic: [bold yellow]{topic}[/bold yellow]"
    )

    session = session_manager.create_session(topic)

    if Confirm.ask("Do you want to ingest papers for this topic now?", default=True):
        with console.status(
            "[bold green]Ingesting papers from arXiv... this may take a few minutes[/bold green]"
        ):
            session = await session_manager.ensure_papers_ingested(session)
        console.print(
            f"[green]✓ Ingested {len(session.papers_ingested)} papers[/green]"
        )
    else:
        console.print(
            "[yellow]You can chat later, but answers will be limited until papers are ingested.[/yellow]"
        )

    return session


async def chat_loop(session: ResearchSession):
    """Main chat loop scoped to the session topic."""
    console.print(
        Panel.fit(
            f"[bold]Active Session[/bold]: {session.session_id}\n"
            f"[bold]Topic[/bold]: {session.topic}\n"
            f"[bold]Papers in memory[/bold]: {len(session.papers_ingested)}\n\n"
            "Ask anything about this research topic.\n"
            "Commands: /history  /papers  /ingest  /exit",
            title="Helix Research Chat",
            border_style="cyan",
        )
    )

    def _dedupe_sources(sources: List[Dict]) -> List[Dict]:
        best = {}
        for s in sources or []:
            pid = s.get("paper_id")
            if not pid:
                continue
            if pid not in best or (s.get("score") or 0) > (best[pid].get("score") or 0):
                best[pid] = s
        return list(best.values())

    def _print_sources(result: dict) -> None:
        sources = _dedupe_sources(result.get("sources") or [])
        if not sources:
            return
        table = Table(
            title="Sources Used",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("arXiv ID", style="cyan")
        table.add_column("Title")
        table.add_column("Score", justify="right")
        table.add_column("Link")
        for src in sources:
            score = src.get("score")
            score_s = f"{score:.3f}" if isinstance(score, (int, float)) else "-"
            title = src.get("title") or "Untitled"
            if len(title) > 60:
                title = title[:57] + "..."
            table.add_row(
                str(src.get("paper_id") or ""),
                title,
                score_s,
                str(src.get("arxiv_url") or ""),
            )
        console.print(table)
        console.print(
            f"[dim]Used {result.get('contexts_used', len(sources))} "
            f"papers from this session's knowledge.[/dim]"
        )

    def _print_latency(result: dict) -> None:
        lat = result.get("latency_ms") or {}
        if not lat:
            return
        console.print(
            f"[dim]Latency: total={lat.get('total_ms', 0):.0f}ms "
            f"(intent={lat.get('intent_ms', 0):.0f} "
            f"retrieve={lat.get('retrieve_ms', 0):.0f} "
            f"llm={lat.get('llm_ms', 0):.0f} "
            f"synth={lat.get('synth_ms', 0):.0f})[/dim]"
        )

    while True:
        try:
            question = console.input("\n[bold green]You > [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]Session saved. Goodbye![/bold]")
            break

        if not question:
            continue

        if question.lower() in {"/exit", "/quit", "exit", "quit", "q"}:
            console.print("[bold]Session saved. Bye![/bold]")
            break

        if question.lower() == "/history":
            if not session.conversation:
                console.print("[dim]No conversation yet.[/dim]")
            else:
                for msg in session.conversation[-10:]:
                    role = (
                        "[bold blue]Assistant[/bold blue]"
                        if msg.role == "assistant"
                        else "[bold green]You[/bold green]"
                    )
                    console.print(f"{role}: {msg.content[:200]}...")
            continue

        if question.lower() == "/papers":
            console.print(
                f"Papers in this session ({len(session.papers_ingested)}):"
            )
            for pid in session.papers_ingested:
                console.print(f"  • {pid}  →  https://arxiv.org/abs/{pid}")
            continue

        if question.lower() == "/ingest":
            with console.status("[bold green]Re-running ingestion...[/bold green]"):
                session = await session_manager.ensure_papers_ingested(
                    session, force=True
                )
            console.print(
                f"[green]✓ Now have {len(session.papers_ingested)} papers[/green]"
            )
            continue

        console.print("[dim]Thinking...[/dim]")
        session_manager.add_message("user", question)

        try:
            result = await query_agent.answer(question, topic=session.topic)

            if not result or not isinstance(result, dict):
                console.print(
                    "[bold red]Error:[/bold red] Query agent returned no result."
                )
                logger.error(f"query_agent.answer returned: {result!r}")
                continue

            answer_text = result.get("answer") or "(empty answer)"
            intent = result.get("intent") or ""
            lat = result.get("latency_ms") or {}
            llm_ms = float(lat.get("llm_ms") or 0)

            # Progressive reveal for long LLM answers (better perceived latency).
            # Does not re-call the model — reveals the completed answer in chunks.
            stream_intents = {
                "paper_describe",
                "fact_lookup",
                "paper_summary",
                "comparison",
            }
            use_stream = (
                intent in stream_intents
                and llm_ms > 500
                and len(answer_text) > 400
            )

            console.print("\n")
            if use_stream:
                from rich.live import Live

                buf = ""
                chunk_size = max(16, len(answer_text) // 50)
                with Live(console=console, refresh_per_second=14) as live:
                    for i in range(0, len(answer_text), chunk_size):
                        buf = answer_text[: i + chunk_size]
                        live.update(
                            Panel(
                                Markdown(buf),
                                title="[bold blue]Research Agent[/bold blue]",
                                border_style="blue",
                            )
                        )
                        await asyncio.sleep(0.015)
            else:
                console.print(
                    Panel(
                        Markdown(answer_text),
                        title="[bold blue]Research Agent[/bold blue]",
                        border_style="blue",
                    )
                )

            _print_sources(result)
            _print_latency(result)

            session_manager.add_message(
                "assistant",
                answer_text,
                sources=_dedupe_sources(result.get("sources") or []),
            )

        except Exception as e:
            logger.exception(f"Chat error: {e}")
            console.print(f"[bold red]Error:[/bold red] {e}")

async def main():
    parser = argparse.ArgumentParser(description="Session-aware Helix Research Chat")
    parser.add_argument("--topic", type=str, help="Directly start with this topic")
    parser.add_argument("--session", type=str, help="Resume a specific session_id")
    args = parser.parse_args()

    # Warm heavy models before the first user question
    _warm_reranker()

    if args.session:
        session = session_manager.load_session(args.session)
        if not session:
            console.print(f"[red]Session {args.session} not found[/red]")
            return
        console.print(f"[green]Resumed session {session.session_id}[/green]")
    elif args.topic:
        session = session_manager.get_or_create_session(args.topic)
        if not session.papers_ingested:
            if Confirm.ask(
                f"No papers yet for '{args.topic}'. Ingest now?", default=True
            ):
                with console.status("[bold green]Ingesting...[/bold green]"):
                    session = await session_manager.ensure_papers_ingested(session)
    else:
        session = await start_session_flow()

    await chat_loop(session)


if __name__ == "__main__":
    asyncio.run(main())