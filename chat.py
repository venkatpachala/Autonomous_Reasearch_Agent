"""
Session-Aware Research Memory Chat
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

from src.agents.session_manager import session_manager
from src.agents.query_agent import query_agent
from src.models.session import ResearchSession


console = Console()


def print_sessions(sessions: list[ResearchSession]):
    if not sessions:
        console.print("[dim]No previous research sessions found.[/dim]")
        return

    table = Table(title="Your Research Sessions", show_header=True, header_style="bold magenta")
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
            str(len(s.conversation))
        )
    console.print(table)


async def start_session_flow() -> ResearchSession:
    """Interactive session selection / creation"""
    sessions = session_manager.list_sessions()

    console.print(Panel.fit(
        "[bold cyan]Research Memory — Session Manager[/bold cyan]\n"
        "Each session is scoped to one research topic.",
        title="Agentic Research Agent"
    ))

    print_sessions(sessions)

    console.print("\n[bold]Options:[/bold]")
    console.print("  • Type a number to resume an existing session")
    console.print("  • Type a new topic name to create a fresh session")
    console.print("  • Type 'q' to quit")

    choice = Prompt.ask("\n[bold green]Your choice[/bold green]").strip()

    if choice.lower() in {"q", "quit", "exit"}:
        console.print("Goodbye!")
        raise SystemExit(0)

    # Resume existing
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(sessions):
            session = sessions[idx]
            session_manager.current_session = session
            console.print(f"\n[green]✓ Resumed session[/green] {session.session_id} — [bold]{session.topic}[/bold]")
            return session
        else:
            console.print("[red]Invalid number[/red]")
            return await start_session_flow()

    # Create new session
    topic = choice
    console.print(f"\nCreating new research session for topic: [bold yellow]{topic}[/bold yellow]")

    session = session_manager.create_session(topic)

    # Ask if we should ingest papers now
    if Confirm.ask("Do you want to ingest papers for this topic now?", default=True):
        with console.status("[bold green]Ingesting papers from arXiv... this may take a few minutes[/bold green]"):
            session = await session_manager.ensure_papers_ingested(session)
        console.print(f"[green]✓ Ingested {len(session.papers_ingested)} papers[/green]")
    else:
        console.print("[yellow]You can chat later, but answers will be limited until papers are ingested.[/yellow]")

    return session


async def chat_loop(session: ResearchSession):
    """Main chat loop scoped to the session topic"""
    console.print(Panel.fit(
        f"[bold]Active Session[/bold]: {session.session_id}\n"
        f"[bold]Topic[/bold]: {session.topic}\n"
        f"[bold]Papers in memory[/bold]: {len(session.papers_ingested)}\n\n"
        "Ask anything about this research topic.\n"
        "Commands: /history  /papers  /ingest  /exit",
        title="Research Memory Chat",
        border_style="cyan"
    ))

    while True:
        try:
            question = console.input("\n[bold green]You > [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\n[bold]Session saved. Goodbye![/bold]")
            break

        if not question:
            continue

        # Special commands
        if question.lower() in {"/exit", "/quit", "exit", "quit", "q"}:
            console.print("[bold]Session saved. Bye![/bold]")
            break

        if question.lower() == "/history":
            if not session.conversation:
                console.print("[dim]No conversation yet.[/dim]")
            else:
                for msg in session.conversation[-10:]:  # last 10
                    role = "[bold blue]Assistant[/bold blue]" if msg.role == "assistant" else "[bold green]You[/bold green]"
                    console.print(f"{role}: {msg.content[:200]}...")
            continue

        if question.lower() == "/papers":
            console.print(f"Papers in this session ({len(session.papers_ingested)}):")
            for pid in session.papers_ingested:
                console.print(f"  • {pid}  →  https://arxiv.org/abs/{pid}")
            continue

        if question.lower() == "/ingest":
            with console.status("[bold green]Re-running ingestion...[/bold green]"):
                session = await session_manager.ensure_papers_ingested(session, force=True)
            console.print(f"[green]✓ Now have {len(session.papers_ingested)} papers[/green]")
            continue

        # Normal question
        console.print("[dim]Thinking...[/dim]")

        # Save user message
        session_manager.add_message("user", question)

        try:
            # Important: always filter by the session topic
            result = await query_agent.answer(question, topic=session.topic)

            # Pretty answer
            console.print("\n")
            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold blue]Research Agent[/bold blue]",
                border_style="blue"
            ))

            # Sources
            if result["sources"]:
                table = Table(title="Sources Used", show_header=True, header_style="bold magenta")
                table.add_column("arXiv ID", style="cyan")
                table.add_column("Title")
                table.add_column("Score", justify="right")
                table.add_column("Link")

                for src in result["sources"]:
                    score = f"{src['score']:.3f}" if src.get("score") is not None else "-"
                    title = src["title"]
                    if len(title) > 60:
                        title = title[:57] + "..."
                    table.add_row(src["paper_id"], title, score, src["arxiv_url"])
                console.print(table)
                console.print(f"[dim]Used {result['contexts_used']} papers from this session's knowledge.[/dim]")

            # Save assistant message
            session_manager.add_message("assistant", result["answer"], sources=result["sources"])

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.exception("Chat error")


async def main():
    parser = argparse.ArgumentParser(description="Session-aware Research Memory Chat")
    parser.add_argument("--topic", type=str, help="Directly start with this topic")
    parser.add_argument("--session", type=str, help="Resume a specific session_id")
    args = parser.parse_args()

    if args.session:
        session = session_manager.load_session(args.session)
        if not session:
            console.print(f"[red]Session {args.session} not found[/red]")
            return
        console.print(f"[green]Resumed session {session.session_id}[/green]")
    elif args.topic:
        session = session_manager.get_or_create_session(args.topic)
        if not session.papers_ingested:
            if Confirm.ask(f"No papers yet for '{args.topic}'. Ingest now?", default=True):
                with console.status("[bold green]Ingesting...[/bold green]"):
                    session = await session_manager.ensure_papers_ingested(session)
    else:
        session = await start_session_flow()

    await chat_loop(session)


if __name__ == "__main__":
    asyncio.run(main())

