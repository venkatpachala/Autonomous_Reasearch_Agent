"""
CLI Chat with your Research Memory
"""

import asyncio
import argparse
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from loguru import logger

from src.agents.query_agent import query_agent

console = Console()


async def chat_loop(topic: str = None):
    console.print(Panel.fit(
        "[bold cyan]Research Memory Chat[/bold cyan]\n"
        "Ask anything about the papers you have ingested.\n"
        "Type 'exit', 'quit' or 'q' to leave.\n"
        f"Topic filter: [yellow]{topic or 'None (search all)'}[/yellow]",
        title="Agentic Research Agent"
    ))

    while True:
        try:
            question = console.input("\n[bold green]You > [/bold green]").strip()
        except (EOFError, KeyboardInterrupt):
            console.print("\nGoodbye!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit", "q"}:
            console.print("[bold]Bye![/bold]")
            break

        console.print("[dim]Thinking...[/dim]")

        try:
            result = await query_agent.answer(question, topic=topic)

            console.print("\n")
            console.print(Panel(
                Markdown(result["answer"]),
                title="[bold blue]Research Agent[/bold blue]",
                border_style="blue"
            ))

            if result["sources"]:
                table = Table(title="Sources Used", show_header=True, header_style="bold magenta")
                table.add_column("arXiv ID", style="cyan")
                table.add_column("Title")
                table.add_column("Score", justify="right")
                table.add_column("Link")

                for src in result["sources"]:
                    score = f"{src['score']:.3f}" if src.get("score") is not None else "-"
                    table.add_row(
                        src["paper_id"],
                        (src["title"][:65] + "...") if len(src["title"]) > 65 else src["title"],
                        score,
                        src["arxiv_url"]
                    )
                console.print(table)
                console.print(f"\n[dim]Used {result['contexts_used']} papers from your knowledge base.[/dim]")

        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.exception("Chat error")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, default=None)
    args = parser.parse_args()
    asyncio.run(chat_loop(topic=args.topic))


if __name__ == "__main__":
    main()