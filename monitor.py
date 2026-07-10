"""
Continuous Monitor Runner
=========================
Usage examples:

# Run once right now (check all known topics)
PYTHONPATH=. python monitor.py

# Run once for specific topics
PYTHONPATH=. python monitor.py --topics "agentic RAG memory systems" "efficient edge LLMs"

# Run as a long-lived background process (checks every 6 hours)
PYTHONPATH=. python monitor.py --daemon --interval 6
"""

import asyncio
import argparse
from datetime import datetime
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from src.agents.monitor_agent import monitor_agent
from src.tools.research_index import research_index


console = Console()


async def run_once(topics: list[str] | None = None):
    console.print(Panel.fit(
        f"[bold cyan]Continuous Monitor Agent[/bold cyan]\n"
        f"Checking arXiv for new papers...\n"
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        title="Research Memory"
    ))

    results = await monitor_agent.run_once(topics=topics)

    if not results:
        console.print("[yellow]No topics to monitor or no new papers found.[/yellow]")
        return

    # Pretty summary table
    table = Table(title="Monitor Results", show_header=True, header_style="bold magenta")
    table.add_column("Topic")
    table.add_column("New Found", justify="right")
    table.add_column("Ingested", justify="right", style="green")
    table.add_column("Failed", justify="right", style="red")
    table.add_column("Paper IDs")

    for r in results:
        paper_ids = ", ".join(r["paper_ids"][:3])
        if len(r["paper_ids"]) > 3:
            paper_ids += f" (+{len(r['paper_ids'])-3} more)"
        table.add_row(
            r["topic"][:40],
            str(r["new_papers_found"]),
            str(r["successfully_ingested"]),
            str(r["failed"]),
            paper_ids or "-"
        )

    console.print(table)

    # Index stats
    stats = research_index.stats()
    console.print(f"\n[dim]Total papers in knowledge base: {stats['total_papers']}[/dim]")
    console.print(f"[dim]Total topics tracked: {stats['total_topics']}[/dim]")


async def run_daemon(interval_hours: float = 6.0, topics: list[str] | None = None):
    console.print(Panel.fit(
        f"[bold green]Daemon Mode Started[/bold green]\n"
        f"Checking every {interval_hours} hours\n"
        f"Press Ctrl+C to stop",
        title="Continuous Monitor"
    ))

    while True:
        try:
            await run_once(topics=topics)
            sleep_seconds = interval_hours * 3600
            console.print(f"\n[dim]Sleeping for {interval_hours} hours until next check...[/dim]\n")
            await asyncio.sleep(sleep_seconds)
        except KeyboardInterrupt:
            console.print("\n[bold]Monitor stopped by user.[/bold]")
            break
        except Exception as e:
            logger.exception(f"Error in daemon loop: {e}")
            console.print(f"[red]Error occurred. Retrying in 30 minutes...[/red]")
            await asyncio.sleep(1800)


def main():
    parser = argparse.ArgumentParser(description="Continuous Monitor Agent for Research Memory")
    parser.add_argument(
        "--topics",
        nargs="+",
        help="Specific topics to monitor (default: all known topics)"
    )
    parser.add_argument(
        "--daemon",
        action="store_true",
        help="Run continuously in the background"
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=6.0,
        help="Hours between checks when running as daemon (default: 6)"
    )
    args = parser.parse_args()

    if args.daemon:
        asyncio.run(run_daemon(interval_hours=args.interval, topics=args.topics))
    else:
        asyncio.run(run_once(topics=args.topics))


if __name__ == "__main__":
    main()

