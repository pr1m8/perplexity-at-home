"""Command-line entrypoint for the package."""

from __future__ import annotations

import argparse
import asyncio
import json

from perplexity_at_home.agents.deep_research.runtime import run_deep_research
from perplexity_at_home.agents.pro_search.runtime import run_pro_search
from perplexity_at_home.agents.quick_search.runtime import run_quick_search
from perplexity_at_home.core.persistence import setup_persistence
from perplexity_at_home.dashboard.launcher import launch_dashboard


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="perplexity-at-home")
    subparsers = parser.add_subparsers(dest="command", required=True)

    deep_research_parser = subparsers.add_parser(
        "deep-research",
        help="Run the top-level deep-research workflow.",
    )
    deep_research_parser.add_argument("question", help="Research question to answer.")
    deep_research_parser.add_argument(
        "--persistent",
        action="store_true",
        help="Use Postgres-backed LangGraph persistence.",
    )
    deep_research_parser.add_argument(
        "--setup-persistence",
        action="store_true",
        help="Initialize persistence tables before running.",
    )
    deep_research_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full final state as JSON instead of the final report.",
    )
    deep_research_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable LangGraph debug mode.",
    )

    pro_search_parser = subparsers.add_parser(
        "pro-search",
        help="Run the top-level pro-search workflow.",
    )
    pro_search_parser.add_argument("question", help="Question to answer.")
    pro_search_parser.add_argument(
        "--persistent",
        action="store_true",
        help="Use Postgres-backed LangGraph persistence.",
    )
    pro_search_parser.add_argument(
        "--setup-persistence",
        action="store_true",
        help="Initialize persistence tables before running.",
    )
    pro_search_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full final state as JSON instead of the final answer.",
    )
    pro_search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable LangGraph debug mode.",
    )

    quick_search_parser = subparsers.add_parser(
        "quick-search",
        help="Run the top-level quick-search workflow.",
    )
    quick_search_parser.add_argument("question", help="Question to answer.")
    quick_search_parser.add_argument(
        "--persistent",
        action="store_true",
        help="Use Postgres-backed LangGraph persistence.",
    )
    quick_search_parser.add_argument(
        "--setup-persistence",
        action="store_true",
        help="Initialize persistence tables before running.",
    )
    quick_search_parser.add_argument(
        "--json",
        action="store_true",
        help="Print the full final state as JSON instead of the final answer.",
    )
    quick_search_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable LangGraph debug mode.",
    )

    persistence_parser = subparsers.add_parser(
        "persistence",
        help="Manage Postgres-backed LangGraph persistence.",
    )
    persistence_subparsers = persistence_parser.add_subparsers(
        dest="persistence_command",
        required=True,
    )
    persistence_subparsers.add_parser(
        "setup",
        help="Create or update the persistence tables.",
    )

    dashboard_parser = subparsers.add_parser(
        "dashboard",
        help="Launch the packaged Streamlit dashboard.",
    )
    dashboard_parser.add_argument("--host", default="127.0.0.1")
    dashboard_parser.add_argument("--port", type=int, default=8501)
    dashboard_parser.add_argument(
        "--headless",
        action="store_true",
        help="Run Streamlit in headless mode.",
    )

    return parser


async def _run_deep_research_command(args: argparse.Namespace) -> int:
    result = await run_deep_research(
        args.question,
        persistent=args.persistent,
        setup_persistence=args.setup_persistence,
        debug=args.debug,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    final_answer = result.get("final_answer", {})
    report_markdown = final_answer.get("report_markdown")
    if isinstance(report_markdown, str) and report_markdown.strip():
        print(report_markdown)
        return 0

    print(json.dumps(final_answer or result, indent=2, default=str))
    return 0


async def _run_pro_search_command(args: argparse.Namespace) -> int:
    result = await run_pro_search(
        args.question,
        persistent=args.persistent,
        setup_persistence=args.setup_persistence,
        debug=args.debug,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    final_answer = result.get("final_answer", {})
    answer_markdown = final_answer.get("answer_markdown")
    if isinstance(answer_markdown, str) and answer_markdown.strip():
        print(answer_markdown)
        return 0

    print(json.dumps(final_answer or result, indent=2, default=str))
    return 0


async def _run_quick_search_command(args: argparse.Namespace) -> int:
    result = await run_quick_search(
        args.question,
        persistent=args.persistent,
        setup_persistence=args.setup_persistence,
        debug=args.debug,
    )
    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return 0

    structured = result.get("structured_response", {})
    model_dump = getattr(structured, "model_dump", None)
    payload = model_dump(mode="json") if callable(model_dump) else structured
    if not isinstance(payload, dict):
        payload = {}

    answer_markdown = payload.get("answer_markdown")
    if isinstance(answer_markdown, str) and answer_markdown.strip():
        print(answer_markdown)
        return 0

    print(json.dumps(payload or result, indent=2, default=str))
    return 0


def _run_persistence_command(args: argparse.Namespace) -> int:
    if args.persistence_command != "setup":
        raise ValueError(f"Unsupported persistence command: {args.persistence_command}")

    asyncio.run(setup_persistence())
    print("Persistence setup complete.")
    return 0


def _run_dashboard_command(args: argparse.Namespace) -> int:
    return launch_dashboard(
        host=args.host,
        port=args.port,
        headless=args.headless,
    )


def main(argv: list[str] | None = None) -> int:
    """Run the package CLI."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "deep-research":
        return asyncio.run(_run_deep_research_command(args))

    if args.command == "pro-search":
        return asyncio.run(_run_pro_search_command(args))

    if args.command == "quick-search":
        return asyncio.run(_run_quick_search_command(args))

    if args.command == "persistence":
        return _run_persistence_command(args)

    if args.command == "dashboard":
        return _run_dashboard_command(args)

    parser.error(f"Unsupported command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
