"""Tavily-powered web search tool for the Trend Strategist agent.

Exposes ``search_tool``, a LangChain tool the strategist node invokes as::

    search_tool.invoke({"query": "Trending perspectives on <topic>"})

If ``TAVILY_API_KEY`` is unset (or the tavily package is missing) it returns a
deterministic mock so the graph still runs end-to-end offline.
"""

import os

from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

try:
    from tavily import TavilyClient
except ImportError:  # tavily-python not installed yet
    TavilyClient = None


def _format_results(results: list[dict]) -> str:
    lines = []
    for r in results:
        title = r.get("title", "Untitled")
        content = (r.get("content") or "").strip()
        url = r.get("url", "")
        lines.append(f"- {title}: {content} (source: {url})")
    return "\n".join(lines)


def _mock(query: str) -> str:
    return (
        f"[OFFLINE MOCK SEARCH for '{query}'] Recent discussion highlights a shift "
        "toward automation, developer efficiency, and practical, framework-driven "
        "implementation. Set TAVILY_API_KEY in your .env for live results."
    )


@tool
def search_tool(query: str) -> str:
    """Search the live web for recent, real-world context on a topic.

    Returns a concatenated digest of the top results, suitable for grounding a
    content strategy brief in current trends.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key or TavilyClient is None:
        return _mock(query)

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, search_depth="advanced", max_results=5)
        results = response.get("results", [])
    except Exception as exc:  # network / auth failure -> degrade gracefully
        return f"[SEARCH ERROR: {exc}] Falling back to mock.\n{_mock(query)}"

    if not results:
        return f"No live results found for '{query}'."
    return _format_results(results)
