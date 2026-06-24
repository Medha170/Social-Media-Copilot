"""Local RAG vector store of high-performing hook templates.

Seeds a persistent ChromaDB collection with proven, high-engagement hook
patterns and exposes ``hook_retriever_tool`` so the strategist can ground its
chosen angle in formats that have historically performed well::

    hook_retriever_tool.invoke({"query": "AI agents for developers"})

This is the project's RAG element (Min 2 Tools requirement, tool #2).
"""

import os

import chromadb
from langchain_core.tools import tool

_PERSIST_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", ".chroma_store")
)
_COLLECTION_NAME = "high_performing_hooks"

# Curated library of hook templates that have historically driven engagement.
# "<topic>" is a slot the strategist/copywriter fills in for the actual subject.
HOOK_TEMPLATES = [
    {"id": "h1", "category": "contrarian",
     "text": "Most people get <topic> completely wrong. Here's the one thing that actually matters."},
    {"id": "h2", "category": "authority",
     "text": "I spent 100 hours on <topic> so you don't have to. 5 lessons:"},
    {"id": "h3", "category": "pattern-interrupt",
     "text": "Stop scrolling. If you work with <topic>, this will save you hours every week."},
    {"id": "h4", "category": "before-after",
     "text": "<topic> a year ago vs <topic> today — the difference is wild. A thread."},
    {"id": "h5", "category": "curiosity-gap",
     "text": "Nobody talks about this side of <topic>, but it changed how I build everything."},
    {"id": "h6", "category": "contrarian",
     "text": "The fastest way to master <topic>? Ignore 90% of the advice online. Do these 3 things."},
    {"id": "h7", "category": "value-bomb",
     "text": "Here's the exact framework I use for <topic> (steal it):"},
    {"id": "h8", "category": "hot-take",
     "text": "Unpopular opinion: <topic> is easier than everyone makes it sound. Proof below."},
    {"id": "h9", "category": "roadmap",
     "text": "If I had to learn <topic> from scratch today, here's my exact 7-day plan."},
    {"id": "h10", "category": "story-lesson",
     "text": "This one mistake in <topic> cost me weeks. Don't repeat it."},
]


def _get_collection():
    """Return the hooks collection, seeding it on first use."""
    client = chromadb.PersistentClient(path=_PERSIST_DIR)
    collection = client.get_or_create_collection(name=_COLLECTION_NAME)
    if collection.count() == 0:
        collection.add(
            ids=[h["id"] for h in HOOK_TEMPLATES],
            documents=[h["text"] for h in HOOK_TEMPLATES],
            metadatas=[{"category": h["category"]} for h in HOOK_TEMPLATES],
        )
    return collection


@tool
def hook_retriever_tool(query: str) -> str:
    """Retrieve high-performing hook templates relevant to a topic via RAG.

    Returns the top 3 matching hook patterns (with their category) so the
    strategist can model the brief's hook angle on a proven format.
    """
    collection = _get_collection()
    results = collection.query(query_texts=[query], n_results=3)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    if not docs:
        return "No matching hook templates found."
    lines = []
    for doc, meta in zip(docs, metas):
        category = (meta or {}).get("category", "general")
        lines.append(f"[{category}] {doc}")
    return "\n".join(lines)


if __name__ == "__main__":  # quick smoke test
    print(hook_retriever_tool.invoke({"query": "ai agents for developers"}))
