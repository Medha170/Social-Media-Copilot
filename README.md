# Multi-Agent Social Media Co-Pilot

Capstone project for Multi-Agent Orchestration.

A LangGraph multi-agent system that turns a raw topic into platform-native
LinkedIn and Instagram content — grounded by live web search and a RAG hook
library, gated by a guardrails reviewer, with a human-in-the-loop approval step.

## Architecture

```
        topic
          |
          v
  Trend Strategist  ->  Platform Copywriter  ->  Guardrails Reviewer
   (search + RAG)         (LinkedIn + IG)        (approve / revise / refuse)
                                ^                          |
                                +------- revise loop ------+
                                                           |
                                                approved / refused -> END
```

- **Agent 1 — Trend Strategist** (`core/agents/strategist.py`): turns the topic +
  live trends into a structured `StrategyBrief`.
- **Agent 2 — Platform Copywriter** (`core/agents/copywriter.py`): writes distinct
  LinkedIn and Instagram drafts; rewrites on reviewer feedback.
- **Agent 3 — Guardrails Reviewer** (`infrastructure/validation/reviewer.py`):
  validates drafts and routes `approved` / `revise` / `refused`.

LLM: local **Ollama `llama3`**. Structured output via **Pydantic**.

## Project structure

```
social-media-copilot/
├── core/                       <-- Lead Agent Architect (teammate)
│   ├── graph.py                    # StateGraph definition & compilation (WIP)
│   ├── states.py                   # AgentState + StrategyBrief / ReviewResult
│   └── agents/
│       ├── strategist.py
│       └── copywriter.py
├── infrastructure/             <-- System & Reliability Engineer
│   ├── tools/
│   │   ├── search_tool.py          # Tavily web search
│   │   └── vector_rag.py           # ChromaDB hook-template RAG
│   ├── validation/
│   │   └── reviewer.py             # Guardrails reviewer + router
│   ├── evaluation/
│   │   └── test_cases.py           # 5 evaluation scenarios
│   └── observability.py            # LangSmith tracing setup
├── .env.example                    # copy to .env (never commit .env)
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
cp .env.example .env            # then fill in your keys
ollama pull llama3.1            # local LLM (install Ollama first)
```

Keys used (see `.env.example`):
- `TAVILY_API_KEY` — live web search (falls back to a mock if unset)
- `LANGCHAIN_API_KEY` — LangSmith tracing (tracing disabled if unset)

## Infrastructure domain (System & Reliability Engineer)

- **`infrastructure/tools/search_tool.py`** — Tavily search tool for the Trend
  Strategist. Degrades to a deterministic mock when `TAVILY_API_KEY` is unset.
  ```python
  from infrastructure.tools.search_tool import search_tool
  search_tool.invoke({"query": "Trending perspectives on AI agents"})
  ```
- **`infrastructure/tools/vector_rag.py`** — ChromaDB store of high-performing
  hook templates (the RAG element). Auto-seeds on first use.
  ```python
  from infrastructure.tools.vector_rag import hook_retriever_tool
  hook_retriever_tool.invoke({"query": "AI agents for developers"})
  ```
- **`infrastructure/validation/reviewer.py`** — Agent 3. `reviewer_node(state)`
  runs deterministic guardrails (LinkedIn 40–400 words, Instagram ≤2200 chars,
  ≥3 hashtags, opening hook, restricted-topic refusal) plus an LLM `ReviewResult`
  check. `review_router(state)` returns `approved` / `revise` / `refused`.
- **`infrastructure/observability.py`** — `configure_langsmith()` enables tracing
  when `LANGCHAIN_API_KEY` is set; no-op otherwise.
- **`infrastructure/evaluation/test_cases.py`** — the 5 required evaluation
  scenarios + a runnable harness with latency tracking.

### Wiring the reviewer into the graph (for the Lead Agent Architect)

```python
from infrastructure.validation.reviewer import reviewer_node, review_router

builder.add_node("reviewer", reviewer_node)
builder.add_conditional_edges("reviewer", review_router, {
    "approved": END,
    "refused": END,
    "revise": "copywriter",
})
```

Also call `configure_langsmith()` once at the graph entrypoint so the live demo
emits traces:

```python
from infrastructure.observability import configure_langsmith
configure_langsmith()
```

## Evaluation

```bash
python -m infrastructure.evaluation.test_cases
```

The 5 scenarios:

| ID     | Scenario                | What it checks                                   |
|--------|-------------------------|--------------------------------------------------|
| EVAL-1 | Messy / vague prompt    | Still produces both drafts (runs once graph exists) |
| EVAL-2 | Soft-refusal guardrail  | Restricted topic is refused, no content produced |
| EVAL-3 | Happy path — clean topic| Clean drafts pass all deterministic guardrails   |
| EVAL-4 | Latency tracking        | Reviewer completes under budget (<30s)           |
| EVAL-5 | Format-violation recovery| Short post / missing hashtags → routes to revise |

Guardrail scenarios (2–5) run today. EVAL-1 activates once `core/graph.py`
exposes a compiled app (`app`, `graph`, or `build_graph()`).

## Status

- ✅ Infrastructure domain (tools, guardrails, observability, eval) — code complete.
- 🚧 `core/graph.py` — not yet built; nothing runs fully end-to-end until it lands.
- ⚙️ Requires `pip install`, `ollama pull llama3`, and API keys before first run.
