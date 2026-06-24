# TODO — Lead Agent Architect (core/ domain)

Hey! The infrastructure side (`infrastructure/`) is code-complete: search tool,
RAG hook store, guardrails reviewer, LangSmith setup, and the 5 eval scenarios
are all in. This is what's left on your side to make the system run end-to-end,
plus the shared items we still owe.

Owners: **You** = Lead Agent Architect (core graph + agents). **Me** = System &
Reliability Engineer (infrastructure, done).

---

## 1. Build `core/graph.py` (the main blocker — nothing runs without this)

- [ ] Define the `StateGraph` over `AgentState` (from `core/states.py`).
- [ ] Add nodes:
  - [ ] `strategist` → `trend_strategist_node`
  - [ ] `copywriter` → `platform_copywriter_node`
  - [ ] `reviewer` → `reviewer_node` (import from infra, see below)
- [ ] Edges: `START → strategist → copywriter → reviewer`.
- [ ] Conditional branching after `reviewer` using my router:
  ```python
  from infrastructure.validation.reviewer import reviewer_node, review_router
  builder.add_conditional_edges("reviewer", review_router, {
      "approved": END,
      "refused": END,
      "revise": "copywriter",   # loops back with feedback
  })
  ```
- [ ] Compile with a checkpointer (needed for interrupts), e.g. `MemorySaver`.
- [ ] **Expose the compiled app** as a module-level `app` (or `graph`, or a
      `build_graph()` function) so my eval harness auto-detects it and runs
      EVAL-1.

## 2. Human-in-the-loop interrupt

- [ ] Add `interrupt_before=["reviewer"]` (or `interrupt_after`) on compile.
- [ ] Pause after drafts are generated so the user can read/approve/edit.
- [ ] Accept user text feedback, update state (`feedback`, `review_approved`),
      and resume so the corrected text flows back through the graph.

## 3. Wire the infra tools into the Strategist

Right now `core/agents/strategist.py` uses `mock_search_results`. Replace it:

- [ ] ```python
      from infrastructure.tools.search_tool import search_tool
      from infrastructure.tools.vector_rag import hook_retriever_tool

      search_results = search_tool.invoke({"query": f"Trending perspectives on {topic}"})
      hook_examples = hook_retriever_tool.invoke({"query": topic})
      ```
- [ ] Feed both into the strategist's user message (real trends + proven hook
      formats) before generating the `StrategyBrief`.

## 4. Enable tracing at the entrypoint

- [ ] Call this once before running the graph (so the live demo emits traces):
      ```python
      from infrastructure.observability import configure_langsmith
      configure_langsmith()
      ```

## 5. Prompt polish (your domain)

- [ ] Tighten the Strategist system prompt so the brief is clean and specific.
- [ ] Tighten the Copywriter prompt so LinkedIn vs Instagram formats stay
      distinct and the `===PLATFORM_SPLIT===` delimiter is always emitted.

---

## Shared checklist (still pending)

### API keys / env — **pending from our end**

- [ ] **Tavily** (`TAVILY_API_KEY`) — sign up at tavily.com, get a key. Until
      set, search silently uses a mock (fine for dev, but we need it live for
      the demo to show real grounding).
- [ ] **LangSmith** (`LANGCHAIN_API_KEY`) — sign up at smith.langchain.com.
      Until set, tracing is disabled (no traces in the dashboard).
- [ ] **LLM** — we're on local Ollama `llama3` (no key needed). If we switch to
      OpenAI/Anthropic, add `OPENAI_API_KEY` to `.env`.
- [ ] Copy `.env.example` → `.env` and paste the keys. **Never commit `.env`.**
- [ ] `pip install -r requirements.txt` and `ollama pull llama3`.

### Verify end-to-end before demo day

- [ ] Run `python -m infrastructure.evaluation.test_cases` — confirm all 5 pass
      (EVAL-1 only runs once `core/graph.py` exposes the app).
- [ ] Confirm a `social-media-copilot` project with traces appears in LangSmith.
- [ ] Do a full manual run: messy topic → drafts → human edit → approved.

### Git

- [ ] Confirm we're both on the agreed branch convention (`main`).
- [ ] Commit + push the infra work and the new `core/graph.py`.

---

## Presentation (10 min, June 25–30 window)

- [ ] **Mins 0–4 (shared):** problem statement, why one chatbot isn't enough,
      the multi-agent graph architecture.
- [ ] **Mins 4–7 (shared):** live end-to-end demo — show the tools, the revise
      loop, and the human-approval step.
- [ ] **Mins 7–8.5 (me):** evaluation results, LangSmith traces, guardrail /
      refusal rules.
- [ ] **Mins 8.5–10 (each):** individual contributions + code decisions.
