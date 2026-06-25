# System Flow — Multi-Agent Social Media Co-Pilot

End-to-end flow of the LangGraph multi-agent system. `[YOU]` marks the
infrastructure domain (System & Reliability Engineer); `[FRIEND]` marks the
core agent domain (Lead Agent Architect).

```
                          ┌─────────────────┐
                          │   topic (text)  │   e.g. "AI agents for devs"
                          └────────┬────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │   AGENT 1: TREND STRATEGIST    [FRIEND]   │
              │   core/agents/strategist.py               │
              │                                           │
              │   calls 2 of YOUR tools to "ground" it:   │
              │   ┌─────────────────────────────────────┐ │
              │   │ search_tool      [YOU]              │ │  live web trends
              │   │ infra/tools/search_tool.py          │ │  (Tavily, mock fallback)
              │   ├─────────────────────────────────────┤ │
              │   │ hook_retriever_tool  [YOU]          │ │  past winning hooks
              │   │ infra/tools/vector_rag.py (RAG)     │ │  (ChromaDB)
              │   └─────────────────────────────────────┘ │
              │                                           │
              │   OUTPUT: StrategyBrief (Pydantic)        │
              │   {hook_angle, key_pillars, keywords}     │
              └────────────────────┬──────────────────────┘
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │   AGENT 2: PLATFORM COPYWRITER  [FRIEND]  │
              │   core/agents/copywriter.py               │
              │                                           │
              │   writes 2 platform-native drafts:        │
              │     • linkedin_draft                      │
              │     • instagram_caption                   │
              │   (rewrites if reviewer sends feedback)   │
              └────────────────────┬──────────────────────┘
                                   │
                                   ▼
                   ╔═══════════════════════════════╗
                   ║   ⏸  HUMAN-IN-THE-LOOP PAUSE  ║   interrupt_before=["reviewer"]
                   ║   graph stops; human approves  ║   (set in core/graph.py [FRIEND])
                   ║   before reviewer runs         ║
                   ╚═══════════════┬═══════════════╝
                                   │
                                   ▼
              ┌──────────────────────────────────────────┐
              │   AGENT 3: GUARDRAILS REVIEWER   [YOU]    │
              │   infra/validation/reviewer.py            │
              │                                           │
              │   1) run_deterministic_checks()           │
              │      • LinkedIn 40–400 words              │
              │      • Instagram ≤2200 chars, ≥3 hashtags │
              │      • opening hook present               │
              │      • BANNED_TERMS → REFUSAL             │
              │   2) _llm_quality_review() → ReviewResult │
              │                                           │
              │   OUTPUT: review_approved, feedback,      │
              │           loop_count                      │
              └────────────────────┬──────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────┐
                    │  review_router()   [YOU]  │   reads the state, picks 1 of 3
                    └───┬──────────┬─────────┬──┘
                        │          │         │
            "refused"   │ "revise" │         │ "approved"
          (banned topic)│          │         │ (or loop_count ≥ 3)
                        │          │         │
                        ▼          │         ▼
                    ┌───────┐      │     ┌───────┐
                    │  END  │      │     │  END  │
                    │ (no   │      │     │ (ship │
                    │ content)     │     │ drafts)│
                    └───────┘      │     └───────┘
                                   │
                                   └──────────► back to COPYWRITER
                                               (try again with feedback)
```

## How the test cases map to this flow

```
EVAL-1  (mode="graph")      → runs the ENTIRE diagram above
                              topic → strategist → copywriter → reviewer → router
                              In LangSmith: appears as one "LangGraph" trace
                              with ALL the nodes nested inside it.

EVAL-2,3,4,5 (mode="guardrail") → runs ONLY the Reviewer box [YOU]
                              (calls reviewer_node directly, skips agents 1 & 2)
                              In LangSmith: appears as a "RunnableSequence"
                              trace with just the reviewer's one LLM call.
                              (EVAL-2 has NO LLM trace at all — the banned-term
                               REFUSAL short-circuits before the model is called.)
```
