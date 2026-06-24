import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

from infrastructure.observability import configure_langsmith
try:
    configure_langsmith()
except Exception:
    pass

from core.states import AgentState
from core.agents.strategist import trend_strategist_node
from core.agents.copywriter import platform_copywriter_node
from infrastructure.validation.reviewer import reviewer_node, review_router

builder = StateGraph(AgentState)

builder.add_node("strategist", trend_strategist_node)
builder.add_node("copywriter", platform_copywriter_node)
builder.add_node("reviewer", reviewer_node)

builder.add_edge(START, "strategist")
builder.add_edge("strategist", "copywriter")
builder.add_edge("copywriter", "reviewer")

builder.add_conditional_edges(
    "reviewer",
    review_router,
    {
        "approved": END,
        "refused": END,
        "revise": "copywriter"
    }
)

memory_checkpointer = MemorySaver()
compiled_internal_graph = builder.compile(
    checkpointer=memory_checkpointer,
    interrupt_before=["reviewer"]
)

# --- EVALUATION HARNESS WRAPPER ---
# This intercepts the harness call and attaches the mandatory thread configuration 
# while still exposing the exact interface your teammate's file expects.
class WrappedApp:
    def invoke(self, state, config=None):
        if config is None:
            config = {"configurable": {"thread_id": "eval_harness_thread"}}
        return compiled_internal_graph.invoke(state, config)
    
    def stream(self, state, config=None, **kwargs):
        if config is None:
            config = {"configurable": {"thread_id": "eval_harness_thread"}}
        return compiled_internal_graph.stream(state, config, **kwargs)

app = WrappedApp()