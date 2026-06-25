"""Streamlit demo for the Multi-Agent Social Media Co-Pilot.

Runs the real LangGraph app, streams every agent step to the UI, captures
tool/LLM calls via a callback handler, and drives the human-in-the-loop
approval gate that the graph exposes through ``interrupt_before=["reviewer"]``.
"""

import os
import sys
import uuid
import traceback
from datetime import datetime

import streamlit as st

# Make `core` / `infrastructure` importable no matter where streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Social Media Co-Pilot", page_icon="🤖", layout="wide")

# --- Import the compiled graph defensively so the UI always launches ---------
GRAPH_IMPORT_ERROR = None
compiled_graph = None
try:
    from core.graph import compiled_internal_graph as compiled_graph
except Exception as exc:  # missing deps, import-time failures, etc.
    GRAPH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"


# =====================================================================
# Event logging
# =====================================================================
KIND_ICON = {
    "system": "⚙️",
    "node": "🧠",
    "tool": "🔧",
    "llm": "💬",
    "human": "🙋",
    "error": "❌",
}


def log_event(kind: str, title: str, body: str = ""):
    """Append a timestamped event to the session log."""
    st.session_state.events.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "kind": kind,
            "title": title,
            "body": body,
        }
    )


try:
    from langchain_core.callbacks import BaseCallbackHandler

    class UILogHandler(BaseCallbackHandler):
        """Streams tool + LLM activity into the UI event log.

        Runs synchronously inside the same Streamlit script run as the graph,
        so writing straight to the session event list is safe.
        """

        def on_tool_start(self, serialized, input_str, **kwargs):
            name = (serialized or {}).get("name", "tool")
            log_event("tool", f"Tool call → {name}", str(input_str)[:600])

        def on_tool_end(self, output, **kwargs):
            log_event("tool", "Tool returned", str(output)[:600])

        def on_chat_model_start(self, serialized, messages, **kwargs):
            log_event("llm", "LLM call started", "")

        def on_llm_start(self, serialized, prompts, **kwargs):
            log_event("llm", "LLM call started", "")

        def on_llm_end(self, response, **kwargs):
            log_event("llm", "LLM call finished", "")

        def on_tool_error(self, error, **kwargs):
            log_event("error", "Tool error", str(error)[:600])

        def on_llm_error(self, error, **kwargs):
            log_event("error", "LLM error", str(error)[:600])

    _HANDLER_CLS = UILogHandler
except Exception:
    _HANDLER_CLS = None


# =====================================================================
# Session state
# =====================================================================
def init_state():
    st.session_state.setdefault("events", [])
    st.session_state.setdefault("phase", "idle")  # idle | paused | final
    st.session_state.setdefault("thread_id", None)
    st.session_state.setdefault("snapshot", {})


init_state()


def _config():
    cfg = {"configurable": {"thread_id": st.session_state.thread_id}}
    if _HANDLER_CLS is not None:
        cfg["callbacks"] = [_HANDLER_CLS()]
    return cfg


def _merge_snapshot(node_name: str, payload: dict):
    """Record a node's output into the running snapshot and the log."""
    if not isinstance(payload, dict):
        return
    st.session_state.snapshot.update(payload)
    summary_keys = ", ".join(payload.keys())
    log_event("node", f"Agent step complete → {node_name}", f"updated: {summary_keys}")


def advance(graph_input):
    """Stream the graph forward until it interrupts (HITL) or finishes.

    ``graph_input`` is the initial state dict for a fresh run, or ``None`` to
    resume a run that is paused at the interrupt before the reviewer.
    """
    config = _config()
    try:
        for chunk in compiled_graph.stream(graph_input, config, stream_mode="updates"):
            for node_name, payload in chunk.items():
                _merge_snapshot(node_name, payload)
    except Exception as exc:
        log_event("error", "Graph execution failed", f"{type(exc).__name__}: {exc}")
        log_event("error", "Traceback", traceback.format_exc()[-1500:])
        st.session_state.phase = "final"
        return

    # Decide where we landed using the checkpointer's view of the run.
    try:
        state = compiled_graph.get_state(config)
        next_nodes = tuple(state.next) if state and state.next else ()
    except Exception as exc:
        log_event("error", "Could not read graph state", str(exc))
        next_nodes = ()

    if "reviewer" in next_nodes:
        st.session_state.phase = "paused"
        log_event("system", "Paused for human review", "Graph interrupted before the reviewer agent.")
    else:
        st.session_state.phase = "final"
        log_event("system", "Run finished", "Graph reached END.")


# =====================================================================
# Human-in-the-loop decision policy
# =====================================================================
def apply_human_decision(action, edited_linkedin, edited_instagram, current):
    """Map the human's checkpoint choice to a LangGraph state update.

    Returns a dict of state keys to overwrite before the run resumes into the
    reviewer (an empty dict means "resume unchanged"). ``action`` is one of the
    radio choices below; ``current`` is the latest snapshot of state.
    """
    if action != "Apply my edits":
        return {}
    updates = {}
    if edited_linkedin != current.get("linkedin_draft"):
        updates["linkedin_draft"] = edited_linkedin
    if edited_instagram != current.get("instagram_caption"):
        updates["instagram_caption"] = edited_instagram
    return updates


# =====================================================================
# UI
# =====================================================================
st.title("🤖 Multi-Agent Social Media Co-Pilot")
st.caption("Trend Strategist → Platform Copywriter → (human review) → Guardrails Reviewer")

if GRAPH_IMPORT_ERROR:
    st.error(
        "The agent graph could not be loaded, so runs are disabled. "
        "Fix setup, then refresh.\n\n"
        f"**Import error:** `{GRAPH_IMPORT_ERROR}`"
    )
    st.info(
        "**Setup checklist**\n"
        "1. `pip install -r requirements.txt`\n"
        "2. Start Ollama and run `ollama pull llama3.1`\n"
        "3. (optional) copy `.env.example` to `.env` for live search + tracing"
    )

col_main, col_log = st.columns([3, 2])

with col_main:
    # ---- Run controls -------------------------------------------------------
    disabled = GRAPH_IMPORT_ERROR is not None or st.session_state.phase != "idle"
    topic = st.text_input(
        "Content topic",
        value="AI agents for developers",
        disabled=disabled,
        placeholder="e.g. Building reliable LangGraph pipelines",
    )
    if st.button("🚀 Run pipeline", disabled=disabled, type="primary"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.snapshot = {}
        log_event("system", "Pipeline started", f"Topic: {topic!r} · thread {st.session_state.thread_id[:8]}")
        advance({"topic": topic, "loop_count": 0})
        st.rerun()

    snap = st.session_state.snapshot

    # ---- Strategy brief -----------------------------------------------------
    brief = snap.get("strategy_brief")
    if brief is not None:
        with st.expander("📋 Strategy brief", expanded=True):
            st.markdown(f"**Hook angle:** {getattr(brief, 'hook_angle', '')}")
            pillars = getattr(brief, "key_pillars", []) or []
            st.markdown("**Key pillars:**\n" + "\n".join(f"- {p}" for p in pillars))
            kws = getattr(brief, "trending_keywords", []) or []
            st.markdown("**Trending keywords:** " + ", ".join(kws))

    # ---- Human-in-the-loop checkpoint --------------------------------------
    if st.session_state.phase == "paused":
        st.subheader("🙋 Human review checkpoint")
        st.info("The drafts below are paused before the automated reviewer. Inspect, optionally edit, then continue.")
        action = st.radio(
            "Decision",
            ["Approve as-is", "Apply my edits"],
            horizontal=True,
        )
        edited_li = st.text_area("LinkedIn draft", value=snap.get("linkedin_draft", "") or "", height=220)
        edited_ig = st.text_area("Instagram brief / caption", value=snap.get("instagram_caption", "") or "", height=220)

        if st.button("✅ Send to Reviewer", type="primary"):
            updates = apply_human_decision(action, edited_li, edited_ig, snap) or {}
            log_event("human", f"Human decision: {action}", f"state keys updated: {list(updates.keys()) or 'none'}")
            if updates:
                try:
                    compiled_graph.update_state(_config(), updates)
                    st.session_state.snapshot.update(updates)
                except Exception as exc:
                    log_event("error", "Failed to apply human edits", str(exc))
            advance(None)
            st.rerun()

    # ---- Final drafts -------------------------------------------------------
    if snap.get("linkedin_draft") and st.session_state.phase != "paused":
        st.subheader("📝 Drafts")
        t_li, t_ig = st.tabs(["LinkedIn", "Instagram"])
        with t_li:
            st.markdown(snap.get("linkedin_draft", ""))
        with t_ig:
            st.markdown(snap.get("instagram_caption", ""))

    # ---- Review verdict -----------------------------------------------------
    if st.session_state.phase == "final" and "review_approved" in snap:
        approved = snap.get("review_approved")
        st.subheader("🛡️ Reviewer verdict")
        if approved:
            st.success(f"Approved ✅  (revision loops: {snap.get('loop_count', 0)})")
        else:
            st.warning(f"Not approved ⚠️  (revision loops: {snap.get('loop_count', 0)})")
        if snap.get("feedback"):
            st.markdown("**Reviewer feedback:**")
            st.code(snap["feedback"])

    if st.session_state.phase == "final":
        if st.button("🔄 New run"):
            st.session_state.phase = "idle"
            st.session_state.snapshot = {}
            st.session_state.thread_id = None
            log_event("system", "Reset", "Ready for a new topic.")
            st.rerun()

with col_log:
    st.subheader("📜 Execution log")
    st.caption(f"{len(st.session_state.events)} events")
    if st.button("Clear log"):
        st.session_state.events = []
        st.rerun()
    for ev in reversed(st.session_state.events):
        icon = KIND_ICON.get(ev["kind"], "•")
        line = f"{icon} `{ev['time']}` **{ev['title']}**"
        st.markdown(line)
        if ev["body"]:
            st.caption(ev["body"])
