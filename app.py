"""Streamlit demo for the Multi-Agent Social Media Co-Pilot.

Runs the real LangGraph app, streams every agent step to the UI, captures
tool/LLM calls via a callback handler, and drives the human-in-the-loop
approval gate that the graph exposes through ``interrupt_before=["reviewer"]``.
"""

import os
import sys
import uuid
import contextlib
import traceback
from datetime import datetime

import streamlit as st

# Make `core` / `infrastructure` importable no matter where streamlit is launched from.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(page_title="Social Media Co-Pilot", page_icon="🤖", layout="wide")

MODEL_NAME = "llama3.1"

# --- Import the compiled graph defensively so the UI always launches ---------
GRAPH_IMPORT_ERROR = None
compiled_graph = None
try:
    from core.graph import compiled_internal_graph as compiled_graph
except Exception as exc:  # missing deps, import-time failures, etc.
    GRAPH_IMPORT_ERROR = f"{type(exc).__name__}: {exc}"

# --- LangSmith observability detection ---------------------------------------
# configure_langsmith() runs at graph import; it sets these env vars when a key
# is present. Reading them here tells us whether traces are being recorded.
TRACING_ON = bool(os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY"))
LANGSMITH_PROJECT = (
    os.getenv("LANGCHAIN_PROJECT") or os.getenv("LANGSMITH_PROJECT") or "social-media-copilot"
)

try:
    from langchain_core.tracers.context import tracing_v2_enabled
except Exception:
    tracing_v2_enabled = None


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
    st.session_state.setdefault("traces", [])  # [{label, url}] LangSmith trace links


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


def _trace_context():
    """LangSmith trace span for one graph advance, or a no-op when tracing is off."""
    if TRACING_ON and tracing_v2_enabled is not None:
        return tracing_v2_enabled(project_name=LANGSMITH_PROJECT)
    return contextlib.nullcontext()


def _record_trace(tracer, label):
    """Capture a clickable LangSmith run URL from the tracer, if available."""
    if tracer is None:
        return
    try:
        url = tracer.get_run_url()
    except Exception:
        url = None
    if url:
        st.session_state.traces.append({"label": label, "url": url})
        log_event("system", "LangSmith trace recorded", label)


def advance(graph_input, trace_label="Pipeline run"):
    """Stream the graph forward until it interrupts (HITL) or finishes.

    ``graph_input`` is the initial state dict for a fresh run, or ``None`` to
    resume a run that is paused at the interrupt before the reviewer.
    """
    config = _config()
    try:
        with _trace_context() as tracer:
            for chunk in compiled_graph.stream(graph_input, config, stream_mode="updates"):
                for node_name, payload in chunk.items():
                    _merge_snapshot(node_name, payload)
            _record_trace(tracer, trace_label)
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
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem;}
      div[data-testid="stMetric"] {
        background: #f7f8fa; border: 1px solid #e6e8eb;
        border-radius: 10px; padding: 12px 16px;
      }
      .ls-badge {
        display:inline-block; padding:3px 10px; border-radius:999px;
        font-size:0.8rem; font-weight:600;
      }
      .ls-on  {background:#e7f6ec; color:#137333; border:1px solid #b7e1c5;}
      .ls-off {background:#f1f3f4; color:#5f6368; border:1px solid #dadce0;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🤖 Multi-Agent Social Media Co-Pilot")
st.caption("Trend Strategist → Platform Copywriter → (human review) → Guardrails Reviewer")

snap = st.session_state.snapshot

# ---- Status strip -----------------------------------------------------------
m1, m2, m3, m4 = st.columns(4)
m1.metric("LangSmith tracing", "On" if TRACING_ON else "Off")
m2.metric("Model", MODEL_NAME)
m3.metric("Stage", st.session_state.phase.capitalize())
m4.metric("Revision loops", snap.get("loop_count", 0))

# ---- Sidebar: observability + setup ----------------------------------------
with st.sidebar:
    st.header("🔭 Observability")
    badge = (
        '<span class="ls-badge ls-on">● TRACING ON</span>'
        if TRACING_ON
        else '<span class="ls-badge ls-off">● TRACING OFF</span>'
    )
    st.markdown(badge, unsafe_allow_html=True)
    st.caption(
        "LangSmith records every step the agents take — each LLM call and tool "
        "call, with its inputs, outputs, latency and token usage — as a replayable "
        "**trace**."
    )
    if TRACING_ON:
        st.markdown(f"**Project:** `{LANGSMITH_PROJECT}`")
        st.link_button("Open project in LangSmith ↗", "https://smith.langchain.com")
    else:
        st.info("Set `LANGCHAIN_API_KEY` in `.env` to record traces, then refresh.")

    if st.session_state.traces:
        st.markdown("**Traces from this session**")
        for i, tr in enumerate(reversed(st.session_state.traces), 1):
            st.markdown(f"{i}. [{tr['label']} ↗]({tr['url']})")

    st.divider()
    st.header("⚙️ Setup")
    st.markdown(
        "1. `pip install -r requirements.txt`\n"
        "2. Start Ollama · `ollama pull llama3.1`\n"
        "3. (optional) copy `.env.example` → `.env` for live search + LangSmith"
    )

if GRAPH_IMPORT_ERROR:
    st.error(
        "The agent graph could not be loaded, so runs are disabled. "
        "Fix setup (see sidebar), then refresh.\n\n"
        f"**Import error:** `{GRAPH_IMPORT_ERROR}`"
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
        advance({"topic": topic, "loop_count": 0}, trace_label="Pipeline run")
        st.rerun()

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
            advance(None, trace_label=f"Human review → reviewer ({action})")
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
            st.session_state.traces = []
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
