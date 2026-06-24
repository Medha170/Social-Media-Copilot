"""Agent 3 — Guardrails Reviewer node.

Combines deterministic policy checks (word counts, hashtag presence, hook
presence, restricted-topic refusal) with an LLM quality review that emits the
``ReviewResult`` Pydantic schema from ``core.states``. Writes
``review_approved`` / ``feedback`` back into the graph state and increments
``loop_count`` to prevent infinite revision loops.

The companion ``review_router`` is a conditional-edge helper for
``core/graph.py``: wire it after this node to decide whether to finish, refuse,
or loop back to the copywriter.
"""

import re

from langchain_ollama import ChatOllama

from core.states import AgentState, ReviewResult

# Deterministic temperature: the reviewer should be consistent, not creative.
llm = ChatOllama(model="llama3.1", temperature=0.0)

# --- Guardrail thresholds ---
LINKEDIN_MIN_WORDS = 40
LINKEDIN_MAX_WORDS = 400          # ~3000 char LinkedIn limit, kept conservative
INSTAGRAM_MAX_CHARS = 2200        # hard Instagram caption limit
MIN_HASHTAGS = 3
MAX_LOOPS = 3                     # stop revising after this many attempts
HOOK_MAX_WORDS = 18              # an opening "hook" line must be punchy

# Topics we refuse to produce marketing content for (soft refusal guardrail).
BANNED_TERMS = [
    "weapon", "explosive", "bomb", "self-harm", "suicide", "illegal drug",
    "hate speech", "csam", "phishing", "malware", "ransomware",
]


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _count_hashtags(text: str) -> int:
    return len(re.findall(r"#\w+", text or ""))


def _has_hook(text: str) -> bool:
    """A hook is a short, punchy, non-empty opening line."""
    if not text or not text.strip():
        return False
    first_line = text.strip().splitlines()[0]
    return 0 < _word_count(first_line) <= HOOK_MAX_WORDS


def run_deterministic_checks(state: AgentState) -> list[str]:
    """Return a list of guardrail violations (empty list == passes)."""
    violations: list[str] = []
    topic = (state.get("topic") or "").lower()
    linkedin = state.get("linkedin_draft") or ""
    instagram = state.get("instagram_caption") or ""

    # Soft refusal guardrail — short-circuit before any formatting checks.
    for term in BANNED_TERMS:
        if term in topic:
            violations.append(
                f"REFUSAL: topic contains restricted term '{term}'. "
                "Content cannot be generated."
            )
            return violations

    # LinkedIn checks
    if not linkedin.strip():
        violations.append("LinkedIn draft is missing or empty.")
    else:
        wc = _word_count(linkedin)
        if wc < LINKEDIN_MIN_WORDS:
            violations.append(
                f"LinkedIn post too short ({wc} words; min {LINKEDIN_MIN_WORDS})."
            )
        if wc > LINKEDIN_MAX_WORDS:
            violations.append(
                f"LinkedIn post too long ({wc} words; max {LINKEDIN_MAX_WORDS})."
            )
        if not _has_hook(linkedin):
            violations.append("LinkedIn post lacks a punchy opening hook line.")

    # Instagram checks
    if not instagram.strip():
        violations.append("Instagram caption/brief is missing or empty.")
    else:
        if len(instagram) > INSTAGRAM_MAX_CHARS:
            violations.append(
                f"Instagram caption exceeds {INSTAGRAM_MAX_CHARS} characters "
                f"({len(instagram)})."
            )
        tags = _count_hashtags(instagram)
        if tags < MIN_HASHTAGS:
            violations.append(
                f"Instagram caption has too few hashtags ({tags}; min {MIN_HASHTAGS})."
            )
        if not _has_hook(instagram):
            violations.append("Instagram caption lacks a strong opening hook line.")

    return violations


def _llm_quality_review(state: AgentState) -> ReviewResult:
    """LLM judgement layer on top of the deterministic checks."""
    system = (
        "You are a strict Social Media Quality Reviewer. Judge whether the drafts "
        "are on-topic, engaging, professional, and free of unsafe or off-brand "
        "content. Approve only if the content is genuinely high quality."
    )
    user = (
        f"Topic: {state.get('topic')}\n\n"
        f"LINKEDIN DRAFT:\n{state.get('linkedin_draft')}\n\n"
        f"INSTAGRAM DRAFT:\n{state.get('instagram_caption')}\n\n"
        "Return your verdict in the required structured format."
    )
    structured_llm = llm.with_structured_output(ReviewResult)
    return structured_llm.invoke(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
    )


def reviewer_node(state: AgentState) -> dict:
    """Agent 3: validate drafts against guardrails + quality bar.

    Returns state updates: ``review_approved``, ``feedback``, ``loop_count``.
    """
    loop_count = state.get("loop_count", 0) + 1

    violations = run_deterministic_checks(state)
    is_refusal = any(v.startswith("REFUSAL:") for v in violations)

    # Skip the (slow) LLM call on hard refusals.
    llm_result = None
    if not is_refusal:
        try:
            llm_result = _llm_quality_review(state)
        except Exception as exc:
            violations.append(f"Quality review could not run ({exc}).")

    approved = (
        not violations
        and llm_result is not None
        and llm_result.approved
    )

    feedback_parts: list[str] = []
    if violations:
        feedback_parts.append("Guardrail violations:\n- " + "\n- ".join(violations))
    if llm_result and not llm_result.approved and llm_result.feedback:
        feedback_parts.append(f"Reviewer critique: {llm_result.feedback}")
    feedback = "\n\n".join(feedback_parts) if feedback_parts else None

    return {
        "review_approved": approved,
        "feedback": feedback,
        "loop_count": loop_count,
    }


def review_router(state: AgentState) -> str:
    """Conditional-edge helper for core/graph.py.

    Maps the review outcome to one of three routes:
      - ``"refused"``  : restricted topic — graph should END, withhold content.
      - ``"approved"`` : drafts pass (or loop budget exhausted) — graph ENDs.
      - ``"revise"``   : send feedback back to the copywriter to try again.
    """
    feedback = state.get("feedback") or ""
    if "REFUSAL:" in feedback:
        return "refused"
    if state.get("review_approved"):
        return "approved"
    if state.get("loop_count", 0) >= MAX_LOOPS:
        # Loop budget exhausted: exit gracefully with the best-effort draft.
        return "approved"
    return "revise"
