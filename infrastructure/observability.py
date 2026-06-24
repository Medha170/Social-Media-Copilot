"""LangSmith observability setup.

Call :func:`configure_langsmith` once at process start (graph entrypoint or the
evaluation runner) to enable automatic tracing of every LLM and tool call.
Reads config from environment / ``.env``; it's a no-op when no API key is set,
so the app still runs without LangSmith configured.
"""

import os

from dotenv import load_dotenv

load_dotenv()


def configure_langsmith(project_name: str = "social-media-copilot") -> bool:
    """Enable LangSmith tracing if an API key is present.

    Returns True if tracing was enabled, False otherwise.
    """
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        print("[observability] LANGCHAIN_API_KEY not set — tracing disabled.")
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = api_key
    os.environ.setdefault("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")
    os.environ.setdefault("LANGCHAIN_PROJECT", project_name)
    print(
        f"[observability] LangSmith tracing enabled → "
        f"project '{os.environ['LANGCHAIN_PROJECT']}'."
    )
    return True
