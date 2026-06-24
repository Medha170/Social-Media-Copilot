from langchain_ollama import ChatOllama
from core.states import AgentState, StrategyBrief
from infrastructure.tools.search_tool import search_tool
from infrastructure.tools.vector_rag import hook_retriever_tool

# Initialize local Ollama
llm = ChatOllama(model="llama3.1", temperature=0.2)

def trend_strategist_node(state: AgentState) -> dict:
    """
    Agent 1: Grounded Research Lead. Uses real-time search data and 
    historical vector RAG store to craft a high-impact content strategy brief.
    """
    topic = state["topic"]
    
    # 1. Invoke infrastructure tools to ground the agent
    try:
        search_results = search_tool.invoke({"query": f"Trending perspectives on {topic}"})
    except Exception:
        search_results = "No recent trend data found. Rely on foundational knowledge."
        
    try:
        hook_examples = hook_retriever_tool.invoke({"query": topic})
    except Exception:
        hook_examples = "No historical hook templates available."
    
    # 2. Tighten prompt engineering for local model constraints
    system_instruction = (
        "You are an expert Social Media Content Strategist. Your job is to analyze a raw topic, "
        "live web search trends, and successful historical copywriting hook templates to create a high-impact brief.\n\n"
        "CRITICAL: You must return your analysis strictly matching the following JSON structure, with absolutely no conversational filler or markdown blocks outside the raw JSON:\n"
        "{\n"
        "  \"hook_angle\": \"A psychological angle or high-converting hook optimized for this specific context.\",\n"
        "  \"key_pillars\": [\"Core point 1\", \"Core point 2\", \"Core point 3\"],\n"
        "  \"trending_keywords\": [\"keyword1\", \"keyword2\", \"keyword3\"]\n"
        "}"
    )
    
    user_message = (
        f"Target Topic: {topic}\n\n"
        f"--- LIVE WEB TRENDS ---\n{search_results}\n\n"
        f"--- SUCCESSFUL HOOK TEMPLATES (RAG) ---\n{hook_examples}\n\n"
        "Synthesize these inputs and generate the structured JSON brief now."
    )
    
    # Bind Pydantic schema for strict structured output validation
    structured_llm = llm.with_structured_output(StrategyBrief)
    
    brief = structured_llm.invoke([
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ])
    
    return {"strategy_brief": brief}