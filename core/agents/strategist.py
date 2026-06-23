from langchain_openai import ChatOpenAI
from core.states import AgentState, StrategyBrief

# Initialize the Language Model with high creativity for strategic angles
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

def trend_strategist_node(state: AgentState) -> dict:
    """
    Agent 1: Research and analyze current trends to construct a content strategy.
    """
    topic = state["topic"]
    
    # -----------------------------------------------------------------
    # NOTE FOR YOUR TEAMMATE: 
    # Once she finishes the tools in 'infrastructure/tools/', we will import 
    # her search_tool here and call it like: 
    # search_results = search_tool.invoke({"query": f"Trending perspectives on {topic}"})
    # For now, we use a placeholder string so your code compiles and runs.
    # -----------------------------------------------------------------
    mock_search_results = (
        f"Recent discussions on {topic} highlight a shift toward automation, "
        f"developer efficiency, and practical architectural implementation frameworks."
    )
    
    # Define a rigorous system prompt to steer the strategist's reasoning
    system_instruction = (
        "You are an expert Social Media Content Strategist. Your job is to take a raw topic "
        "and real-time market trends, then synthesize them into a highly engaging strategy brief.\n\n"
        "Focus on extracting a compelling psychological hook angle that makes readers stop scrolling. "
        "Identify exactly 3 core technical pillars that must be covered, and pick high-impact, "
        "trending keywords/hashtags suitable for technical audiences."
    )
    
    user_message = (
        f"Target Topic: {topic}\n"
        f"Current Market Context/Trends: {mock_search_results}\n\n"
        "Generate a structured content brief adhering exactly to the requested output format."
    )
    
    # Bind the LLM to output ONLY our structured Pydantic schema
    # This enforces the strict "Structured Output" requirement of the rubric!
    structured_llm = llm.with_structured_output(StrategyBrief)
    
    # Execute the model
    brief = structured_llm.invoke([
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ])
    
    # Return the dictionary updates to be merged into the central Graph State
    return {"strategy_brief": brief}