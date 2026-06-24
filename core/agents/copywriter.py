from langchain_ollama import ChatOllama
from core.states import AgentState

llm = ChatOllama(model="llama3.1", temperature=0.6)

def platform_copywriter_node(state: AgentState) -> dict:
    """
    Agent 2: Multi-Platform Execution Copywriter. Consumes the strategy brief 
    to draft native text variations, handling refinement iterations dynamically.
    """
    topic = state["topic"]
    brief = state["strategy_brief"]
    feedback = state.get("feedback", None)
    
    system_instruction = (
        "You are an elite multi-platform copywriter specializing in developer and technical creator growth.\n\n"
        "Review the Strategy Brief and produce two distinct channel variants:\n"
        "1. LINKEDIN POST: Professional, punchy tone. Use clean headers, broad paragraph breaks, and actionable bullet points.\n"
        "2. INSTAGRAM BRIEF: Provide explicit step-by-step visual storyboard directives for a Carousel/Reel, followed by an engaging caption with a high-retention 3-second hook line and a targeted SEO hashtag cluster.\n\n"
        "CRITICAL COMPLIANCE RULES:\n"
        "- Do not combine the sections into a single markdown article.\n"
        "- You MUST split the LinkedIn post from the Instagram brief using the exact literal token string: '===PLATFORM_SPLIT==='\n"
        "- Example format:\n"
        "[Insert LinkedIn content here]\n"
        "===PLATFORM_SPLIT===\n"
        "[Insert Instagram storyboard, caption, and hashtags here]"
    )
    
    user_message = (
        f"Topic: {topic}\n"
        f"Angle: {brief.hook_angle}\n"
        f"Required Core Pillars: {', '.join(brief.key_pillars)}\n"
        f"Algorithmic Target Keywords: {', '.join(brief.trending_keywords)}\n\n"
    )
    
    # Dynamic loop handling
    if feedback and not state.get("review_approved", False):
        user_message += (
            f"⚠️ CRITICAL REVISION REQUESTED:\n"
            f"Your previous attempt was rejected by our Quality Reviewer with this feedback: '{feedback}'\n"
            f"Modify your approach to perfectly address this critique while preserving the core elements."
        )
    else:
        user_message += "Generate the initial platform variations now."

    response = llm.invoke([
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ])
    
    raw_output = response.content
    
    # Parse text cleanly across platform buffers
    if "===PLATFORM_SPLIT===" in raw_output:
        parts = raw_output.split("===PLATFORM_SPLIT===")
        linkedin_content = parts[0].strip()
        instagram_content = parts[1].strip()
    else:
        # Graceful fallback to avoid unhandled extraction faults
        linkedin_content = raw_output
        instagram_content = "Instagram Generation Error: Platform boundary missing during generation."

    return {
        "linkedin_draft": linkedin_content,
        "instagram_caption": instagram_content,
        "instagram_visual_brief": "Visual storyboard embedded inside the caption payload."
    }