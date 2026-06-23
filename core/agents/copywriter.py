from langchain_ollama import ChatOllama
from core.states import AgentState

# Initialize the LLM with a temperature of 0.7 for strong, engaging copy variations
llm = ChatOllama(model="llama3", temperature=0.6)

def platform_copywriter_node(state: AgentState) -> dict:
    """
    Agent 2: Transforms a structured strategy brief into platform-native 
    content for LinkedIn and Instagram, handling revisions if feedback exists.
    """
    topic = state["topic"]
    brief = state["strategy_brief"]
    feedback = state.get("feedback", None)
    
    # Base system prompt defining the persona and formatting instructions
    system_instruction = (
        "You are an expert Copywriter and Multi-Platform Content Creator.\n\n"
        "Your task is to review a structured Strategy Brief and turn it into two distinct pieces of content:\n"
        "1. A LINKEDIN POST: Professional, punchy, well-spaced text utilizing bold headers, bulleted takeaways, and clear line breaks.\n"
        "2. AN INSTAGRAM BRIEF: An explicit scene-by-scene script/visual directive layout for a Reel or Carousel, followed by a highly catchy Instagram caption (with a strong initial hook line) and an SEO hashtag block.\n\n"
        "Always separate the LinkedIn post from the Instagram brief using a clear string delimiter: '===PLATFORM_SPLIT==='"
    )
    
    # Construct user message incorporating the structured brief data
    user_message = (
        f"Core Topic: {topic}\n"
        f"Strategy Angle: {brief.hook_angle}\n"
        f"Key Pillars to hit: {', '.join(brief.key_pillars)}\n"
        f"Target Keywords: {', '.join(brief.trending_keywords)}\n\n"
    )
    
    # Critical Logic: If the Reviewer rejected the previous draft, force a revision path
    if feedback and not state["review_approved"]:
        user_message += (
            f"⚠️ ATTENTION: Your previous draft was REJECTED by the Quality Reviewer.\n"
            f"Critique/Feedback to fix: {feedback}\n"
            f"Please rewrite the drafts, making sure to fully resolve this critique."
        )
    else:
        user_message += "Generate the initial platform drafts now."

    # Invoke the LLM
    response = llm.invoke([
        {"role": "system", "content": system_instruction},
        {"role": "user", "content": user_message}
    ])
    
    raw_output = response.content
    
    # Parse the text to separate the two platform components cleanly
    linkedin_content = ""
    instagram_content = ""
    
    if "===PLATFORM_SPLIT===" in raw_output:
        parts = raw_output.split("===PLATFORM_SPLIT===")
        linkedin_content = parts[0].strip()
        instagram_content = parts[1].strip()
    else:
        # Fallback split if the LLM omits the exact token
        linkedin_content = raw_output
        instagram_content = "Instagram Brief generation failed to split properly. Retrying via graph loop may be required."

    # Return the dictionary updates to be merged into the central Graph State
    return {
        "linkedin_draft": linkedin_content,
        "instagram_caption": instagram_content,
        # We split visual scripts out into its own key if needed, or bundle it here for convenience
        "instagram_visual_brief": "Visual structure embedded in caption payload."
    }