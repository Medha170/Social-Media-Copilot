from typing import TypedDict, List, Optional
from pydantic import BaseModel, Field

# =====================================================================
# 1. Structured Outputs for Agent Handoffs (Pydantic Schemas)
# =====================================================================

class StrategyBrief(BaseModel):
    """Schema for the Trend Strategist's output."""
    hook_angle: str = Field(
        description="The psychological angle or hook optimized for current trends."
    )
    key_pillars: List[str] = Field(
        description="3 core technical concepts or pillars to cover in the content."
    )
    trending_keywords: List[str] = Field(
        description="Keywords or hashtags currently gaining algorithmic traction."
    )

class ReviewResult(BaseModel):
    """Schema for the Guardrails Reviewer's quality check."""
    approved: bool = Field(
        description="True if the content passes all quality, safety, and formatting guardrails, False otherwise."
    )
    feedback: Optional[str] = Field(
        default=None,
        description="Detailed constructive critique outlining what needs revision if approved is False."
    )

# =====================================================================
# 2. Shared Graph State (The Central Memory Matrix)
# =====================================================================

class AgentState(TypedDict):
    """
    The shared state dictionary passed between all nodes in LangGraph.
    Tracks input topic, generated assets, and intermediate agent logic.
    """
    # Inputs
    topic: str
    
    # Context Grounding
    strategy_brief: Optional[StrategyBrief]
    
    # Content Artifacts
    linkedin_draft: Optional[str]
    instagram_caption: Optional[str]
    instagram_visual_brief: Optional[str]  # Scene-by-scene script or carousel layout
    
    # Quality & Flow Control
    review_approved: bool
    feedback: Optional[str]
    loop_count: int  # Tracking loops prevents infinite agent iterations