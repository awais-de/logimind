"""PlannerAgent: decides what information is needed to answer a user query."""

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient
from pydantic import BaseModel

from monitoring.prompt_versions.planner import PLANNER_SYSTEM_PROMPT_V1


class Plan(BaseModel):
    """Structured plan produced by PlannerAgent for a user query.

    Attributes:
        needs_knowledge_search: Whether DHL's knowledge base should be
            searched to answer the query.
        search_query: The query to search with, if needs_knowledge_search.
        needs_tracking_lookup: Whether a shipment tracking lookup is needed.
        tracking_number: The tracking number to look up, if
            needs_tracking_lookup and one was found in the user's message.
    """

    needs_knowledge_search: bool
    search_query: str | None = None
    needs_tracking_lookup: bool
    tracking_number: str | None = None


def build_planner_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Build the PlannerAgent.

    Args:
        model_client: The chat completion client the agent uses.

    Returns:
        An AssistantAgent that produces a Plan for each user query, without
        calling any tools itself.
    """
    return AssistantAgent(
        name="PlannerAgent",
        model_client=model_client,
        system_message=PLANNER_SYSTEM_PROMPT_V1,
        output_content_type=Plan,
    )
