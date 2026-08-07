"""PlannerAgent: decides what information is needed to answer a user query."""

import re
from typing import Literal

from autogen_agentchat.agents import AssistantAgent
from autogen_core.models import ChatCompletionClient, RequestUsage
from pydantic import BaseModel, ValidationError

from monitoring.prompt_versions.planner import PLANNER_SYSTEM_PROMPT_V3


class Step(BaseModel):
    """One action within a Plan.

    Attributes:
        tool: Which tool this step calls.
        search_query: Query text for a "knowledge_search" step. May
            reference an earlier step's result via a `{{step_N.field}}`
            placeholder (1-indexed), resolved against that step's result
            by run_retriever before the tool is called.
        tracking_number: Tracking number for a "tracking_lookup" step. May
            also use a `{{step_N.field}}` placeholder.
        category: Item category or classification for a
            "compliance_lookup" step.
        destination: Destination country/region code for a
            "compliance_lookup" step. May also use a `{{step_N.field}}`
            placeholder.
    """

    tool: Literal["knowledge_search", "tracking_lookup", "compliance_lookup"]
    search_query: str | None = None
    tracking_number: str | None = None
    category: str | None = None
    destination: str | None = None


class Plan(BaseModel):
    """Structured plan produced by PlannerAgent for a user query.

    Attributes:
        steps: Ordered actions needed to answer the query, empty if none
            are needed. A later step may depend on an earlier one's result
            via a `{{step_N.field}}` placeholder in its own fields.
    """

    steps: list[Step] = []


def build_planner_agent(model_client: ChatCompletionClient) -> AssistantAgent:
    """Build the PlannerAgent.

    Args:
        model_client: The chat completion client the agent uses.

    Returns:
        An AssistantAgent that responds with a JSON-encoded Plan for each
        user query, without calling any tools itself.

    Note:
        Structured output (output_content_type) isn't used here: AutoGen's
        Anthropic client raises "Structured output is currently not
        supported for Anthropic models" if a Pydantic type is passed as
        json_output. The system prompt instead instructs the model to
        respond with plain JSON text, parsed by parse_plan.
    """
    return AssistantAgent(
        name="PlannerAgent",
        model_client=model_client,
        system_message=PLANNER_SYSTEM_PROMPT_V3,
    )


def parse_plan(raw_text: str) -> Plan:
    """Parse a Plan out of the PlannerAgent's raw text response.

    Args:
        raw_text: The agent's response text, expected to be a JSON object,
            optionally wrapped in a markdown code fence.

    Returns:
        The parsed Plan.

    Raises:
        ValueError: If no JSON object is found, or it doesn't match Plan's
            schema.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if match is None:
        raise ValueError(f"No JSON object found in planner response: {raw_text!r}")

    try:
        return Plan.model_validate_json(match.group(0))
    except ValidationError as exc:
        raise ValueError(f"Planner response did not match Plan schema: {raw_text!r}") from exc


async def run_planner(agent: AssistantAgent, message: str) -> tuple[Plan, RequestUsage | None]:
    """Run the PlannerAgent on a user message and parse its Plan.

    Args:
        agent: A PlannerAgent built by build_planner_agent.
        message: The user's message.

    Returns:
        A tuple of the parsed Plan and the model's token usage for this
        call (None if the client didn't report it), for cost tracking.
    """
    result = await agent.run(task=message)
    last_message = result.messages[-1]
    return parse_plan(last_message.content), last_message.models_usage
