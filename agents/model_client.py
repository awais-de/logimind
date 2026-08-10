"""Shared Anthropic (Claude) model client factory for the agents."""

import os

from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

from agents.planner import Plan

DEFAULT_MODEL = "claude-sonnet-5"

# The two routing tiers for ResponseAgent (see select_response_model below).
# PlannerAgent always runs on DEFAULT_MODEL: it has to decide plan
# complexity before that complexity is known, and a wrong tool/SQL
# decision cascades into every later step, so it isn't routed down.
STRONG_MODEL = DEFAULT_MODEL
CHEAP_MODEL = "claude-haiku-4-5"

# autogen-ext's built-in Anthropic model registry does not yet recognize
# Claude 5 family model names, so capabilities are declared explicitly here
# rather than relying on its internal lookup table.
_MODEL_INFO = ModelInfo(
    vision=False,
    function_calling=True,
    json_output=True,
    family=ModelFamily.CLAUDE_3_5_SONNET,
    structured_output=True,
    multiple_system_messages=False,
)


def build_claude_client(
    model: str = DEFAULT_MODEL, api_key: str | None = None
) -> AnthropicChatCompletionClient:
    """Build a Claude model client for use by the agents.

    Args:
        model: Anthropic model ID to use.
        api_key: Anthropic API key. Falls back to the ANTHROPIC_API_KEY
            environment variable if not given.

    Returns:
        A configured AnthropicChatCompletionClient.
    """
    return AnthropicChatCompletionClient(
        model=model,
        api_key=api_key or os.environ["ANTHROPIC_API_KEY"],
        model_info=_MODEL_INFO,
    )


def select_response_model(plan: Plan) -> str:
    """Pick which model tier ResponseAgent should run on for this query.

    A plan is "complex" -- and gets the stronger model -- if it has more
    than one step, or if any step is a sql_query: multi-step answers need
    to synthesize several sources correctly, and SQL results need to be
    stated exactly rather than paraphrased, both of which benefit from
    the stronger model. Everything else (a single knowledge_search,
    tracking_lookup, or compliance_lookup step, or no steps at all) is
    "simple" and routes to the cheaper, faster tier.

    Args:
        plan: The PlannerAgent's plan for this query.

    Returns:
        CHEAP_MODEL for a simple plan, STRONG_MODEL for a complex one.
    """
    is_complex = len(plan.steps) > 1 or any(step.tool == "sql_query" for step in plan.steps)
    return STRONG_MODEL if is_complex else CHEAP_MODEL
