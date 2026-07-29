"""Shared Anthropic (Claude) model client factory for the agents."""

import os

from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.anthropic import AnthropicChatCompletionClient

DEFAULT_MODEL = "claude-sonnet-5"

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
