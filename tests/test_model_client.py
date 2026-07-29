"""Unit tests for agents/model_client.py."""

from autogen_ext.models.anthropic import AnthropicChatCompletionClient

from agents.model_client import DEFAULT_MODEL, build_claude_client


def test_build_claude_client_with_explicit_args() -> None:
    client = build_claude_client(model="claude-opus-5", api_key="dummy-test-key")

    assert isinstance(client, AnthropicChatCompletionClient)


def test_build_claude_client_reads_api_key_from_env(monkeypatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "dummy-test-key")

    client = build_claude_client()

    assert isinstance(client, AnthropicChatCompletionClient)


def test_build_claude_client_uses_default_model_name() -> None:
    client = build_claude_client(api_key="dummy-test-key")

    assert client._create_args.get("model") == DEFAULT_MODEL
