"""Tests for the LangGraph agent implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from langgraph_agent import AgentState, check_continue, get_llm, process_input


def test_agent_state_initialization():
    """Test AgentState initialization with valid data."""
    state = AgentState(messages=[], topic="test", user_input="")
    assert state["messages"] == []
    assert state["topic"] == "test"
    assert state["user_input"] == ""


@pytest.mark.parametrize(
    "user_input,expected",
    [
        ("hello", "continue"),
        ("exit", "__end__"),
        ("EXIT", "__end__"),
        ("Exit", "__end__"),
    ],
)
def test_check_continue(user_input, expected):
    """Test the check_continue function with various inputs."""
    state = AgentState(messages=[], topic="test", user_input=user_input)
    assert check_continue(state) == expected


def test_get_llm_missing_provider():
    """Test get_llm with missing LLM_PROVIDER."""
    with patch.dict(os.environ, clear=True):
        with pytest.raises(
            ValueError, match="LLM_PROVIDER environment variable not set"
        ):
            get_llm()


def test_get_llm_invalid_provider():
    """Test get_llm with invalid provider."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "invalid"}):
        with pytest.raises(ValueError, match="Unsupported LLM provider: invalid"):
            get_llm()


def test_get_llm_openai_missing_key():
    """Test get_llm with OpenAI provider but missing API key."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "openai"}, clear=True):
        with pytest.raises(
            ValueError, match="OPENAI_API_KEY environment variable not set"
        ):
            get_llm()


def test_get_llm_gemini_missing_key():
    """Test get_llm with Gemini provider but missing API key."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "gemini"}, clear=True):
        with pytest.raises(
            ValueError, match="GOOGLE_API_KEY environment variable not set"
        ):
            get_llm()


def test_process_input():
    """Test process_input with mocked LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Test response")

    with patch("langgraph_agent.get_llm", return_value=mock_llm):
        state = AgentState(
            messages=[{"role": "user", "content": "test question"}],
            topic="test",
            user_input="test question",
        )
        new_state = process_input(state)

        assert len(new_state["messages"]) == 2
        assert new_state["messages"][-1]["role"] == "assistant"
        assert new_state["messages"][-1]["content"] == "Test response"

        # Verify LLM was called with correct system message
        call_args = mock_llm.invoke.call_args[0][0]
        assert any("You are an expert on 'test'" in msg["content"] for msg in call_args)
