"""Tests for the LangGraph agent implementation."""

import os
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage

from langgraph_agent import (
    AgentState,
    check_continue,
    get_llm,
    process_input,
    search_ddg,
)


def test_agent_state_initialization():
    """Test AgentState initialization with valid data."""
    state = AgentState(messages=[], topic="test", user_input="", search_results="")
    assert state["messages"] == []
    assert state["topic"] == "test"
    assert state["user_input"] == ""
    assert state["search_results"] == ""


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


def test_search_ddg_success():
    """Test search_ddg function with mocked search."""
    mock_search = MagicMock()
    mock_search.run.return_value = "Test search results"
    with patch("langgraph_agent.DuckDuckGoSearchRun", return_value=mock_search):
        state = AgentState(
            messages=[],
            topic="test topic",
            user_input="test input",
            search_results="",
        )
        new_state = search_ddg(state)

        # Check search results are stored
        assert new_state["search_results"] == "Test search results"

        # Verify search query
        mock_search.run.assert_called_once_with("test topic test input")

        # Check formatted message
        assert len(new_state["messages"]) == 1
        system_message = new_state["messages"][0]
        assert system_message["role"] == "system"
        assert "[SEARCH_RESULTS]" in system_message["content"]
        assert "Query: 'test topic test input'" in system_message["content"]
        assert "Results:\nTest search results" in system_message["content"]


def test_search_ddg_failure():
    """Test search_ddg function when search fails."""
    mock_search = MagicMock()
    mock_search.run.side_effect = Exception("Search failed")
    with patch("langgraph_agent.DuckDuckGoSearchRun", return_value=mock_search):
        state = AgentState(
            messages=[],
            topic="test topic",
            user_input="test input",
            search_results="",
        )
        new_state = search_ddg(state)
        # Check that search results are empty
        assert new_state["search_results"] == ""
        # Verify no messages were added
        assert len(new_state["messages"]) == 0


def test_process_input():
    """Test process_input function with mocked LLM."""
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="🔍 Based on my search, here's what I found..."
    )
    with patch("langgraph_agent.get_llm", return_value=mock_llm):
        state = AgentState(
            messages=[
                {
                    "role": "system",
                    "content": "[SEARCH_RESULTS]\nQuery: 'test query'\n\nResults: Test search results",
                },
                {"role": "user", "content": "test input"},
            ],
            topic="test topic",
            user_input="test input",
            search_results="Test search results",
        )
        new_state = process_input(state)

        # Check assistant's response was added
        assert len(new_state["messages"]) == 3  # system, user, and assistant messages
        assert new_state["messages"][-1]["role"] == "assistant"
        assert "🔍 Based on my search" in new_state["messages"][-1]["content"]

        # Verify system message format
        messages_to_llm = mock_llm.invoke.call_args[0][0]
        system_message = messages_to_llm[0]["content"]
        assert "You are an expert on 'test topic'" in system_message
        assert "When using search results:" in system_message
        assert "Quote specific information" in system_message

        # Verify search results were included
        assert any("[SEARCH_RESULTS]" in msg["content"] for msg in messages_to_llm)
