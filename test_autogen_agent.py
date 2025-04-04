import os
from unittest.mock import MagicMock, patch

import pytest
from autogen import AssistantAgent, ConversableAgent, UserProxyAgent

from autogen_agent import EvaluatingUser, get_config_list, setup_agents


@pytest.fixture
def mock_env_vars():
    with patch.dict(
        os.environ,
        {
            "OPENAI_API_KEY": "test-openai-key",
            "GOOGLE_API_KEY": "test-google-key",
            "LLM_PROVIDER": "openai",
        },
    ):
        yield


def test_evaluating_user_initialization():
    """Test EvaluatingUser initialization with default parameters."""
    user = EvaluatingUser(
        name="TestUser",
        code_execution_config={"use_docker": False},  # Disable Docker for tests
    )

    assert user.name == "TestUser"
    assert user.max_stale_turns == 3  # Default value
    assert user.stale_turn_count == 0
    assert not user.solution_found
    assert len(user.evaluation_criteria) == 5  # All criteria present
    assert all(
        not v for v in user.evaluation_criteria.values()
    )  # All criteria start as False


def test_evaluating_user_custom_max_turns():
    """Test EvaluatingUser initialization with custom max_stale_turns."""
    user = EvaluatingUser(
        name="TestUser",
        max_stale_turns=5,
        code_execution_config={"use_docker": False},  # Disable Docker for tests
    )
    assert user.max_stale_turns == 5


def test_evaluate_solution_empty_message():
    """Test solution evaluation with empty message."""
    user = EvaluatingUser(
        name="TestUser",
        code_execution_config={"use_docker": False},  # Disable Docker for tests
    )
    evaluation = user.evaluate_solution("")

    assert not evaluation["is_solution"]
    assert evaluation["score"] == 0
    assert all(
        not v for v in evaluation["criteria_met"].values()
    )  # All criteria should be False


def test_evaluate_solution_complete_message():
    """Test solution evaluation with a message meeting all criteria."""
    message = """
    I recommend using Plotly for visualization:
    1. Easy to use with simple syntax
    2. Interactive features like zooming and hovering
    3. Here's a basic example:
       ```python
       import plotly.express as px
       df = px.data.iris()
       fig = px.scatter(df, x='sepal_width', y='sepal_length')
       fig.show()
       ```
    4. Pros:
       - Rich interactive features
       - Good documentation
       Cons:
       - Larger package size
       - Learning curve for advanced features
    """

    user = EvaluatingUser(
        name="TestUser",
        code_execution_config={"use_docker": False},  # Disable Docker for tests
    )
    evaluation = user.evaluate_solution(message)

    assert evaluation["is_solution"]
    assert evaluation["score"] == 5
    assert len(evaluation["criteria_met"]) == 5


@pytest.mark.parametrize(
    "provider,expected_model",
    [
        ("openai", "gpt-4o-mini"),
        ("gemini", "gemini-2.0-flash-lite"),
    ],
)
def test_get_config_list(mock_env_vars, provider, expected_model):
    """Test LLM configuration for different providers."""
    with patch.dict(os.environ, {"LLM_PROVIDER": provider}):
        config = get_config_list()

        assert isinstance(config, list)
        assert len(config) == 1
        assert config[0]["model"] == expected_model

        if provider == "openai":
            assert config[0]["api_key"] == "test-openai-key"
        else:  # gemini
            assert config[0]["api_key"] == "test-google-key"
            assert config[0]["use_google"] is True


def test_get_config_list_missing_provider(mock_env_vars):
    """Test configuration with missing LLM provider."""
    with patch.dict(os.environ, {"LLM_PROVIDER": ""}, clear=True):
        with pytest.raises(
            ValueError, match="LLM_PROVIDER environment variable not set"
        ):
            get_config_list()


@pytest.mark.parametrize(
    "provider,env_var",
    [
        ("openai", "OPENAI_API_KEY"),
        ("gemini", "GOOGLE_API_KEY"),
    ],
)
def test_get_config_list_missing_api_key(mock_env_vars, provider, env_var):
    """Test configuration with missing API keys."""
    with patch.dict(os.environ, {"LLM_PROVIDER": provider, env_var: ""}, clear=True):
        with pytest.raises(ValueError, match=f"{env_var} environment variable not set"):
            get_config_list()


def test_get_config_list_invalid_provider(mock_env_vars):
    """Test configuration with invalid provider."""
    with patch.dict(os.environ, {"LLM_PROVIDER": "invalid"}):
        with pytest.raises(ValueError, match="Unsupported LLM provider: invalid"):
            get_config_list()


def test_evaluating_user_reply_solution_found():
    """Test user reply when solution is found."""
    user = EvaluatingUser(
        name="TestUser",
        code_execution_config={"use_docker": False},  # Disable Docker for tests
    )
    user.solution_found = True

    messages = [
        {
            "content": "I recommend using Plotly for visualization:\n1. Easy to use\n2. Interactive features\n3. Example code\n4. Pros and cons"
        }
    ]
    reply = user.generate_reply(messages=messages)

    # Verify the reply indicates solution found
    assert reply["terminate"] is True
    assert "meets all my requirements" in reply["content"]


def test_evaluating_user_reply_stale_turns_exceeded():
    """Test user reply when stale turns limit is exceeded."""
    user = EvaluatingUser(
        name="TestUser",
        max_stale_turns=2,
        code_execution_config={"use_docker": False},  # Disable Docker for tests
    )
    # First evaluate a message that won't meet any criteria
    messages = [{"content": "What about matplotlib?"}]
    sender = MagicMock()
    user.on_receive(messages=messages, sender=sender)
    user.stale_turn_count = 3  # Force stale turns exceeded

    # Now test the reply
    reply = user.generate_reply(messages=messages)

    # Verify the reply indicates stale turns exceeded
    assert reply["terminate"] is True
    assert "Time to wrap up" in reply["content"]
    assert "missing information" in reply["content"].lower()


def test_evaluating_user_reply_continue_conversation():
    """Test user reply when conversation should continue."""
    user = EvaluatingUser(
        name="TestUser",
        code_execution_config={"use_docker": False},  # Disable Docker for tests
    )
    # Ensure we're not in a termination state
    user.solution_found = False
    user.stale_turn_count = 1  # Below max_stale_turns

    # Mock parent's generate_reply to return our expected response
    expected_reply = {"content": "Let's continue the discussion", "terminate": False}
    with patch.object(
        UserProxyAgent, "generate_reply", return_value=expected_reply
    ) as mock_reply:
        messages = [{"content": "How about using matplotlib?"}]
        sender = MagicMock()
        config = None
        reply = user.generate_reply(messages=messages, sender=sender, config=config)

        # Verify we called parent's generate_reply with correct arguments
        mock_reply.assert_called_once_with(
            messages=messages, sender=sender, config=config
        )
        # Verify we got the expected continuation response
        assert reply == expected_reply
        assert not reply["terminate"]


@pytest.mark.usefixtures("mock_env_vars")
def test_run_conversation_integration():
    user, assistant = setup_agents()
    problem = """
    We have approximately 100,000 sales records and static data is sufficient.
    We want line charts for monthly trends, bar charts for regional sales, and pie charts for product categories.
    The dashboard will be a web application deployed with Flask.
    The users are sales and marketing staff with limited technical skills, so it must be user-friendly.
    Can you recommend the most suitable Python visualization library and explain why?
    """

    # Mock the assistant's response
    assistant_response = {
        "content": """
        I recommend using Plotly for visualization:
        1. Easy to use with simple syntax
        2. Interactive features like zooming and hovering
        3. Example:
           ```python
           import plotly.express as px
           df = px.data.iris()
           fig = px.scatter(df, x='sepal_width', y='sepal_length')
           fig.show()
           ```
        4. Pros: interactive, well-documented
           Cons: larger package size
        """,
        "terminate": False,
    }

    # Mock only the assistant's responses, let the user evaluate normally
    with patch.object(
        AssistantAgent, "generate_reply", return_value=assistant_response
    ):
        chat_result = user.initiate_chat(
            assistant,
            message=problem,
            max_turns=3,  # Limit to 3 turns for testing
        )

    # Verify the conversation produced expected results
    assert chat_result is not None
    assert len(user.evaluation_history) > 0

    # Check that the assistant's response was evaluated
    best_evaluation = max(user.evaluation_history, key=lambda x: x["score"])
    assert best_evaluation["score"] >= 3  # At least 3 criteria should be met

    # Normalize whitespace and check content
    content = " ".join(assistant_response["content"].split())
    assert "interactive features" in content.lower()


# Configure pytest to ignore specific warnings
@pytest.fixture(autouse=True)
def ignore_warnings():
    import warnings

    with warnings.catch_warnings():
        # Ignore all Pydantic deprecation warnings
        warnings.filterwarnings(
            "ignore", category=DeprecationWarning, module="pydantic.*"
        )
        # Ignore AutoGen API key format warnings
        warnings.filterwarnings(
            "ignore", message=".*API key specified is not a valid OpenAI format.*"
        )
        # Ignore pytest assert rewrite warnings
        warnings.filterwarnings("ignore", category=pytest.PytestAssertRewriteWarning)
        yield
