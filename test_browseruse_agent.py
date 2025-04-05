import unittest
from unittest.mock import AsyncMock, patch

from browseruse_agent import Agent, get_llm, run_agent_with_retries


class TestBrowserUseAgent(unittest.IsolatedAsyncioTestCase):
    @patch("browseruse_agent.get_llm")
    @patch("browseruse_agent.Agent")
    async def test_run_agent_with_retries(self, MockAgent, mock_get_llm):
        # Setup mock
        mock_get_llm.return_value = "mock_llm"
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run = AsyncMock(side_effect=[Exception("Fail"), None])

        # Test
        await run_agent_with_retries(mock_agent_instance, retries=2)

        # Assert
        self.assertEqual(mock_agent_instance.run.call_count, 2)

    @patch("browseruse_agent.get_llm")
    @patch("browseruse_agent.Agent")
    async def test_run_agent_max_retries_exceeded(self, MockAgent, mock_get_llm):
        # Setup mock
        mock_get_llm.return_value = "mock_llm"
        mock_agent_instance = MockAgent.return_value
        mock_agent_instance.run = AsyncMock(side_effect=Exception("Fail"))

        # Test
        with self.assertRaises(Exception) as context:
            await run_agent_with_retries(mock_agent_instance, retries=3)

        # Assert
        self.assertEqual(str(context.exception), "Fail")
        self.assertEqual(mock_agent_instance.run.call_count, 3)


if __name__ == "__main__":
    unittest.main()
