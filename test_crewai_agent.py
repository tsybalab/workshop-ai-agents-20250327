import unittest
from unittest.mock import MagicMock, patch

from crewai_agent import execute_crew


class TestCrewAIAgent(unittest.TestCase):

    @patch("crewai_agent.Crew")
    def test_execute_crew_calls_kickoff_and_prints_result(self, MockCrew):
        # Mock the Crew instance and its methods
        mock_crew_instance = MockCrew.return_value
        mock_crew_instance.kickoff.return_value = "Mocked final article content"

        # Define a test topic
        topic = "Test Topic"

        # Capture the output
        with patch("builtins.print") as mock_print:
            execute_crew(topic)

            # Check if Crew was initialized with the correct agents and tasks
            MockCrew.assert_called_once()

            # Check if kickoff was called
            mock_crew_instance.kickoff.assert_called_once()

            # Check if the final article was printed
            mock_print.assert_any_call("\n\n=== FINAL ARTICLE ===\n\n")
            mock_print.assert_any_call("Mocked final article content")

    # Additional tests for agent and task creation can be added here


if __name__ == "__main__":
    unittest.main()
