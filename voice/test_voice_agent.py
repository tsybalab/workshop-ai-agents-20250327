import unittest
from unittest.mock import Mock, patch

import voice_agent


class TestVoiceAgent(unittest.TestCase):
    @patch("voice_agent.requests.post")
    def test_main_call_creation_success(self, mock_post):
        # Mock the response to simulate a successful API call
        mock_response = Mock()
        mock_response.status_code = 201
        mock_post.return_value = mock_response

        # Call the main function
        with patch("builtins.print") as mock_print:
            voice_agent.main()

        # Check if the success message was printed
        mock_print.assert_any_call("✅ Call created successfully.")

    @patch("voice_agent.requests.post")
    def test_main_call_creation_failure(self, mock_post):
        # Mock the response to simulate a failed API call
        mock_response = Mock()
        mock_response.status_code = 400
        mock_response.text = "Error: Bad Request"
        mock_post.return_value = mock_response

        # Call the main function
        with patch("builtins.print") as mock_print:
            voice_agent.main()

        # Check if the failure message was printed
        mock_print.assert_any_call("❌ Failed to create call")
        mock_print.assert_any_call("Error: Bad Request")


if __name__ == "__main__":
    unittest.main()
