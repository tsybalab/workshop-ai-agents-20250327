import unittest
from unittest.mock import MagicMock, patch

from rag_agent import AgentWithChromaDB, FileType, get_file_type


class TestAgentWithChromaDB(unittest.TestCase):

    @patch("rag_agent.chromadb.Client")
    def setUp(self, MockClient):
        # Mock the ChromaDB client
        self.mock_client = MockClient.return_value
        self.agent = AgentWithChromaDB()

    def test_get_file_type(self):
        self.assertEqual(get_file_type("document.pdf"), FileType.PDF)
        self.assertEqual(get_file_type("report.docx"), FileType.DOCX)
        self.assertIsNone(get_file_type("unsupported.txt"))

    def test_ingest_document(self):
        # Mock the embedding function
        self.agent.embedding_function = MagicMock()

        # Test ingestion
        text = "Sample document text."
        metadata = {"filename": "sample.pdf", "type": "pdf"}
        self.agent.ingest_document(text, metadata)

        # Verify that the document was added
        self.agent.collection.add = MagicMock()
        self.agent.ingest_document("Sample text", {"filename": "sample.pdf"})
        self.agent.collection.add.assert_called_once()

    @patch("rag_agent.ChatOpenAI")
    def test_answer_question(self, MockChatOpenAI):
        # Create a mocked LLM instance
        mock_llm = MockChatOpenAI.return_value
        mock_llm.invoke.return_value.content = "Mocked answer."

        # Inject mocked LLM into the agent
        self.agent.llm = mock_llm

        # Patch search_documents to simulate retrieved content
        self.agent.search_documents = MagicMock(
            return_value=[
                {
                    "text": "Sample content",
                    "metadata": {"filename": "mocked.docx"},
                    "id": "test-id",
                    "distance": 0.01,
                }
            ]
        )

        # Call the method
        answer = self.agent.answer_question("What is the content of the document?")

        # Assert the mocked response is returned
        self.assertEqual(answer, "Mocked answer.")


if __name__ == "__main__":
    unittest.main()
