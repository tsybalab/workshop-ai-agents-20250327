"""
Assignment 4: Vector Database with ChromaDB
Implement an AI agent that uses ChromaDB as a vector database for storing and retrieving information.

# TODO:
1. Use PDF or DOCX instead of hardcoded sample_documents
"""

import os
import uuid
from typing import List, Dict, Any
from dotenv import load_dotenv
import chromadb
from chromadb.utils import embedding_functions
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.schema import Document

# Load environment variables
load_dotenv()

class AgentWithChromaDB:
    def __init__(self, collection_name="documents"):
        """
        Initialize the agent with ChromaDB for vector storage and retrieval.
        
        Args:
            collection_name: Name of the ChromaDB collection to use
        """
        # Initialize the LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.7,
        )
        
        # Initialize ChromaDB client
        self.client = chromadb.Client()
        
        # Set up the embedding function
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        
        # Create or get the collection
        self.collection = self._get_or_create_collection(collection_name)
        
        # Initialize the text splitter for chunking documents
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
    
    def _get_or_create_collection(self, collection_name):
        """
        Get an existing collection or create a new one if it doesn't exist.
        """
        try:
            # Try to get an existing collection
            collection = self.client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Using existing collection: {collection_name}")
        except:
            # Create a new collection if it doesn't exist
            collection = self.client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
            print(f"Created new collection: {collection_name}")
        
        return collection
    
    def ingest_document(self, document_text, metadata=None):
        """
        Ingest a document into the vector database.
        
        Args:
            document_text: The text content of the document
            metadata: Optional metadata for the document
            
        Returns:
            List of document IDs for the ingested chunks
        """
        if metadata is None:
            metadata = {}
        
        # Split the document into chunks
        chunks = self.text_splitter.split_text(document_text)
        
        # Generate unique IDs for each chunk
        doc_ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        
        # Prepare metadata for each chunk
        metadatas = [{**metadata, "chunk": i, "total_chunks": len(chunks)} for i in range(len(chunks))]
        
        # Add the chunks to the collection
        self.collection.add(
            documents=chunks,
            ids=doc_ids,
            metadatas=metadatas
        )
        
        print(f"Ingested document with {len(chunks)} chunks")
        return doc_ids
    
    def search_documents(self, query, n_results=3):
        """
        Search for relevant document chunks based on a query.
        
        Args:
            query: The search query
            n_results: Number of results to return
            
        Returns:
            List of relevant document chunks with their metadata
        """
        # Perform similarity search
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        
        # Format the results
        documents = []
        if results and results['documents'] and results['documents'][0]:
            for i, doc_text in enumerate(results['documents'][0]):
                documents.append({
                    "text": doc_text,
                    "metadata": results['metadatas'][0][i] if results['metadatas'] and results['metadatas'][0] else {},
                    "id": results['ids'][0][i] if results['ids'] and results['ids'][0] else None,
                    "distance": results['distances'][0][i] if results['distances'] and results['distances'][0] else None
                })
        
        return documents
    
    def answer_question(self, question):
        """
        Answer a question using information retrieved from the vector database.
        
        Args:
            question: The user's question
            
        Returns:
            The agent's response to the question
        """
        # Search for relevant documents
        relevant_docs = self.search_documents(question)
        
        if not relevant_docs:
            return "I don't have enough information to answer that question."
        
        # Create a context from the relevant documents
        context_parts = []
        for i, doc in enumerate(relevant_docs):
            # Add document text with metadata information
            metadata_str = ", ".join([f"{k}: {v}" for k, v in doc["metadata"].items() if k != "chunk" and k != "total_chunks"])
            context_parts.append(f"Document {i+1} ({metadata_str}):\n{doc['text']}")
        
        context = "\n\n".join(context_parts)
        
        # Create a prompt template
        prompt = ChatPromptTemplate.from_messages([
            SystemMessagePromptTemplate.from_template(
                "You are a helpful assistant that answers questions based on the provided context. "
                "If the answer cannot be found in the context, acknowledge that you don't know. "
                "Cite the specific documents you used to formulate your answer."
            ),
            HumanMessagePromptTemplate.from_template(
                f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
            )
        ])
        
        # Generate the answer
        response = self.llm.invoke(prompt.format_messages())
        
        return response.content

def main():
    # Create the agent with ChromaDB
    agent = AgentWithChromaDB()
    
    # Sample documents to ingest
    sample_documents = [
        {
            "text": """
            Python is a high-level, interpreted programming language known for its readability and versatility.
            It was created by Guido van Rossum and first released in 1991. Python supports multiple programming
            paradigms, including procedural, object-oriented, and functional programming. It has a comprehensive
            standard library and a large ecosystem of third-party packages.
            """,
            "metadata": {"title": "Python Programming Language", "category": "Programming"}
        },
        {
            "text": """
            Machine learning is a subset of artificial intelligence that focuses on developing systems that can
            learn from and make decisions based on data. Common machine learning techniques include supervised
            learning, unsupervised learning, and reinforcement learning. Popular machine learning libraries in
            Python include TensorFlow, PyTorch, and scikit-learn.
            """,
            "metadata": {"title": "Machine Learning Basics", "category": "AI"}
        },
        {
            "text": """
            Vector databases store data as high-dimensional vectors and enable efficient similarity search.
            They are particularly useful for applications involving natural language processing, image recognition,
            and recommendation systems. ChromaDB is an open-source vector database that provides simple APIs for
            storing and retrieving vectors, making it ideal for building AI applications.
            """,
            "metadata": {"title": "Vector Databases", "category": "Databases"}
        }
    ]
    
    print("Ingesting sample documents...")
    for doc in sample_documents:
        agent.ingest_document(doc["text"], doc["metadata"])
    
    print("Agent with ChromaDB is ready. Type 'exit' to quit.")
    print("Ask questions about the ingested documents.")
    
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == "exit":
            break
        
        # Process the input and get a response
        response = agent.answer_question(user_input)
        
        # Print the agent's response
        print(f"\nAgent: {response}")

if __name__ == "__main__":
    main()
