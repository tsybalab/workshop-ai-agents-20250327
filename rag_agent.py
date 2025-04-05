"""
Assignment 4: Vector Database with ChromaDB
Implement an AI agent that uses ChromaDB as a vector database for storing and retrieving information.

# TODO:
1. Use PDF or DOCX instead of hardcoded sample_documents
"""

import argparse
import os
import sys
import uuid
from enum import Enum
from typing import List, Optional

import chromadb
from chromadb.utils import embedding_functions
from docx import Document as DocxDocument
from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from pypdf import PdfReader

# === Environment Setup ===
load_dotenv()


# === File Type Handling ===
class FileType(Enum):
    PDF = "pdf"
    DOCX = "docx"


EXTENSION_MAP = {
    ".pdf": FileType.PDF,
    ".docx": FileType.DOCX,
    ".doc": FileType.DOCX,
}


def get_file_type(file_path: str) -> Optional[FileType]:
    ext = os.path.splitext(file_path)[1].lower()
    return EXTENSION_MAP.get(ext)


def read_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def read_docx(file_path: str) -> str:
    doc = DocxDocument(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


# === LLM Initialization ===
def get_llm():
    llm_provider = os.getenv("LLM_PROVIDER")
    if not llm_provider:
        raise ValueError("LLM_PROVIDER environment variable not set")

    if llm_provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        return ChatOpenAI(model="gpt-4o-mini", api_key=api_key, temperature=0.7)

    elif llm_provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not set")
        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash", api_key=api_key, temperature=0.7
        )

    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider}")


# === Vector Agent ===
class AgentWithChromaDB:
    def __init__(self, collection_name="documents"):
        self.llm = get_llm()
        self.client = chromadb.Client()
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
        )
        self.collection = self._get_or_create_collection(collection_name)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )

    def _get_or_create_collection(self, name):
        try:
            return self.client.get_collection(
                name=name, embedding_function=self.embedding_function
            )
        except Exception as e:
            if "does not exist" in str(e):
                print(f"Collection '{name}' does not exist. Creating a new one.")
            else:
                print(f"Error retrieving collection '{name}': {e}")
            return self.client.create_collection(
                name=name, embedding_function=self.embedding_function
            )

    def ingest_document(self, content: str, metadata: dict = None):
        metadata = metadata or {}
        chunks = self.text_splitter.split_text(content)
        ids = [str(uuid.uuid4()) for _ in chunks]
        metadatas = [
            {**metadata, "chunk": i, "total_chunks": len(chunks)}
            for i in range(len(chunks))
        ]
        self.collection.add(documents=chunks, ids=ids, metadatas=metadatas)
        print(f"Ingested {len(chunks)} chunks from: {metadata.get('filename')}")

    def search_documents(self, query, n_results=3):
        results = self.collection.query(query_texts=[query], n_results=n_results)
        documents = []
        for i, text in enumerate(results.get("documents", [[]])[0]):
            documents.append(
                {
                    "text": text,
                    "metadata": results["metadatas"][0][i],
                    "id": results["ids"][0][i],
                    "distance": results["distances"][0][i],
                }
            )
        return documents

    def answer_question(self, question):
        docs = self.search_documents(question)
        if not docs:
            return "No information available to answer your question."

        context = "\n\n".join(
            f"Document {i+1} ({', '.join(f'{k}: {v}' for k, v in doc['metadata'].items() if k not in ['chunk', 'total_chunks'])}):\n{doc['text']}"
            for i, doc in enumerate(docs)
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessagePromptTemplate.from_template(
                    "You are a helpful assistant. Use the provided context to answer the question. If unknown, say so."
                ),
                HumanMessagePromptTemplate.from_template(
                    f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"
                ),
            ]
        )

        return self.llm.invoke(prompt.format_messages()).content


# === CLI Execution ===
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="📄 Ingest PDF and DOCX files into ChromaDB for semantic search using an AI agent.",
        epilog="💡 Example: python agent_with_chromadb.py document.pdf report.docx",
    )

    parser.add_argument(
        "file_paths", nargs="*", help="Path(s) to .pdf, .docx or .doc file(s) to ingest"
    )
    args = parser.parse_args()

    # If no file paths are provided, print help and exit
    if not args.file_paths:
        parser.print_help()
        print("\nℹ️  Please provide at least one file to ingest.")
        sys.exit(0)

    return args


def main():
    args = parse_arguments()
    agent = AgentWithChromaDB()

    for path in args.file_paths:
        if not os.path.exists(path):
            print(f"❌ File not found: {path}")
            continue
        file_type = get_file_type(path)
        if file_type == FileType.PDF:
            text = read_pdf(path)
        elif file_type == FileType.DOCX:
            text = read_docx(path)
        else:
            print(f"⚠️ Unsupported file type: {path}")
            continue

        agent.ingest_document(
            text, {"filename": os.path.basename(path), "type": file_type.value}
        )

    print("\n✅ Agent is ready. Type a question or 'exit' to quit.")

    while True:
        query = input("\nYou: ")
        if query.lower() == "exit":
            break
        print(f"\nAgent: {agent.answer_question(query)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Exiting gracefully. Bye!")
