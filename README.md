# AI Agents: Workshop @ Hyperskill 2025-03-27

## Table of Contents
- Assignment 1: Basic Agent Setup
- Assignment 2: Multi-Agent Conversation
- Assignment 3: Specialized Crew Creation
- Assignment 4: Vector Database with ChromaDB
- Assignment 5: Browser Use
- Assignment 6: Voice Assistant with Vapi

## Installation

### Prerequisites
- Python 3.11.x (tested with 3.11.7)
- pip (Python package installer)
- virtualenv
- make (for Linux/macOS users)

### Setup

#### For Linux/macOS Users
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/workshop-ai-agents-20250327.git
   cd workshop-ai-agents-20250327
   ```

2. **Setup environment for an assignment:**
   ```bash
   make setup-01  # Replace 01 with the assignment number
   ```

3. **Activate the environment:**
   ```bash
   source .venv-01/bin/activate  # Replace 01 with the assignment number
   ```

4. **Run the assignment:**
   Follow the specific instructions provided for each assignment.

5. **Clean environments:**
   ```bash
   make clean
   ```

#### For Windows Users
1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/workshop-ai-agents-20250327.git
   cd workshop-ai-agents-20250327
   ```

2. **Create and activate a virtual environment:**
   ```cmd
   python -m venv .venv-01  # Replace 01 with the assignment number
   .venv-01\Scripts\activate
   ```

3. **Install requirements:**
   ```cmd
   pip install --upgrade pip
   pip install -r requirements-01.txt  # Replace 01 with the assignment number
   ```

4. **Run the assignment:**
   Follow the specific instructions provided for each assignment.

5. **Clean environments manually:**
   ```cmd
   rmdir /s /q .venv-01  # Replace 01 with the assignment number
   ```

### Optional: Makefile Alternative for Windows
Consider creating a `make.bat` file with equivalent commands for easier setup on Windows.

```batch
@echo off
:: Usage: make.bat setup-01
if "%1"=="setup-01" (
    python -m venv .venv-01
    .venv-01\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements-01.txt
)
if "%1"=="clean" (
    rmdir /s /q .venv-*
)
```

## Recommended Workflow (per assignment)

### Setup
```bash
make setup-01  # Create environment for Assignment 1
source .venv-01/bin/activate
make dev       # Install development tools (optional)
```

Running the Assignment

```bash
python 01_langgraph_agent.py  # Assignment 1
deactivate
```

## Development & Testing 
To install development tools:

```bash
make dev # Recommended
```
or manually:

```bash
pip install -r requirements-dev.txt
```

## Using AI models:

### OpenAI

```python
from langchain_openai import ChatOpenAI

OPENAI_API_KEY="sk-..."  # Obtain from https://platform.openai.com/api-keys
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY, temperature=0)
```

### GitHub's OpenAI

```python
from langchain_openai import ChatOpenAI

GITHUB_TOKEN = "..." # Obtain from: https://github.com/marketplace/models/azure-openai/gpt-4o/playground
llm = ChatOpenAI(model="gpt-4o-mini", api_key=GITHUB_TOKEN, base_url="https://models.inference.ai.azure.com/", temperature=0)
```

### OpenRounter's OpenAI

```python
from langchain_openai import ChatOpenAI

OPENROUTER_API_KEY = "..." # Obtain from: https://openrouter.ai/settings/keys
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENROUTER_API_KEY, base_url="https://openrouter.ai/api/v1", temperature=0)
```

### Google Gemini

```python

from langchain_community import ChatGoogleGenerativeAI

GOOGLE_API_KEY = "..." # Obtain from: https://aistudio.google.com/apikey
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", api_key=GOOGLE_API_KEY, temperature=0)
```

## Usage

```bash
python 01_langgraph_agent.py
python 02_crewai_agent.py
python 03_crewai_crew.py
python 04_rag.py
python 05_browseruse.py
python 06_voice.py
```

## Notes on Project Structure

This workshop contains **independent assignments**, each demonstrating a different AI agent framework:

| Assignment | Agent Framework |
|------------|-----------------|
| 01 | LangGraph |
| 02 | AutoGen |
| 03 | CrewAI |
| 04 | LangChain + ChromaDB |
| 05 | Browser-Use Agent |
| 06 | Vapi Voice Assistant |

⚠ **Important**:  
Each assignment may have **incompatible dependency versions** due to rapidly evolving AI agent libraries.

To avoid conflicts, it is recommended to use a **separate virtual environment for each assignment**.
