# AI Agents: Workshop @ Hyperskill 2025-03-27

## Table of Contents
- Assignment 1: Basic Agent Setup
- Assignment 2: Multi-Agent Conversation
- Assignment 3: Specialized Crew Creation
- Assignment 4: Vector Database with ChromaDB
- Assignment 5: Browser Use
- Assignment 6: Voice Assistant with Vapi

## Installation

```bash
virtualenv venv
source venv/bin/activate
pip install -r requirements.txt
```

For `05_browseruse.py`:
```bash
playwright install
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


