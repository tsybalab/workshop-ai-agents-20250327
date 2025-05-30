# AI Agents: Workshop @ Hyperskill 2025-03-27

## Table of Contents
| Assignment | Description                      | Make Target            | Script File                   |
|------------|----------------------------------|------------------------|-------------------------------|
| 1          | Basic Agent Setup                | make setup-langgraph   | langgraph/langgraph_agent.py  |
| 2          | Multi-Agent Conversation         | make setup-autogen     | autogen/autogen_agent.py      |
| 3          | Specialized Crew Creation        | make setup-crewai      | crewai/crewai_agent.py        |
| 4          | Vector Database with ChromaDB    | make setup-rag         | rag/rag_agent.py              |
| 5          | Browser Use                      | make setup-browseruse  | browseruse/browseruse_agent.py|
| 6          | Voice Assistant with Vapi        | make setup-voice       | voice/voice_agent.py          |

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
   make setup-langgraph  # Replace langgraph with the assignment number
   ```

3. **Activate the environment:**
   ```bash
   source .venv-langgraph/bin/activate  # Replace langgraph with the assignment number
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
   python -m venv .venv-langgraph  # Replace langgraph with the assignment number
   .venv-langgraph\Scripts\activate
   ```

3. **Install requirements:**
   ```cmd
   pip install --upgrade pip
   pip install -r langgraph/requirements.txt # Replace langgraph with the assignment number
   ```

4. **Run the assignment:**
   Follow the specific instructions provided for each assignment.

5. **Clean environments manually:**
   ```cmd
   rmdir /s /q .venv-langgraph  # Replace langgraph with the assignment number
   ```

### Optional: Makefile Alternative for Windows
Consider creating a `make.bat` file with equivalent commands for easier setup on Windows.

```batch
@echo off
:: Usage: make.bat setup-langgraph
if "%1"=="setup-langgraph" (
    python -m venv .venv-langgraph
    .venv-langgraph\Scripts\activate
    pip install --upgrade pip
    pip install -r requirements-langgraph.txt
)
if "%1"=="clean" (
    rmdir /s /q .venv-*
)
```

## Recommended Workflow (per assignment)

### Setup
```bash
make setup-langgraph  # Setup for Assignment 1
source .venv-langgraph/bin/activate
make dev       # Install development tools (optional)
```

Running the Assignment

```bash
python langgraph/langgraph_agent.py  # Assignment 1
deactivate
```

## Development & Testing
To install development tools, which include `pytest`, use:

```bash
make dev # Recommended
```
or manually:

```bash
pip install -r requirements-dev.txt
```

## Testing

To run tests for the assignments, use `pytest`. Ensure you have installed the development requirements (which include `pytest`) either by running `make dev` or manually with:

```bash
pip install -r requirements-dev.txt
```

Run tests with:

```bash
pytest
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
python langgraph/langgraph_agent.py
python autogen/autogen_agent.py
python crewai/crewai_agent.py
python rag/rag_agent.py
python browseruse/browseruse_agent.py
python voice/voice_agent.py
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

## Code Formatting

This project uses Black for code formatting and isort for import sorting. The configuration is in `pyproject.toml`.

### VSCode Setup
1. Install the required extensions:
   - Black Formatter (ms-python.black-formatter)
   - Python (ms-python.python)

2. Select the correct Python interpreter:
   - Open any Python file
   - Press Cmd+Shift+P (Mac) or Ctrl+Shift+P (Windows)
   - Type "Python: Select Interpreter"
   - Choose the interpreter from your current assignment's environment (e.g., `.venv-langgraph/bin/python`)

3. The following settings are already configured in the project:
   - Format on save is enabled
   - Black is set as the default formatter
   - Import sorting is configured to use Google style

### IDE Setup

This project uses standard configuration files to ensure consistent formatting across different IDEs:

- `.editorconfig`: Basic editor settings (works in most IDEs)
- `pyproject.toml`: Black and isort configuration

#### VSCode
1. Install extensions:
   - Black Formatter (ms-python.black-formatter)
   - Python (ms-python.python)
   - EditorConfig (editorconfig.editorconfig)

#### PyCharm
1. Enable EditorConfig:
   - Settings → Editor → Code Style
   - Check "Enable EditorConfig support"
2. Set Black as formatter:
   - Settings → Tools → Actions on Save
   - Enable "Reformat code"
   - Set Python formatter to "Black"

#### Sublime Text
1. Install packages:
   - EditorConfig
   - Python Black
2. Enable format on save:
   - Preferences → Package Settings → Python Black → Settings
   - Add: `"on_save": true`

### Command Line Usage
To format files from the command line, first activate your virtual environment:

```bash
source .venv-langgraph/bin/activate   # Match the assignment folder name (e.g., autogen, rag, etc.)
black your_file.py                    # Format a single file
black .                               # Format all Python files in the project
isort .                               # Sort imports in all Python files
```

### Making Changes

1. **Before starting work:**
   ```bash
   git pull                   # Get latest changes
   source .venv-langgraph/bin/activate
   ```

2. **While working:**
   - Let your IDE format on save (VSCode, PyCharm)
   - Or format manually:
     ```bash
     black .
     isort .
     ```

3. **Before committing:**
   ```bash
   # Format all files
   black .
   isort .

   # Review changes
   git status
   git diff

   # Important: Always commit formatting configs together
   # - .editorconfig: Basic editor settings
   # - pyproject.toml: Black and isort settings
   # This ensures all developers use the same formatting

   # Commit
   git add .
   git commit -m "descriptive message"
   git push
   ```
