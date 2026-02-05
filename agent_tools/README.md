# Agent Tools

This directory contains user-defined tools for the agent framework.

## Structure

```
agent_tools/
├── __init__.py          # Exports all tools
├── file_tools.py        # File operation tools (unzip, etc.)
├── web_tools.py         # Web-related tools (search, etc.)
├── data_tools.py        # Data processing tools
└── README.md            # This file
```

## Adding New Tools

1. Create or edit a tool file (e.g., `file_tools.py`)
2. Use the `@tool` decorator from `agent_framework`:

```python
from agent_framework import tool

@tool
def my_tool(param: str) -> str:
    """Tool description for the LLM."""
    # Your tool logic here
    return result
```

3. Export it in `__init__.py`:

```python
from .file_tools import my_tool

__all__ = [
    "my_tool",
    # ... other tools
]
```

## Using Tools with Agents

```python
from agent_framework import Agent, LlmClient
from agent_tools import unzip_file  # Import your tools

# Create agent with tools
agent = Agent(
    model=LlmClient(model="gpt-5-mini"),
    tools=[unzip_file],  # Add your tools here
    instructions="You are a helpful assistant."
)

# Or import all tools
from agent_tools import *
all_tools = [unzip_file, ...]  # List all your tools
```

## Tool Categories

### File Tools (`file_tools.py`)
- `unzip_file` - Extract zip files

### Web Tools (`web_tools.py`)
- (Add web-related tools here)

### Data Tools (`data_tools.py`)
- (Add data processing tools here)

