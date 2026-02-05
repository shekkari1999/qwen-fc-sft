"""Example: How to use agent_tools with your agent."""

import asyncio
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent_framework import Agent, LlmClient
from agent_tools import unzip_file, list_files, read_file, read_media_file
import asyncio
from agent_framework import Agent, LlmClient
from agent_tools import search_web, list_files, read_file
from agent_tools.file_tools import delete_file
from agent_tools.web_tools import search_compressor
from agent_framework.agent import approval_callback

async def main():
    agent = Agent(
        model=LlmClient(model="gpt-5-mini"),  # Use a valid model name
        tools=[search_web, list_files, read_file, delete_file],
        instructions="You are a helpful assistant that can search the web and explore files to answer questions.",
        max_steps=20,
        before_tool_callbacks=[approval_callback],
        after_tool_callbacks=[search_compressor],
    )
    
    result = await agent.run("search about andrej karpathy")   
    print(result.output)

if __name__ == "__main__":
    asyncio.run(main())