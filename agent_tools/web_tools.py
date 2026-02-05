"""Web search tools for the agent framework."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from rag.embeddings import get_embeddings, vector_search
from rag.chunking import fixed_length_chunking
# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_framework import tool, ToolResult, ToolCall, ExecutionContext

# Load environment variables
load_dotenv()

# Import optional dependencies
try:
    from tavily import TavilyClient
    TAVILY_AVAILABLE = True
except ImportError:
    TAVILY_AVAILABLE = False


@tool
def search_web(query: str, max_results: int = 5) -> str:
    """Search the web for current information on any topic.
    
    Args:
        query: Search query string
        max_results: Maximum number of results to return (default: 5)
    
    Returns:
        Formatted search results with title, URL, and content for each result.
    
    Example:
        results = search_web("latest AI developments", max_results=3)
    """
    if not TAVILY_AVAILABLE:
        return "Error: tavily-python is required for web search. Install with: pip install tavily-python"
    
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return "Error: TAVILY_API_KEY not found in environment variables. Please set it in your .env file."
    
    try:
        tavily_client = TavilyClient(api_key=api_key)
        response = tavily_client.search(
            query=query,
            max_results=max_results,
        )
        
        results = response.get("results", [])
        if not results:
            return f"No results found for query: {query}"
        
        formatted_results = []
        for r in results:
            formatted_results.append(
                f"Title: {r.get('title', 'N/A')}\n"
                f"URL: {r.get('url', 'N/A')}\n"
                f"Content: {r.get('content', 'N/A')}"
            )
        
        return "\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching web: {str(e)}"

def _extract_search_query(context: ExecutionContext, tool_call_id: str) -> str:
    """Extracts the search query for a specific tool_call_id from context."""
    for event in context.events:
        for item in event.content:
            if (isinstance(item, ToolCall) 
                and item.name == "search_web" 
                and item.tool_call_id == tool_call_id):
                return item.arguments.get("query", "")
    return ""

## callbacks
# def search_compressor(context: ExecutionContext, tool_result: ToolResult):
#     """Callback that compresses web search results."""
#     # Pass through unchanged if not a search tool
#     if tool_result.name != "search_web":
#         return None
    
#     original_content = tool_result.content[0]
    
#     # No compression needed if result is short enough
#     if len(original_content) < 2000:
#         return None
    
#     # Extract search query matching the tool_call_id
#     query = _extract_search_query(context, tool_result.tool_call_id)
#     if not query:
#         return None
    
#     # Use functions implemented in section 5.3
#     chunks = fixed_length_chunking(original_content, chunk_size=500, overlap=50)
#     embeddings = get_embeddings(chunks)
#     results = vector_search(query, chunks, embeddings, top_k=3)
    
#     # Create compressed result
#     compressed = "\n\n".join([r['chunk'] for r in results])
    
#     return ToolResult(
#         tool_call_id=tool_result.tool_call_id,
#         name=tool_result.name,
#         status="success",
#         content=[compressed]
#     )

## callbacks
def search_compressor(context: ExecutionContext, tool_result: ToolResult):
    """Callback that compresses web search results."""
    # Pass through unchanged if not a search tool
    if tool_result.name != "search_web":
        print("DEBUG: Callback skipped - not a search_web tool")
        return None
    
    original_content = tool_result.content[0]
    print(f"DEBUG: Callback triggered! Original content length: {len(original_content)}")
    
    # No compression needed if result is short enough
    if len(original_content) < 2000:
        print("DEBUG: Callback skipped - content too short")
        return None
    
    # Extract search query matching the tool_call_id
    query = _extract_search_query(context, tool_result.tool_call_id)
    if not query:
        print("DEBUG: Callback skipped - could not extract query")
        return None
    
    print(f"DEBUG: Compressing search results for query: {query}")
    # Use functions implemented in section 5.3
    chunks = fixed_length_chunking(original_content, chunk_size=500, overlap=50)
    embeddings = get_embeddings(chunks)
    results = vector_search(query, chunks, embeddings, top_k=3)
    
    # Create compressed result
    compressed = "\n\n".join([r['chunk'] for r in results])
    print(f"DEBUG: Compressed from {len(original_content)} to {len(compressed)} chars")
    
    return ToolResult(
        tool_call_id=tool_result.tool_call_id,
        name=tool_result.name,
        status="success",
        content=[compressed]
    )