"""Mathematical tools for the agent framework."""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from agent_framework import tool


@tool
def calculator(expression: str) -> float:
    """Calculate mathematical expressions. Supports basic math operations like +, -, *, /, **, etc.
    
    Args:
        expression: Mathematical expression as a string (e.g., "2 + 2", "10 * 5", "100 / 4")
    
    Returns:
        The calculated result as a float.
    
    Example:
        result = calculator("1234 * 5678")  # Returns 7006652.0
    """
    try:
        # Use eval with restricted builtins for safety
        # In production, consider using a safer math parser like 'simpleeval'
        return float(eval(expression))
    except Exception as e:
        return f"Error calculating expression: {str(e)}"

