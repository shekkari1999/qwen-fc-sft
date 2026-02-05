"""Tool system for the agent framework."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Callable
import inspect
from .models import ExecutionContext
from .utils import function_to_input_schema, format_tool_definition


class BaseTool(ABC):
    """Abstract base class for all tools."""
    
    def __init__(
        self, 
        name: str = None, 
        description: str = None, 
        tool_definition: Dict[str, Any] = None,
        # Confirmation support
        requires_confirmation: bool = False,
        confirmation_message_template: str = None
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
        self.requires_confirmation = requires_confirmation
        self.confirmation_message_template = confirmation_message_template or (
            "The agent wants to execute '{name}' with arguments: {arguments}. "
            "Do you approve?"
        )
    
    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        return self._tool_definition
    
    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass
    
    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)

    def get_confirmation_message(self, arguments: dict[str, Any]) -> str:
        """Generate a confirmation message for this tool call."""
        return self.confirmation_message_template.format(
            name=self.name,
            arguments=arguments
        )
class FunctionTool(BaseTool):
    """Wraps a Python function as a BaseTool."""
    
    def __init__(
        self, 
        func: Callable, 
        name: str = None, 
        description: str = None,
        tool_definition: Dict[str, Any] = None,
        requires_confirmation: bool = False,
        confirmation_message_template: str = None
    ):
        self.func = func
        self.needs_context = 'context' in inspect.signature(func).parameters
        
        self.name = name or func.__name__
        self.description = description or (func.__doc__ or "").strip()
        tool_definition = tool_definition or self._generate_definition()
        
        super().__init__(
            name=self.name, 
            description=self.description, 
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message_template
        )
    
    async def execute(self, context: ExecutionContext = None, **kwargs) -> Any:
        """Execute the wrapped function.
        
        Context is only required if the wrapped function has a 'context' parameter.
        """
        if self.needs_context:
            if context is None:
                raise ValueError(
                    f"Tool '{self.name}' requires a context parameter. "
                    f"Please provide an ExecutionContext instance."
                )
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)
        
        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result
    
    def _generate_definition(self) -> Dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    tool_definition: Dict[str, Any] = None,
    requires_confirmation: bool = False,
    confirmation_message: str = None
):
    """Decorator to convert a function into a FunctionTool.
    
    Usage:
        @tool
        def my_function(x: int) -> int:
            return x * 2
        
        # Or with parameters:
        @tool(name="custom_name", description="Custom description")
        def my_function(x: int) -> int:
            return x * 2
        
        # With confirmation:
        @tool(requires_confirmation=True, confirmation_message="Delete file?")
        def delete_file(filename: str) -> str:
            ...
    """
    from typing import Union
    
    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(
            func=f,
            name=name,
            description=description,
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message
        )
    
    if func is not None:
        return decorator(func)
    return decorator

