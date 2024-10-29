import json
from httpx import ConnectError
from ollama._types import ResponseError
from dataclasses import dataclass
from typing import Any, Dict, List
from .logging import setup_logging

logger = setup_logging(__name__)


@dataclass
class Function:
    name: str
    arguments: str  # Must be a JSON string

    @classmethod
    def from_ollama(cls, function_data: dict) -> "Function":
        # Convert dict arguments to JSON string if needed
        arguments = function_data.get("arguments", {})
        if isinstance(arguments, dict):
            arguments = json.dumps(arguments)
        return cls(name=function_data.get("name", ""), arguments=arguments)


@dataclass
class ToolCall:
    id: str
    type: str = "function"
    function: Function = None

    @classmethod
    def from_ollama(cls, tool_call_data: dict, index: int) -> "ToolCall":
        return cls(
            id=tool_call_data.get("id", f"call_{index}"),
            type=tool_call_data.get("type", "function"),
            function=Function.from_ollama(tool_call_data.get("function", {})),
        )


@dataclass
class Message:
    content: str
    role: str
    tool_calls: List[ToolCall] = (
        None  # Most Ollama models currently don't support tool calls
    )

    def model_dump_json(self) -> str:
        data = {
            "content": self.content,
            "role": self.role,
        }
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in self.tool_calls
            ]
        return json.dumps(data)


@dataclass
class Choice:
    message: Message


class WrappedResponse:
    """
    Wrap the Ollama response to provide a consistent interface.

    Args:
        ollama_response (Dict[str, Any]): The response from the Ollama client.
    """

    def __init__(self, ollama_response: Dict[str, Any]):
        message_data = ollama_response.get("message", {})
        message = Message(
            content=message_data.get("content", ""),
            role=message_data.get("role", ""),
        )

        # Handle tool calls if present
        if "tool_calls" in message_data:
            tool_calls = []
            for tc in message_data["tool_calls"]:
                tool_calls.append(
                    ToolCall(
                        id=tc.get("id", f"call_{len(tool_calls)}"),
                        type=tc.get("type", "function"),
                        function=Function.from_ollama(tc.get("function", {})),
                    )
                )
            message.tool_calls = tool_calls
        self.choices = [Choice(message=message)]


class ChatCompletions:
    """
    Initialize the ChatCompletions with an Ollama client.

    Args:
        client: The Ollama client instance.
    """

    def __init__(self, client):
        self.client = client
        self.completions = self

    def create(
        self,
        model: str,
        messages: List[Dict[str, str]],
        stream: bool = False,
        tools: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> WrappedResponse:
        """
        Create a chat completion using the specified model and messages.

        Args:
            model (str): The model name to use (e.g., 'llama2:13b')
            messages (List[Dict[str, str]]): List of conversation messages
            stream (bool, optional): Whether to stream the response. Defaults to False.
            tools (List[Dict[str, Any]], optional): List of tools/functions available. Defaults to None.
            **kwargs: Additional arguments passed to Ollama

        Returns:
            WrappedResponse: The wrapped response from the Ollama client.
        """

        # Clean and format messages
        clean_messages = []
        for msg in messages:
            clean_msg = {"role": msg["role"], "content": msg["content"]}
            clean_messages.append(clean_msg)

        # Any additional kwargs are ignored or can be handled as needed for Ollama (like tools/tool_choice)
        ollama_kwargs = {
            "model": model,
            "messages": clean_messages,
            "stream": stream,
        }

        # Format tools correctly for Ollama
        if tools:
            formatted_tools = []
            for tool in tools:
                formatted_tool = {
                    "type": "function",
                    "function": {
                        "name": tool["function"]["name"],
                        "description": tool["function"].get("description", ""),
                        "parameters": {
                            "type": "object",
                            "properties": tool["function"]["parameters"]["properties"],
                            "required": tool["function"]["parameters"].get(
                                "required", []
                            ),
                        },
                    },
                }
                formatted_tools.append(formatted_tool)
            ollama_kwargs["tools"] = formatted_tools  # Add tools to ollama_kwargs

        try:
            # Debug print to see what we're sending to Ollama
            debug_request = {
                "model": ollama_kwargs["model"],
                "messages": ollama_kwargs["messages"],
                "tools": ollama_kwargs.get("tools", []),
            }
            logger.debug(
                "Sending to Ollama: %s",
                json.dumps(debug_request, indent=2, ensure_ascii=False),
            )

            response = self.client.chat(**ollama_kwargs)
            logger.debug(
                "Received response: %s",
                json.dumps(response, indent=2, ensure_ascii=False),
            )
            # Parse function calls from response content
            if "[" in response.get("message", {}).get("content", ""):
                content = response["message"]["content"]
                # Extract function call if present
                if "[" in content and "]" in content:
                    function_call = content[content.find("[") + 1 : content.find("]")]
                    if "(" in function_call and ")" in function_call:
                        func_name = function_call.split("(")[0].strip()
                        func_args = function_call.split("(")[1].split(")")[0].strip()
                        # Create tool call structure
                        tool_call = {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": func_name,
                                "arguments": "{}" if not func_args else func_args,
                            },
                        }
                        # Clean content and add tool calls
                        response["message"]["content"] = content.replace(
                            f"[{function_call}]", ""
                        ).strip()
                        response["message"]["tool_calls"] = [tool_call]

        except ResponseError as e:
            raise NameError(f"LLM model error: {e}")
        except ConnectError as e:
            raise ConnectionError(
                f"Connection error occurred.. Is the `ollama serve` running?: {e}"
            )
        except Exception as e:
            # logger.error("Unexpected error: %s", str(e), exc_info=True)
            raise RuntimeError(f"Failed to get chat response: {e}")
        return WrappedResponse(response)


class OllamaWrapper:
    """
    Wrap the Ollama client to provide a consistent interface.

    Args:
        client: The Ollama client instance.
    """

    def __init__(self, client):
        self.client = client
        self.chat = ChatCompletions(client)

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying client.

        Args:
            name (str): The attribute name.

        Returns:
            Any: The attribute from the client.
        """
        return getattr(self.client, name)
