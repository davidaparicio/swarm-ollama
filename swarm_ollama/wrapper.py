import json
from httpx import ConnectError
from ollama._types import ResponseError
from dataclasses import dataclass
from typing import Any, Dict, List


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
        # return json.dumps({"content": self.content, "role": self.role})
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
        # message_data = ollama_response.get("message", {})
        # self.choices = [
        #     Choice(
        #         Message(
        #             content=message_data.get("content", ""),
        #             role=message_data.get("role", ""),
        #         )
        #     )
        # ]


class ChatCompletions:
    """
    Initialize the ChatCompletions with an Ollama client.

    Args:
        client: The Ollama client instance.
    """

    def __init__(self, client):
        self.client = client
        self.completions = self

    # def create(
    #     self, model: str, messages: List[Dict[str, str]], stream: bool = False, **kwargs
    # ) -> WrappedResponse:
    def create(self, **kwargs) -> WrappedResponse:
        """
        Create a chat completion using the specified model and messages.

        Args:
            model (str): The model name.
            messages (List[Dict[str, str]]): The conversation messages.
            stream (bool, optional): Whether to stream the response. Defaults to False.

        Returns:
            WrappedResponse: The wrapped response from the Ollama client.
        """

        # Any additional kwargs are ignored or can be handled as needed (like tools)

        # Clean and format messages
        messages = []
        for msg in kwargs.get("messages", []):
            clean_msg = {"role": msg["role"], "content": msg["content"]}
            messages.append(clean_msg)
        # messages = []
        # for msg in kwargs.get("messages", []):
        #     clean_msg = {"role": msg["role"], "content": msg["content"]}

        #     # Handle tool calls differently
        #     if "tool_calls" in msg:
        #         clean_msg["tool_calls"] = []
        #         for tc in msg["tool_calls"]:
        #             tool_call = {
        #                 "id": tc.get("id", ""),
        #                 "type": "function",
        #                 "function": {
        #                     "name": tc["function"]["name"],
        #                     "arguments": tc["function"]["arguments"]
        #                 }
        #             }
        #             clean_msg["tool_calls"].append(tool_call)

        #     # Handle tool responses
        #     if msg.get("role") == "tool":
        #         clean_msg = {
        #             "role": "assistant",
        #             "content": msg["content"]
        #         }

        #     messages.append(clean_msg)

        # ollama_kwargs = {
        #     "model": model,
        #     "messages": messages,
        #     #"tools": tools or None,
        #     #"tool_choice": self.client.tool_choice,
        #     "stream": stream,
        # }

        ollama_kwargs = {
            "model": kwargs.get("model"),
            "messages": messages,
            "stream": kwargs.get("stream", False),
            # Remove tool_choice as it's not supported by Ollama
        }

        # Format tools correctly for Ollama
        if kwargs.get("tools"):
            formatted_tools = []
            for tool in kwargs["tools"]:
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
            print("Sending to Ollama:", json.dumps(debug_request, indent=2))

            response = self.client.chat(**ollama_kwargs)

            # If response contains tool calls, ensure they're properly formatted
            # if "tool_calls" in response.get("message", {}):
            #     for tool_call in response["message"]["tool_calls"]:
            #         if isinstance(tool_call["function"]["arguments"], dict):
            #             tool_call["function"]["arguments"] = json.dumps(
            #                 tool_call["function"]["arguments"]
            #             )

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

            # Parse function calls from response content
            # if "tool_calls" in response.get("message", {}):
            #     tool_calls = []
            #     for idx, tool_call in enumerate(response["message"]["tool_calls"]):
            #         tool_calls.append(
            #             ToolCall(
            #                 id=tool_call.get("id", f"call_{idx}"),
            #                 type=tool_call.get("type", "function"),
            #                 function=Function(
            #                     name=tool_call["function"]["name"],
            #                     arguments=tool_call["function"]["arguments"]
            #                 )
            #             )
            #         )
            #     response["message"]["tool_calls"] = tool_calls

            # if self.tools and "[" in response.get("message", {}).get("content", ""):
            #     content = response["message"]["content"]
            #     # Extract function call if present
            #     if "[" in content and "]" in content:
            #         function_call = content[content.find("[")+1:content.find("]")]
            #         if "(" in function_call and ")" in function_call:
            #             func_name = function_call.split("(")[0].strip()
            #             func_args = function_call.split("(")[1].split(")")[0].strip()

            #             # Create tool call structure
            #             tool_call = ToolCall(
            #                 id="call_1",  # Simple ID for now
            #                 function=Function(
            #                     name=func_name,
            #                     arguments="{}" if not func_args else func_args
            #                 )
            #             )

            #             # Clean content and add tool calls
            #             response["message"]["content"] = content.replace(f"[{function_call}]", "").strip()
            #             response["message"]["tool_calls"] = [tool_call]

            # response.raise_for_status()
        except ResponseError as e:
            raise NameError(f"LLM model error: {e}")
        except ConnectError as e:
            raise ConnectionError(
                f"Connection error occurred.. Is the `ollama serve` running?: {e}"
            )
        # except HTTPStatusError as e:
        #   raise ConnectionError(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
        except Exception as e:
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
