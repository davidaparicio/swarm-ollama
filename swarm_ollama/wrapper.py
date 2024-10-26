import json
from httpx import ConnectError
from ollama._types import ResponseError
from dataclasses import dataclass
from typing import Any, Dict, List


@dataclass
class Message:
    content: str
    role: str
    tool_calls: Any = None  # Ollama doesn't support tool calls

    def model_dump_json(self) -> str:
        return json.dumps({"content": self.content, "role": self.role})


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
        self.choices = [
            Choice(
                Message(
                    content=message_data.get("content", ""),
                    role=message_data.get("role", ""),
                )
            )
        ]


class OllamaWrapper:
    """
    Wrap the Ollama client to provide a consistent interface.

    Args:
        client: The Ollama client instance.
    """

    def __init__(self, client):
        self.client = client
        self.chat = self.ChatCompletions(client)

    class ChatCompletions:
        def __init__(self, client):
            self.client = client
            self.completions = self

        def create(
            self,
            model: str,
            messages: List[Dict[str, str]],
            stream: bool = False,
            **kwargs,
        ) -> WrappedResponse:
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

            ollama_kwargs = {
                "model": model,
                "messages": messages,
                "stream": stream,
            }

            try:
                response = self.client.chat(**ollama_kwargs)
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

    def __getattr__(self, name):
        """
        Delegate attribute access to the underlying client.

        Args:
            name (str): The attribute name.

        Returns:
            Any: The attribute from the client.
        """
        return getattr(self.client, name)
