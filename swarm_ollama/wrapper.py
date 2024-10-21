import json


class OllamaWrapper:
    def __init__(self, client):
        self.client = client
        self.chat = self.ChatCompletions(client)

    class ChatCompletions:
        def __init__(self, client):
            self.client = client
            self.completions = self

        def create(self, **kwargs):
            # Map Swarm parameters to Ollama parameters
            ollama_kwargs = {
                "model": kwargs.get("model"),
                "messages": kwargs.get("messages"),
                "stream": kwargs.get("stream", False),
            }

            response = self.client.chat(**ollama_kwargs)

            # Wrap the Ollama response to match OpenAI's structure
            class WrappedResponse:
                def __init__(self, ollama_response):
                    self.choices = [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": ollama_response["message"][
                                            "content"
                                        ],
                                        "role": ollama_response["message"]["role"],
                                        "tool_calls": None,  # Ollama doesn't support tool calls
                                        "model_dump_json": lambda: json.dumps(
                                            {
                                                "content": ollama_response["message"][
                                                    "content"
                                                ],
                                                "role": ollama_response["message"][
                                                    "role"
                                                ],
                                            }
                                        ),
                                    },
                                )
                            },
                        )
                    ]

            return WrappedResponse(response)

    def __getattr__(self, name):
        return getattr(self.client, name)
