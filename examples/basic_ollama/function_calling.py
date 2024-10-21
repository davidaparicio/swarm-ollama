from swarm_ollama.core import Swarm, Agent


client = Swarm()


def get_weather(location) -> str:
    return "{'temp':67, 'unit':'F'}"


# Define the function in the format expected by Llama 3.2
llama_tool_call = """[
    {
        "name": "get_weather",
        "description": "Get the weather for a specific location",
        "parameters": {
            "type": "object",
            "required": ["location"],
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The location to get weather for"
                }
            }
        }
    }
]"""

# Create the system prompt with the function definition
instructions = f"""You are a helpful agent. You can use the following function to get weather information:

{llama_tool_call}

If you need to use the function, format your response as:
[get_weather(location="<location>")]

Only use the function when necessary, and provide a natural language response after using it."""

agent = Agent(
    name="Agent",
    model="llama3.2:3b",
    instructions=instructions,
    functions=[
        get_weather
    ],  # We still pass the function, but it won't be used directly by Llama 3.2
)

messages = [{"role": "user", "content": "What's the weather in NYC?"}]

response = client.run(agent=agent, messages=messages)
# print(response.messages[-1]["content"])

# Parse the response to check if the function was called
response_content = response.messages[-1]["content"]
if "[get_weather(" in response_content:
    # Extract the function call
    start = response_content.index("[get_weather(")
    end = response_content.index(")]", start) + 2
    function_call = response_content[start:end]

    # Execute the function
    exec(f"result = {function_call}")
    weather_data = eval("result")

    # Update the response with the actual weather data
    updated_response = response_content.replace(
        function_call, f"The weather in NYC is {weather_data}"
    )
    print(updated_response)
else:
    print("Function was not called in the response")
