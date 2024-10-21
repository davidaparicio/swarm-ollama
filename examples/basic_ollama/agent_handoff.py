from swarm_ollama.core import Swarm, Agent


client = Swarm()

english_agent = Agent(
    name="English Agent",
    model="llama3.2:3b",
    instructions="You only speak English.",
)

spanish_agent = Agent(
    name="Spanish Agent",
    model="llama3.2:3b",
    instructions="You only speak Spanish.",
)


def transfer_to_spanish_agent():
    """Transfer spanish speaking users immediately."""
    return spanish_agent


llama_tool_call = """[{
        "name": "transfer_to_spanish_agent",
        "description": "Transfer the conversation to a Spanish-speaking agent",
        "parameters": {
            "type": "object",
            "properties": {}
        }
        ]"""

english_agent.functions.append(transfer_to_spanish_agent)
english_agent.instructions = f"""You only speak English. You can use the following functions:

{llama_tool_call}

If you need to use a function, format your response as:
[function_name()]

Only use the functions when necessary, and provide a natural language response after using them.
If the user speaks Spanish or requests Spanish assistance, use the transfer_to_spanish_agent function."""

messages = [{"role": "user", "content": "Hola. ¿Como estás?"}]
response = client.run(agent=english_agent, messages=messages)


if "[transfer_to_spanish_agent(" in response.messages[-1]["content"]:
    transfer_agent = transfer_to_spanish_agent()
    result = client.run(agent=transfer_agent, messages=response.messages)
    print(result.messages[-1]["content"])
# print(response.messages[-1]["content"])
