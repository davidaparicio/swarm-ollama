from swarm_ollama.core import Swarm, Agent


client = Swarm()


def instructions(context_variables):
    context = context_variables.copy()
    tool_call = context.pop("llama_tool_call", "")
    name = context.get("name", "User")
    # Create the system prompt with the function definition
    return f"""You are a helpful agent. Greet the user by name ({name}). You can use the following function to print account details:

    {tool_call}

    If you need to use the function, format your response as. please note that the context variables is already given to you!
    You just need to the information EXACTLY as below:
    [print_account_details({context})]

    Only use the function when necessary when using it ONLY RETURN THE FUNCTION CALL, and provide a natural language response after using it."""


def print_account_details(context_variables: dict):
    user_id = context_variables.get("user_id", None)
    name = context_variables.get("name", None)
    print(f"Account Details: {name} {user_id}")
    return "Success"


llama_tool_call = """[
    {
        "name": "print_account_details",
        "description": "Print the account details for a user",
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "The name of the user"
                },
                "user_id": {
                    "type": "integer",
                    "description": "The user ID"
                }
            },
            "required": ["name", "user_id"]
        }
    }
]"""

context_variables = {
    "name": "James",
    "user_id": 123,
    "llama_tool_call": llama_tool_call,
}
agent = Agent(
    name="Agent",
    model="llama3.2:3b",
    instructions=instructions,
    functions=[print_account_details],
)


response = client.run(
    messages=[{"role": "user", "content": "Hi! what are my account details?"}],
    agent=agent,
    context_variables=context_variables,
)

# Execute the function call
if "[print_account_details(" in response.messages[-1]["content"]:
    result = print_account_details(context_variables)
    print(result)

# print(response.messages[-1]["content"])
