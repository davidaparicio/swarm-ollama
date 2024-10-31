from swarm_ollama.core import Swarm, Agent

client = Swarm()

MODEL = "llama3.2:3b"
OTHER_MODEL = "mistral-nemo:12b"


def transfer_to_llama(**kwargs):
    """Transfer to llama model."""
    return llama_agent


def transfer_to_mistral(**kwargs):
    """Transfer to mistral model."""
    return mistral_agent


def instructions(me, other):
    return f"You are the {me} model and you are in a rap battle with the {other} model. Make some rhymes explaining why you are the best and why the other model is not."


llama_agent = Agent(
    model=MODEL,
    name="LLama Agent",
    instructions=instructions(MODEL, OTHER_MODEL),
    functions=[transfer_to_mistral],
)

mistral_agent = Agent(
    model=OTHER_MODEL,
    name="Mistral Agent",
    instructions=instructions(OTHER_MODEL, MODEL),
    functions=[transfer_to_llama],
)

messages = [
    {
        "role": "user",
        "content": "Start a rap battle about which model is better, llama3.2:3b or mistral-nemo:12b?",
    }
]
response = client.run(agent=mistral_agent, messages=messages)

for message in response.messages:
    if "role" in message:
        print(f"{message['role']}: {message['content']}")
        print("-" * 50)

response = client.run(agent=llama_agent, messages=messages)

for message in response.messages:
    if "role" in message:
        print(f"{message['role']}: {message['content']}")
        print("-" * 50)
