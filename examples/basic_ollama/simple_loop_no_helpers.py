from swarm_ollama.core import Swarm, Agent


client = Swarm()

my_agent = Agent(
    model="llama3.2:3b",
    name="Agent",
    instructions="You are a helpful agent.",
)


def pretty_print_messages(messages):
    for message in messages:
        if message["content"] is None:
            continue
        print(
            f"{message['role']}: {message['content']}"
        )  # this was adopted to ollama style completion.


messages = []
agent = my_agent
while True:
    user_input = input("> ")
    messages.append({"role": "user", "content": user_input})

    response = client.run(agent=agent, messages=messages)
    messages = response.messages
    agent = response.agent
    pretty_print_messages(messages)
