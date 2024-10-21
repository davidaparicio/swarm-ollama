from swarm_ollama.core import Swarm, Agent


client = Swarm()

agent = Agent(
    model="llama3.2:3b",
    name="Agent",
    instructions="You are a helpful agent.",
)

messages = [{"role": "user", "content": "Hi!"}]
response = client.run(agent=agent, messages=messages)

print(response.messages[-1]["content"])
