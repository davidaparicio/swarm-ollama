from swarm_ollama import Swarm, Agent

client = Swarm()
## IF NEEDED, you change surcharge the base_url with the OLLAMA_URL
# client = Swarm(base_url="http://<your-ip>:11434")


def transfer_to_agent_b():
    return agent_b


agent_a = Agent(
    name="Agent A",
    model="llama3.2:3b",
    instructions="You are a helpful agent.",
    functions=[transfer_to_agent_b],
)

agent_b = Agent(
    name="Agent B",
    model="tinyllama:1.1b",
    instructions="Only speak in Haikus.",
)

response = client.run(
    agent=agent_a,
    messages=[{"role": "user", "content": "I want to talk to agent B."}],
)

print(response.messages[-1]["content"])
