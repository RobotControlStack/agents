AGENTS = {}


def register_agent(name: str, agent_class: type["Agent"]) -> None:
    """
    Register an agent class with a given name.

    Args:
        name (str): The name of the agent.
        agent_class (type[Agent]): The agent class to register.
    """
    AGENTS[name] = agent_class


from vlagents.policies.interface import Agent

__version__ = "0.2.0"
__all__ = ["__doc__", "__version__", "AGENTS"]
