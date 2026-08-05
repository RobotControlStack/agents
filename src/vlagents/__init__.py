AGENTS = {}
ENVS = {}


def register_agent(name: str, agent_class: type["Agent"]) -> None:
    """
    Register an agent class with a given name.

    Args:
        name (str): The name of the agent.
        agent_class (type[Agent]): The agent class to register.
    """
    AGENTS[name] = agent_class


def register_env(name: str, env_class: type["EvalEnv"]) -> None:
    """
    Register an environment class with a given name.

    Args:
        name (str): The name of the environment.
        env_class (type[EvalEnv]): The environment class to register.
    """
    ENVS[name] = env_class


from vlagents.envs import duobench, libero, maniskill  # noqa: E402, F401
from vlagents.envs.interface import EvalEnv
from vlagents.policies.interface import Agent

__version__ = "0.2.0"
__all__ = ["__doc__", "__version__", "AGENTS", "ENVS"]
