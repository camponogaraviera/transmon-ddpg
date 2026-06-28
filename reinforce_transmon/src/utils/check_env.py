from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register, registry
import gymnasium as gym

ENV_ID = "TransmonQubit-v0"

if ENV_ID not in registry:
    register(
        id=ENV_ID,
        entry_point="reinforce_transmon.src.envs:TransmonQubitEnv",
    )

# Initialize the environment:
env = gym.make(ENV_ID)


def _check_env() -> None:
    """Checks whether the environment is GYM-compliant.

    Returns:
        None.
    """

    try:
        check_env(env.unwrapped)
    except Exception as e:
        print(e)
    else:
        print("\nCongratz! The environment is gym-compliant!")


if __name__ == "__main__":
    _check_env()
