import configparser
from pathlib import Path
import random

import numpy as np
import tensorflow as tf

from reinforce_transmon.src.envs import TransmonQubitEnv
from reinforce_transmon.src.rl import DDPGAgent, train


def load_config(config_path: str | Path) -> configparser.ConfigParser:
    """Load configuration from a file.

    Args:
        config_path: Path to the configuration file.

    Returns:
        config: Configuration object.

    """

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(str(config_path))
    return config


def setup_env_and_agent(config: configparser.ConfigParser, max_steps: int):
    """Initialize the environment and agent.

    Args:
        config: Configuration object.
        max_steps: Maximum number of steps per episode.

    Returns:
        env: Environment instance.
        agent: Agent instance.
    """

    # Environment hyperparameters:
    min_ej = config.getfloat("ENV", "min_ej")
    max_ej = config.getfloat("ENV", "max_ej")
    min_ec = config.getfloat("ENV", "min_ec")
    max_ec = config.getfloat("ENV", "max_ec")
    min_ng = config.getfloat("ENV", "min_ng")
    max_ng = config.getfloat("ENV", "max_ng")

    # Instantiate the environment:
    env = TransmonQubitEnv(min_ej, max_ej, min_ec, max_ec, min_ng, max_ng, max_steps)

    # Neural Network hyperparameters:
    layer_act_dims = eval(config.get("NEURAL_NET", "layer_act_dims"))
    layer_crit_dims = eval(config.get("NEURAL_NET", "layer_crit_dims"))

    # Agent's hyperparameters:
    lr_actor = config.getfloat("AGENT", "lr_actor")
    lr_critic = config.getfloat("AGENT", "lr_critic")
    rho = config.getfloat("AGENT", "rho")
    gamma = config.getfloat("AGENT", "gamma")
    noise = config.getfloat("AGENT", "noise")
    buffer_size = config.getint("AGENT", "buffer_size")
    batch_size = config.getint("AGENT", "batch_size")

    # Instantiate the agent:
    agent = DDPGAgent(
        env.observation_space,
        env.action_space,
        layer_act_dims,
        layer_crit_dims,
        lr_actor,
        lr_critic,
        rho,
        gamma,
        noise,
        buffer_size,
        batch_size,
    )

    return env, agent


def set_global_seed(seed: int) -> None:
    """Set process-wide seeds for reproducible training and evaluation."""

    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


def main(render=False, inference=False) -> None:
    """Main function to train the agent.

    Args:
        render: Whether to render the environment during training.
        inference: Whether to run in inference mode (no learning updates).

    Returns:
        None.
    """

    # Get the path to the config.cfg file:
    project_root = Path(__file__).resolve().parents[2]
    config_file_path = project_root / "configs" / "config.cfg"

    # Load the configuration file:
    config = load_config(config_file_path)

    # Training hyperparameters:
    num_episodes = config.getint("TRAIN", "num_episodes")
    max_steps = config.getint("TRAIN", "max_steps")
    seed = config.getint("TRAIN", "seed", fallback=32)

    set_global_seed(seed)

    # Instantiate the environment and agent:
    env, agent = setup_env_and_agent(config, max_steps)

    if inference:
        agent.load_full_models()

    train(
        env=env,
        agent=agent,
        num_episodes=num_episodes,
        max_steps=max_steps,
        render=render,
        inference=inference,
        seed=seed,
    )


if __name__ == "__main__":
    main()
