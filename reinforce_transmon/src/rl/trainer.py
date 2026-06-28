"""Custom DDPG Training Loop."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from pathlib import Path

from reinforce_transmon.src.rl.agent import DDPGAgent
from reinforce_transmon.src.utils import create_gif, plot_learning_curve

# Set directories for saving plots after training:
PROJECT_ROOT = Path(__file__).resolve().parents[3]
FRAMES_DIR = PROJECT_ROOT / "assets" / "render"
GIF_DIR = PROJECT_ROOT / "assets"
PLOT_DIR = PROJECT_ROOT / "assets" / "plots"


def _validate_inputs(num_episodes: int, max_steps: int, buffer_size: int) -> None:
    """
    Validate training input parameters.

    Args:
        num_episodes: Number of episodes.
        max_steps: Max number of transitions per episode.
        buffer_size: Maximum number of transitions the replay buffer can store.

    Raises:
        ValueError: If num_episodes or max_steps is not greater than zero.
    """

    if num_episodes <= 0:
        raise ValueError("num_episodes must be greater than zero.")
    if max_steps > buffer_size:
        raise ValueError("max_steps must be less than or equal to buffer_size.")


def _run_episode(env, agent, inference: bool, seed: int | None = None):
    """
    Run a single episode.

    Args:
        env: Gym-compatible environment.
        agent: DDPG agent instance.
        inference: Inference mode (disables learning updates).
        seed: Optional random seed for reproducibility.

    Returns:
        Total episode reward and info dict.
    """
    obs, info = env.reset(seed=seed)
    terminated, truncated = False, False
    episode_reward = 0.0

    while not (terminated or truncated):
        action = agent.get_action(obs, inference=inference)
        next_obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated
        if not inference:
            agent.store_transition(obs, action, reward, next_obs, done)
            # Learn after replay buffer warm up (at least batch_size transitions):
            agent.learn()

        obs = next_obs
        episode_reward += float(reward)

    return episode_reward, info


def _log_episode_result(episode_reward: float, info: dict) -> None:
    """
    Log episode results to the console.

    Args:
        episode_reward: Total reward for the episode.
        info: Optional dictionary containing environment-specific info.

    Returns:
        None
    """
    if info:
        print(
            f"Reward: {episode_reward:.2f}, "
            f"Ej: {info.get('ej'):.2f}, "
            f"Ec: {info.get('ec'):.2f}, "
            f"Ng: {info.get('ng'):.2f}, "
            f"Ej/Ec: {info.get('ej_ec'):.2f}, "
            f"Anharmonicity: {info.get('anharmonicity'):.2f}, "
            f"Dispersion: {info.get('dispersion'):.2f}, "
            f"T2: {info.get('t2'):.2f}, "
        )
    else:
        print(f"Reward: {episode_reward}")


def train(
    env: gym.Env,
    agent: DDPGAgent,
    num_episodes: int = 10,
    max_steps: int = 70,
    render: bool = False,
    inference: bool = False,
    seed: int | None = None,
) -> list[float]:
    """
    Train a DDPG agent in a custom environment.

    This follows the standard Gym-style pattern, where the training loop
    lives outside the agent class.

    Args:
        env: Gym-compatible environment.
        agent: DDPG agent instance.
        num_episodes: Number of episodes.
        max_steps: Max number of transitions per episode.
        config_path: Optional path to a config file containing TRAIN defaults.
        save_best_model: Save weights when the moving-average score improves.
        render: Call env.render() at episode end.
        inference: Evaluation mode (disables learning updates).
    """

    _validate_inputs(num_episodes, max_steps, agent.buffer_size)

    if hasattr(env, "max_steps"):
        env.max_steps = max_steps

    best_score = env.reward_range[0]  # float("-inf")
    score_history: list[float] = []

    mode = "inference" if inference else "training"
    print(
        f"\nStarting {mode} for {num_episodes} episodes and {max_steps} steps each..."
    )

    for ep in range(num_episodes):
        print(f"\nEpisode {ep+1:03d}.")

        episode_seed = None if seed is None else seed + ep
        episode_reward, info = _run_episode(env, agent, inference, seed=episode_seed)

        # At the end of the episode:

        # Update score history and compute moving average:
        score_history.append(episode_reward)
        avg_score = float(np.mean(score_history[-100:]))

        # Save model if performance improves:
        if not inference and avg_score > best_score:
            best_score = avg_score
            agent.save_model()

        # Log results to console:
        _log_episode_result(episode_reward, info)

        # Render environment if enabled:
        if render:
            env.render()

    # After training, plot learning curve and create GIFs if rendering was enabled:
    if render:
        plot_learning_curve(score_history, PLOT_DIR)
        create_gif(
            frames_dir=FRAMES_DIR / "anharmonicity",
            output_dir=GIF_DIR,
            gif_name="anharmonicity.gif",
        )
        """
        create_gif(
            frames_dir=FRAMES_DIR / "coherence",
            output_dir=GIF_DIR,
            gif_name="coherence.gif",
        )
        """

    return score_history
