"""Replay Buffer in NumPy Since Storage is CPU-heavy."""

import numpy as np
import tensorflow as tf


class ReplayBuffer:
    """
    A fixed-size circular replay buffer for storing and sampling
    experience transitions.

    The buffer stores tuples of (state, action, reward, next_state, done)
    and overwrites old experiences once capacity is reached.

    Attributes:
        buffer_size: Maximum number of transitions the buffer can store.
        buffer: Dictionary containing preallocated NumPy arrays for:
            - "states": Stored states/observations.
            - "actions": Stored actions.
            - "rewards": Stored rewards.
            - "next_states": Stored next states/observations.
            - "dones": Stored episode termination flags.
        position: Current insertion index in the circular buffer.
        num_experiences: Current number of stored transitions.
        rng: NumPy random number generator used for sampling batches.
    """

    def __init__(
        self,
        buffer_size: int,
        state_dim: tuple[int, ...],
        num_actions: int,
        seed: int = 42,
    ) -> None:
        """
        Initialize a fixed-size replay buffer.

        Args:
            buffer_size: Maximum number of transitions to store.
            state_dim: Shape of the state/observation.
            action_dim: Shape of an action.
        """
        self.buffer_size = buffer_size

        self.buffer = {
            "states": np.empty((buffer_size, *state_dim), dtype=np.float32),
            "actions": np.empty((buffer_size, num_actions), dtype=np.float32),
            "rewards": np.empty(buffer_size, dtype=np.float32),
            "next_states": np.empty((buffer_size, *state_dim), dtype=np.float32),
            "dones": np.empty(buffer_size, dtype=np.float32),
        }
        # Counter for the position/index of each transition:
        self.position = 0
        # Counter for the number of transitions/experiences stored in the buffer:
        self.num_experiences = 0
        # Random generator:
        self.rng = np.random.default_rng(seed)

    def push(
        self,
        state: np.ndarray,
        action: np.ndarray | tf.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """
        Store a transition in the buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Episode termination flag.

        Returns:
            None.
        """

        if isinstance(action, tf.Tensor):
            action = action.numpy()

        self.buffer["states"][self.position] = state
        self.buffer["actions"][self.position] = action
        self.buffer["rewards"][self.position] = float(reward)
        self.buffer["next_states"][self.position] = next_state
        self.buffer["dones"][self.position] = float(done)

        # Ensuring a circular buffer:
        self.position = (self.position + 1) % self.buffer_size
        # Preventing self.num_experiences from growing beyond self.buffer_size:
        self.num_experiences = min(self.num_experiences + 1, self.buffer_size)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample a batch of transitions.

        Args:
            batch_size: Number of transitions to sample.

        Returns:
            Tuple of sampled (states, actions, rewards, next_states, dones).

        Raises:
            ValueError: If batch_size exceeds stored experiences.
        """
        if batch_size > self.num_experiences:
            raise ValueError(
                (
                    f"Not enough samples in buffer "
                    f"Requested batch_size {batch_size} exceeds the number "
                    f"of experiences {self.num_experiences} available in the buffer."
                )
            )

        # Sampling without replacement:
        indices = self.rng.choice(self.num_experiences, size=batch_size, replace=False)

        return (
            self.buffer["states"][indices],
            self.buffer["actions"][indices],
            self.buffer["rewards"][indices],
            self.buffer["next_states"][indices],
            self.buffer["dones"][indices],
        )
