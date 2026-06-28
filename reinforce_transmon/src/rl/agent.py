"""DDPG Agent."""

from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

from reinforce_transmon.src.rl.networks import ActorNetwork, CriticNetwork
from reinforce_transmon.src.rl.replay_buffer import ReplayBuffer
from reinforce_transmon.src.utils import min_max_norm_env

# Type:
CheckpointMode = Literal["full", "weights", "both"]


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent implementation.

    This agent implements an actor–critic architecture with target networks
    and experience replay for continuous control environments.

    Attributes:
        obs_dim: Shape of the observation space.
        state_low: Lower bounds of the observation space.
        state_high: Upper bounds of the observation space.

        action_space: Gym action space definition.
        num_actions: Dimensionality of the action space.

        layer_act_dims: Hidden layer sizes of the actor network.
        layer_crit_dims: Hidden layer sizes of the critic network.

        lr_actor: Learning rate for the actor optimizer.
        lr_critic: Learning rate for the critic optimizer.

        rho: Soft update coefficient (Polyak averaging factor) for target networks.
        gamma: Discount factor for future rewards.

        noise: Standard deviation of Gaussian exploration noise.

        buffer_size: Maximum capacity of the replay buffer.
        batch_size: Number of samples per training update.

        chkpt_dir: Directory path for saving model checkpoints.

        replay_buffer: Experience replay buffer storing transitions.

        actor_net: Main actor network.
        critic_net: Main critic network.

        target_actor_net: Target actor network used for stable updates.
        target_critic_net: Target critic network used for stable Q-value estimation.
    """

    def __init__(
        self,
        obs_space: gym.spaces.Box | None = None,
        action_space: gym.spaces.Box | None = None,
        layer_act_dims: list[int] | None = None,
        layer_crit_dims: list[int] | None = None,
        lr_actor: float = 1e-4,
        lr_critic: float = 3e-4,
        rho: float = 0.005,
        gamma: float = 0.98,
        noise: float = 0.2,
        buffer_size: int = 100000,
        batch_size: int = 128,
        chkpt_path: str | Path | None = None,
    ) -> None:
        """
        Initialize the DDPG agent and its networks.

        Args:
            obs_space: Observation space of the environment.
            action_space: Action space of the environment.
            layer_act_dims: Hidden layer sizes for the actor.
            layer_crit_dims: Hidden layer sizes for the critic.
            lr_actor: Actor learning rate.
            lr_critic: Critic learning rate.
            rho: Polyak average factor for target networks update. Range: [0, 1].
            gamma: Discount factor. Range: [0, 1].
            noise: Standard deviation for Gaussian exploration noise.
            buffer_size: Maximum number of transitions in replay buffer.
            batch_size: Number of transitions sampled from the replay buffer.
            chkpt_path: Directory to save model checkpoints.
                Defaults to <project_root>/artifacts/model_and_weights.
        """

        # Observation space:
        self.obs_dim = obs_space.shape
        self.state_low = obs_space.low
        self.state_high = obs_space.high

        # Action space:
        if action_space is None:
            action_space = gym.spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        self.action_space = action_space
        self.num_actions = action_space.shape[0]

        # Hyperparameters:
        self.layer_act_dims = layer_act_dims or [256, 256]
        self.layer_crit_dims = layer_crit_dims or [256, 256]
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.rho = rho
        self.gamma = gamma
        self.noise = noise
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        # Model path:
        if chkpt_path is None:
            project_root = Path(__file__).resolve().parents[3]
            chkpt_path = project_root / "artifacts" / "model_and_weights"
        self.chkpt_dir = Path(chkpt_path).expanduser().resolve()

        # Instantiate the replay buffer class:
        self.replay_buffer = ReplayBuffer(buffer_size, self.obs_dim, self.num_actions)

        # Instantiate the models:
        self.actor_net = ActorNetwork(
            layer_act_dims, num_actuators=self.num_actions, name="actor"
        )
        self.critic_net = CriticNetwork(layer_crit_dims, name="critic")
        self.target_actor_net = ActorNetwork(
            layer_act_dims, num_actuators=self.num_actions, name="target_actor"
        )
        self.target_critic_net = CriticNetwork(layer_crit_dims, name="target_critic")

        # Build the models:
        self._build_networks()

        # Compile trainable networks:
        self.actor_net.compile(optimizer=Adam(learning_rate=lr_actor))
        self.critic_net.compile(optimizer=Adam(learning_rate=lr_critic))

        # Creating a copy of weights to target weights:
        self.target_actor_net.set_weights(self.actor_net.get_weights())
        self.target_critic_net.set_weights(self.critic_net.get_weights())

    def _build_networks(self) -> None:
        """
        Build networks for the load_model_weights() method.

        Returns:
            None.
        """

        dummy_state = tf.zeros((1, *self.obs_dim), dtype=tf.float32)
        dummy_action = tf.zeros((1, self.num_actions), dtype=tf.float32)

        self.actor_net(dummy_state, training=False)
        self.target_actor_net(dummy_state, training=False)
        self.critic_net((dummy_state, dummy_action), training=False)
        self.target_critic_net((dummy_state, dummy_action), training=False)

    def _scale_action(self, action: tf.Tensor) -> tf.Tensor:
        """
        Linearly scale actions from [-1, 1] to the environment's action space.

        Args:
            action: Tensor with values assumed to be in [-1, 1].

        Returns:
            The rescaled action in [action_space.low, action_space.high].
        """
        low = self.action_space.low
        high = self.action_space.high
        return low + (action + 1.0) * 0.5 * (high - low)

    def _update_target_networks(self) -> None:
        """
        Soft-update target actor and critic networks using Polyak averaging.

        Returns:
            None.
        """

        # Update target actor:
        for target_var, var in zip(
            self.target_actor_net.trainable_variables,
            self.actor_net.trainable_variables,
        ):
            target_var.assign(self.rho * var + (1 - self.rho) * target_var)

        # Update target critic:
        for target_var, var in zip(
            self.target_critic_net.trainable_variables,
            self.critic_net.trainable_variables,
        ):
            target_var.assign(self.rho * var + (1 - self.rho) * target_var)

    def get_action(self, state: np.ndarray, inference: bool = False) -> tf.Tensor:
        """
        Get an action from the current policy network.
        The policy is deterministic, i.e., for a fixed set of network parameters
        and a fixed state, the same action will be produced (if noise is disabled).

        Args:
            state: Current environment state.
            inference: Whether to disable noise for exploration.

        Returns:
            The selected action tensor.
        """

        # Normalize input state for neural network training:
        state = min_max_norm_env(state, self.state_low, self.state_high)

        # Convert normalized state to TF tensor:
        state = tf.convert_to_tensor([state], dtype=tf.float32)

        # Use the actor in inference mode for environment interaction.
        actions = self.actor_net(state, training=False)

        # Scale action to environment bounds:
        actions = self._scale_action(actions)

        # Adding a Gaussian noise to ensure exploration:
        if not inference:
            noise = (
                tf.random.normal(shape=tf.shape(actions))
                * self.noise
                * (self.action_space.high - self.action_space.low)
                / 2
            )
            actions += noise

        # Clip the actions to be bounded by the action space:
        actions = tf.clip_by_value(
            actions, self.action_space.low, self.action_space.high
        )

        return actions[0].numpy()

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray | tf.Tensor,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """
        Add a transition to the replay buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Episode termination flag.

        Returns:
            None.
        """

        self.replay_buffer.push(state, action, reward, next_state, done)

    def learn(self) -> None:
        """
        Train actor and critic networks using a batch from the replay buffer.
        A `training=Boolean` argument is passed for when batch normalization is used.

        Returns:
            None.
        """

        # If there are not enough experiences in the replay buffer, exit:
        if self.replay_buffer.num_experiences < self.batch_size:
            return

        # Randomly sample a batch of experiences from the replay buffer:
        state, action, reward, next_state, done = self.replay_buffer.sample(
            self.batch_size
        )

        # Convert to TF tensors:
        state = min_max_norm_env(state, self.state_low, self.state_high)
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        action = tf.convert_to_tensor(action, dtype=tf.float32)
        reward = tf.convert_to_tensor(reward, dtype=tf.float32)
        next_state = min_max_norm_env(next_state, self.state_low, self.state_high)
        next_state = tf.convert_to_tensor(next_state, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)

        # Update critic network:
        with tf.GradientTape() as tape:
            target_action = self._scale_action(
                self.target_actor_net(next_state, training=False)
            )

            critic_value = tf.squeeze(
                self.critic_net((state, action), training=True),
                axis=1,
            )

            target_critic_value = tf.squeeze(
                self.target_critic_net((next_state, target_action), training=False),
                axis=1,
            )

            target = reward + self.gamma * tf.stop_gradient(target_critic_value) * (
                1 - done
            )

            critic_loss = tf.keras.losses.MeanSquaredError()(target, critic_value)
        critic_weights = self.critic_net.trainable_variables
        critic_grads = tape.gradient(critic_loss, critic_weights)
        self.critic_net.optimizer.apply_gradients(zip(critic_grads, critic_weights))

        # Update actor network:
        with tf.GradientTape() as tape:
            actor_actions = self._scale_action(self.actor_net(state, training=True))
            actor_loss = -tf.reduce_mean(
                self.critic_net((state, actor_actions), training=True), axis=0
            )
        actor_weights = self.actor_net.trainable_variables
        actor_grads = tape.gradient(actor_loss, actor_weights)
        self.actor_net.optimizer.apply_gradients(zip(actor_grads, actor_weights))

        # Soft target network updates:
        self._update_target_networks()

    def save_model(self, mode: CheckpointMode = "both") -> None:
        """
        Save the full model, including architecture and weights.

        Args:
            mode: "full", "weights", or "both".

        Returns:
            None.
        """

        if mode not in {"full", "weights", "both"}:
            raise ValueError(f"Invalid mode: {mode}")

        # Creating a folder to save models:
        self.chkpt_dir.mkdir(parents=True, exist_ok=True)

        # Models to save:
        models_to_save = [
            self.actor_net,
            self.critic_net,
            self.target_actor_net,
            self.target_critic_net,
        ]

        # Saving models:
        for model in models_to_save:
            if model is None:
                raise ValueError("Encountered a None model during saving.")

            # Save model architecture + weights + optimizer state:
            if mode in ("full", "both"):
                model_path = self.chkpt_dir / f"{model.name}.keras"
                model.save(str(model_path), overwrite=True)

            # Save only model weights:
            if mode in ("weights", "both"):
                model_weights_path = self.chkpt_dir / f"{model.name}.weights.h5"
                model.save_weights(str(model_weights_path), overwrite=True)

        print(f"[INFO] Models saved successfully in '{self.chkpt_dir}' (mode={mode}).")

    def load_model_weights(self) -> None:
        """
        Load model weights only (no optimizer state).

        Returns:
            None.
        """

        # Build networks before loading weights is required in Keras:
        self._build_networks()

        # Models to load:
        models_to_load = [
            self.actor_net,
            self.critic_net,
            self.target_actor_net,
            self.target_critic_net,
        ]

        for model in models_to_load:
            weights_path = self.chkpt_dir / f"{model.name}.weights.h5"

            if not weights_path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {weights_path}")

            try:
                model.load_weights(str(weights_path))
            except Exception as e:
                raise RuntimeError(f"Failed loading weights for '{model.name}': {e}")

        print(f"[INFO] Weights loaded successfully from '{self.chkpt_dir}'.")

    def load_full_models(self, compile_models: bool = False) -> None:
        """
        Load full models (architecture + weights + optimizer state).

        Args:
            compile_models: Whether to restore optimizer state.

        Returns:
            None.
        """

        # Models to load:
        model_attrs = [
            "actor_net",
            "critic_net",
            "target_actor_net",
            "target_critic_net",
        ]

        for attr in model_attrs:
            model = getattr(self, attr)
            path = self.chkpt_dir / f"{model.name}.keras"

            if not path.exists():
                raise FileNotFoundError(f"Missing checkpoint: {path}")

            try:
                loaded_model = tf.keras.models.load_model(
                    str(path), compile=compile_models
                )
            except Exception as e:
                raise RuntimeError(f"Failed loading full model '{model.name}': {e}")

            setattr(self, attr, loaded_model)

        print(f"[INFO] Full models loaded successfully from '{self.chkpt_dir}'.")
