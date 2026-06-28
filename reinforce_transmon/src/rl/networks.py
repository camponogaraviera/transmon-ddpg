"""Neural Networks Using the Model Subclassing API."""

from typing import Any, Tuple
from tensorflow.keras.layers import Dense
import tensorflow as tf
from tensorflow.types.experimental import TensorLike


@tf.keras.utils.register_keras_serializable(package="reinforce_transmon")
class ActorNetwork(tf.keras.Model):
    """
    Actor network.

    Attributes:
        layer_dims: List of integers specifying the number of units
            in each hidden layer.
        num_actuators: Dimensionality of the action space (i.e., number of continuous
            action outputs produced by the policy).
    """

    def __init__(
        self,
        layer_dims: list[int],
        num_actuators: int,
        name: str = "actor",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the actor network.

        Args:
            layer_dims: Sizes of hidden layers.
            num_actuators: Shape of the action space.
            name: Base name for the model.
        """

        super().__init__(name=name, **kwargs)
        self.layer_dims = list(layer_dims)
        self.num_actuators = num_actuators

        # Fully-connected 1:
        self.fc1 = Dense(
            self.layer_dims[0],
            activation="relu",
            kernel_initializer="glorot_normal",
        )

        # Fully-connected 2:
        self.fc2 = Dense(
            self.layer_dims[1],
            activation="relu",
            kernel_initializer="glorot_normal",
        )

        # DDPG uses a deterministic Policy:
        self.act_output = Dense(self.num_actuators, activation="tanh")

    def call(
        self,
        inputs: tf.Tensor,
        training: bool | None = False,
        mask: TensorLike | None = None,
    ) -> tf.Tensor:
        """
        Forward pass for action prediction.

        Args:
            inputs: Batch of state tensors.
            training: Whether the model is in training mode (for batch normalization).
            mask: Optional mask for variable-length sequences (not used here).

        Returns:
            Action tensor.
        """
        x = self.fc1(inputs)
        x = self.fc2(x)
        actions = self.act_output(x)
        return actions

    def get_config(self) -> dict[str, Any]:
        """
        Return the config required to serialize this model.

        Returns:
            config: Model configuration dictionary including custom fields.
        """

        config = super().get_config()
        config.update(
            {
                "layer_dims": self.layer_dims,
                "num_actuators": self.num_actuators,
            }
        )
        return config


@tf.keras.utils.register_keras_serializable(package="reinforce_transmon")
class CriticNetwork(tf.keras.Model):
    """
    Critic network.

    Attributes:
        layer_dims: List of integers specifying the number of units
            in each hidden layer.
    """

    def __init__(
        self,
        layer_dims: list[int],
        name: str = "critic",
        **kwargs: Any,
    ) -> None:
        """
        Initialize the critic network.

        Args:
            layer_dims: Sizes of hidden layers.
            name: Base name for the model.
        """

        super().__init__(name=name, **kwargs)
        self.layer_dims = list(layer_dims)

        # Fully-connected 1:
        self.fc1 = Dense(
            self.layer_dims[0],
            activation="relu",
            kernel_initializer="glorot_normal",
        )

        # Fully-connected 2:
        self.fc2 = Dense(
            self.layer_dims[1],
            activation="relu",
            kernel_initializer="glorot_normal",
        )
        self.q_output = Dense(1, activation="linear")

    def call(
        self,
        inputs: Tuple[tf.Tensor, tf.Tensor],
        training: bool | None = False,
        mask: TensorLike | None = None,
    ) -> tf.Tensor:
        """
        Forward pass for state-action value estimation.

        Args:
            inputs: Tuple of (state, action) tensors.
            training: Whether the model is in training mode (for batch normalization).
            mask: Optional mask for variable-length sequences (not used here).

        Returns:
            Estimated Q-values.
        """
        x = tf.concat(inputs, axis=1)
        x = self.fc1(x)
        x = self.fc2(x)
        q_value = self.q_output(x)
        return q_value

    def get_config(self) -> dict[str, Any]:
        """
        Return the config required to serialize this model.

        Returns:
            config: Model configuration dictionary including custom fields.
        """

        config = super().get_config()
        config.update({"layer_dims": self.layer_dims})
        return config
