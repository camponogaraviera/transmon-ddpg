import gymnasium as gym  # Compatibility with Gym-style environments.
import tensorflow as tf  # Tensor operations.
import scqubits as scq  # Transmon qubit, Anharmonicity, T1 and T2 times.
import numpy as np  # Tensor operations.
from pathlib import Path  # Directory access.
from reinforce_transmon.src.utils import (
    # plot_transmon_coherence,
    plot_transmon_energy_levels,
)

scq.settings.T1_DEFAULT_WARNING = False


class TransmonQubitEnv(gym.Env):
    """
    Reinforcement learning environment for a transmon qubit.

    The environment models a superconducting transmon qubit with state variables
    corresponding to physical parameters (EJ, EC, ng). The agent interacts by
    applying bounded continuous actions that modify these parameters to
    optimize a physics-informed reward based on charge dispersion, anharmonicity,
    and coherence time (T2).

    Attributes:
        action_space (gym.spaces.Box):
            Continuous 3D action space in [-1, 1] used to perturb (EJ, EC, ng)
            indirectly via scaled updates to the observation.

        observation_space (gym.spaces.Box):
            Continuous state space representing physical qubit parameters:
            [EJ, EC, ng], bounded by user-defined limits.

        reward_range (tuple[float, float]):
            Theoretical range of rewards (unbounded in practice, but defined as
            (-inf, inf).

        max_steps (int):
            Maximum number of steps per episode before truncation.

        episode_id (int):
            Counter tracking the current episode number.

        temp (float):
            Temperature (in Kelvin) used in coherence (T2) calculations for
            noise modeling.

        step_count (int):
            Counter tracking the current step within an episode.

        observation (np.ndarray):
            Current state of the environment: [EJ, EC, ng].

        ej (float):
            Cached Josephson energy from the latest observation.

        ec (float):
            Cached charging energy from the latest observation.

        ng (float):
            Cached offset charge from the latest observation.

        tmon1 (scq.Transmon):
            The primary transmon object used for energy level and coherence
            calculations.

        t2 (float):
            Effective coherence time (T2) computed for the current state.

        anharmonicity (float):
            Difference in energy level spacing.

        dispersion (float):
            Charge dispersion metric, computed as the difference in transition
            frequencies between two offset charges.
    """

    def __init__(
        self,
        min_ej: float = 0.05,
        max_ej: float = 12.0,
        min_ec: float = 0.15,
        max_ec: float = 0.3,
        min_ng: float = -1.0,
        max_ng: float = 1.0,
        max_steps: int = 100,
    ) -> None:
        """
        Initialize a transmon qubit environment.

        Args:
            min_ej: Minimum Josephson energy.
            max_ej: Maximum Josephson energy.
            min_ec: Minimum charging energy.
            max_ec: Maximum charging energy.
            min_ng: Minimum offset charge.
            max_ng: Maximum offset charge.
            max_steps: Maximum steps per episode.
        """
        super().__init__()

        self.action_space = gym.spaces.Box(
            low=np.array([-1, -1, -1], dtype=np.float32),
            high=np.array([1, 1, 1], dtype=np.float32),
            dtype=np.float32,
        )

        self.observation_space = gym.spaces.Box(
            low=np.array([min_ej, min_ec, min_ng], dtype=np.float32),
            high=np.array([max_ej, max_ec, max_ng], dtype=np.float32),
            dtype=np.float32,
        )

        self.reward_range = (-float("inf"), float("inf"))  # [min, max]

        self.max_steps = max_steps
        self.episode_id = 0

        self.temp = 0.050

        self.step_count = 0
        self.observation = None

        self.ej = None
        self.ec = None
        self.ng = None

        self.tmon1 = None
        self.t2 = None
        self.anharmonicity = None
        self.dispersion = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, dict[str, float]]:
        """
        Reset the environment to the initial state.

        Args:
            seed: Random seed for reproducibility.

        Returns:
            A tuple of (observation, info).
        """

        print("Resetting the environment...")

        # Seed for reproducibility when using RNG:
        super().reset(seed=seed)

        # Increment episode counter:
        self.episode_id += 1

        # Reset step counter:
        self.step_count = 0

        # Initialize the env. to a random state:
        # self.observation = self.observation_space.sample() # Uses RNG.

        # Initialize the env. to a fixed state:
        # self.observation = self.observation_space.low.copy()

        # Initialize the env. to a custom state:
        self.observation = np.array([0.05, 0.3, 0.0], dtype=np.float32)

        # Additional info:
        info: dict[str, float] = options or {}

        return self.observation, info

    def _reward_function(
        self, state: np.ndarray, w_disp=0.8, w_anh=0.9, w_t2=0.8
    ) -> float:
        """
        Compute the reward value for a given state.

        Physics-informed reward:
        - Minimize charge dispersion.
        - Maximize anharmonicity.
        - Maximize coherence time T2.

        Args:
            state: Current environment state.
            w_disp: Weight for charge dispersion.
            w_anh: Weight for anharmonicity.
            w_t2: Weight for T2 coherence.

        Returns:
            Reward value.
        """

        # Unpack the state:
        ej, ec, ng = state

        # Add a shift to compute the difference:
        ng_shift = 0.5
        low = self.observation_space.low[2]
        high = self.observation_space.high[2]
        ng1 = ng
        ng2 = ng + ng_shift
        if ng2 > high:
            ng2 = high - (ng2 - high)
        elif ng2 < low:
            ng2 = low + (low - ng2)

        # Build transmon qubits:
        tmon1 = scq.Transmon(EJ=ej, EC=ec, ng=ng1, ncut=31)
        tmon2 = scq.Transmon(EJ=ej, EC=ec, ng=ng2, ncut=31)

        # Eigenvalues (in GHz):
        e1 = tmon1.eigenvals(evals_count=3)  # First three levels (m = 0,1,2).
        e2 = tmon2.eigenvals(evals_count=3)

        # Transition frequencies:
        w01_1 = e1[1] - e1[0]
        w01_2 = e2[1] - e2[0]

        # Anharmonicity (E12 - E01):
        anh = (e1[2] - e1[1]) - (e1[1] - e1[0])

        # Charge dispersion:
        delta_w01 = abs(w01_1 - w01_2)

        # T2 coherence:
        t2 = tmon1.t2_effective(common_noise_options={"i": 1, "j": 0, "T": self.temp})

        # =========================================================
        # Normalized Components
        # =========================================================

        # Anharmonicity (GHz) normalization:
        anh_scale = 0.3
        anh_norm = np.tanh(abs(anh) / anh_scale)

        # T2 (ns) normalization:
        t2_ref = 100
        t2_norm = np.tanh(t2 / t2_ref)

        # =========================================================
        # Combine Reward
        # =========================================================

        reward = -w_disp * delta_w01 + w_anh * anh_norm + w_t2 * t2_norm

        # Normalize reward:
        reward /= w_disp + w_anh + w_t2

        # =========================================================
        # Logging
        # =========================================================

        self.t2 = float(t2)
        self.tmon1 = tmon1
        self.anharmonicity = float(anh)
        self.dispersion = float(delta_w01)

        return float(reward)

    def step(
        self, action: np.ndarray | tf.Tensor
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, float]]:
        """
        Advance the environment by one step.

        Args:
            action: Action taken by the agent.

        Returns:
            A tuple of (new_obs, reward, terminated, truncated, info).
        """

        # Update counter:
        self.step_count += 1

        # Observation range:
        obs_low = self.observation_space.low
        obs_high = self.observation_space.high
        obs_range = obs_high - obs_low

        # X% of range per step:
        step_fraction = 0.1

        # Scale action to the range of the observation space:
        action = np.asarray(action, dtype=np.float32)
        scaled_action = action * obs_range * step_fraction

        # Clip the new observation to the range of the observation space:
        new_obs = np.clip(
            self.observation + scaled_action,
            obs_low,
            obs_high,
        )

        # Update the observation:
        self.observation = new_obs

        # Compute the reward:
        reward = self._reward_function(new_obs)

        # Update info:
        self.ej, self.ec, self.ng = new_obs
        info = {
            "anharmonicity": self.anharmonicity,
            "dispersion": self.dispersion,
            "ej": self.ej,
            "ec": self.ec,
            "ng": self.ng,
            "t2": self.t2,
            "ej_ec": self.ej / self.ec,
        }

        # Flags:
        terminated, truncated = False, False

        # Update flag at max episode length (time steps) reached:
        if self.step_count == self.max_steps:
            truncated = True
            print(f"Episode truncated after {self.step_count} steps.")

        return new_obs, reward, terminated, truncated, info

    def render(self) -> None:
        """
        Plot and save energy levels and coherence to disk.

        Returns:
            None.
        """

        print("Rendering environment...")

        project_root = Path(__file__).resolve().parents[3]
        output_dir = project_root / "assets" / "render"

        plot_transmon_energy_levels(
            transmon=self.tmon1,
            output_dir=output_dir / "anharmonicity",
            episode_id=self.episode_id,
            step_count=self.step_count,
            ej=self.ej,
            ec=self.ec,
        )

        """
        plot_transmon_coherence(
            transmon=self.tmon1,
            output_dir=output_dir / "coherence",
            episode_id=self.episode_id,
            step_count=self.step_count,
        )
        """

    def close(self) -> None:
        """
        Close the environment.

        Returns:
            None.
        """
        pass
