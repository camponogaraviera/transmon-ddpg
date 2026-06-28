from pathlib import Path
from collections.abc import Sequence
from typing import Any
import matplotlib.pyplot as plt
import numpy as np


def plot_learning_curve(
    scores: Sequence[float],
    output_dir: str | Path,
    filename_stem: str = "score_trend",
) -> None:
    """Plot and save the learning curve.

    Args:
        scores: Sequence of episode scores.
        output_dir: Directory to save the plot.
        filename_stem: Base filename without extension.

    Returns:
        None.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / (filename_stem + ".png")

    x = np.arange(1, len(scores) + 1)

    plt.plot(x, scores)
    plt.title("Learning Curve")
    plt.xlabel("Training Episodes")
    plt.ylabel("RL Score")
    plt.savefig(file_path)
    plt.close()


def plot_transmon_energy_levels(
    transmon: Any,
    output_dir: str | Path,
    episode_id: int,
    step_count: int,
    ej: float,
    ec: float,
) -> Path:
    """Plot transmon energy levels and save the figure.

    Args:
        transmon: Transmon object exposing `plot_evals_vs_paramvals`.
        output_dir: Directory to save the figure.
        episode_id: Current episode index.
        step_count: Current step index.
        ej: Josephson energy.
        ec: Charging energy.

    Returns:
        Saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    step_label = f"Episode: {episode_id}"
    ej_label = f"Ej: {ej:.2f}"
    ec_label = f"Ec: {ec:.2f}"
    ej_over_ec_label = f"Ej/Ec: {ej/ec:.2f}"
    ng_list = np.linspace(-1, 1, 100)

    fig, axes = transmon.plot_evals_vs_paramvals(
        "ng", ng_list, evals_count=6, subtract_ground=False
    )
    transmon.plot_evals_vs_paramvals(
        "ng",
        ng_list,
        evals_count=6,
        subtract_ground=False,
        fig_ax=(fig, axes),
    )
    axes.legend(
        [step_label, ej_label, ec_label, ej_over_ec_label],
        loc="upper right",
    )

    file_path = output_dir / (
        f"anharmonicity_at_episode_{episode_id:04d}_step_{step_count:03d}.png"
    )
    plt.savefig(file_path)
    plt.close()
    return file_path


def plot_transmon_coherence(
    transmon: Any,
    output_dir: str | Path,
    episode_id: int,
    step_count: int,
) -> Path:
    """Plot transmon coherence trends and save the figure.

    Args:
        transmon: Transmon object exposing `plot_coherence_vs_paramvals`.
        output_dir: Directory to save the figure.
        episode_id: Current episode index.
        step_count: Current step index.

    Returns:
        Saved file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    transmon.plot_t2_effective_vs_paramvals(
        param_name="ng",
        param_vals=np.linspace(-0.5, 0.5, 100),
        common_noise_options={"i": 3, "j": 2, "total": False},
        color="brown",
    )

    file_path = output_dir / (
        f"coherence_at_episode_{episode_id:04d}_step_{step_count:03d}.png"
    )
    plt.savefig(file_path)
    plt.close()
    return file_path
