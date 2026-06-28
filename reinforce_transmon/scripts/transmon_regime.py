"""Module to plot the transmon regime curve for different ng values."""

from collections.abc import Sequence
from pathlib import Path
import matplotlib.pyplot as plt
import scqubits as scq
import numpy as np


def relative_anharmonicity(ej: float, ec: float, ng: float) -> float:
    """Compute relative anharmonicity for given parameters.

    Args:
        ej: Josephson energy.
        ec: Charging energy.
        ng: Offset charge.

    Returns:
        Relative anharmonicity value.
    """

    transmon = scq.Transmon(EJ=ej, EC=ec, ng=ng, ncut=31)
    energies = transmon.eigenvals(evals_count=3)
    E01 = energies[1] - energies[0]
    E12 = energies[2] - energies[1]
    return (E12 - E01) / E01


def plot_anharmonicity(
    ng_values: Sequence[float],
    ej_values: Sequence[float],
    output_dir: str | Path,
) -> None:
    """Plot relative anharmonicity curves and save the figure.

    Args:
        ng_values: Iterable of ng values to plot.
        ej_values: Iterable of Ej values for the x-axis.
        output_dir: Directory to save the plot.

    Returns:
        None.
    """

    output_dir = Path(output_dir)

    _, ax = plt.subplots()
    ax.set_title("Transmon Regime")
    ax.set_xlabel(r"$E_j/E_c$")
    ax.set_ylabel("Relative Anharmonicity")

    for ng in ng_values:
        y = [relative_anharmonicity(ej, ec=1, ng=ng) for ej in ej_values]
        ax.plot(ej_values, y, label=f"$E_c$ = 1, $ng$ = {ng:.2f}")

    ax.legend()

    # Save plot:
    plt.savefig(output_dir / "transmon_regime.png")
    print(f"Transmon regime plot saved at {output_dir}")


def main() -> None:
    """Entry point for generating the transmon regime plot.

    Returns:
        None.
    """

    # Set directory for saving plots:
    project_root = Path(__file__).resolve().parents[2]
    output_dir = project_root / "assets" / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    ng_values = np.linspace(0, 0.4, 10)
    ej_values = np.linspace(0, 20, 50)

    # Plot and save at ~/assets/plots/:
    plot_anharmonicity(ng_values, ej_values, output_dir)


if __name__ == "__main__":
    main()
