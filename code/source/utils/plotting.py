
from datetime import datetime
from pathlib import Path
import random
import sys
from matplotlib import pyplot as plt

IS_INTERACTIVE = hasattr(sys, 'ps1')
date_now = datetime.now()

def plot_evolution(
    values: list,
    plotting: str,
    dataset: str,
    datamode: str,
    color: tuple[float, float, float] | None = None,
    clean: bool = True
) -> None:
    """
    Plots the evolution of a metric (e.g., loss or accuracy) over epochs.

    Parameters
    ----------
    values : list
        Sequence of metric values per epoch (e.g., training loss).
    plotting : str
        Name of the metric being plotted (e.g., "Loss", "Accuracy").
    dataset : str
        Name of the dataset (used for labeling and filenames).
    datamode : str
        Mode of the dataset (e.g., "CIRRHOTIC_STATE").
    color : tuple of float, optional
        RGB color tuple for the line plot. Random if not provided.
    clean : bool, default=True
        Whether to clear the current matplotlib figure before plotting.

    """
    if not color:
        color = (random.random(), random.random(), random.random())
    if clean:
        plt.close()
    plt.plot(values, linewidth=0.7, color=color,  label=f"{dataset}")
    print(plotting)
    plt.title(f"{plotting} - {datamode}")
    plt.xlabel("Epochs")
    plt.ylabel(plotting)
    plt.legend()
    if IS_INTERACTIVE:
        plt.show()
    else:
        file = Path.cwd() / \
            f"data/plots/{plotting}/{date_now}_{dataset}_{datamode}.png"
        print(
            f"Non-interactive backend detected. Saving plot to file {file}.")
        plt.savefig(file)


def plot_conf_matrix(cm_metric, plotting: str) -> None:
    """
    Plots and optionally saves a confusion matrix from a TorchMetrics ConfusionMatrix object.

    Parameters
    ----------
    cm_metric : ConfusionMatrix
        TorchMetrics confusion matrix object with a `.plot()` method.
    plotting : str
        Label to include in the plot title and filename (e.g., "val", "test").

    """
    fig, ax = cm_metric.plot()
    plt.title(f'Confusion Matrix -{plotting} ')
    if IS_INTERACTIVE:
        plt.show()
    else:
        file = Path.cwd() / \
            f"data/plots/conf_matrix/{date_now}_cm_{plotting}.png"
        print(
            f"Non-interactive backend detected. Saving plot to file {file}.")
        plt.savefig(file)
