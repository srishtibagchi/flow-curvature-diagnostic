"""Notebook-friendly plotting without implicit display or filesystem writes."""

from matplotlib import pyplot as plt
from matplotlib.figure import Figure

from .data import Array


def plot_trajectories(
    trajectories: dict[str, Array], max_paths: int = 32
) -> Figure:
    if not trajectories or max_paths < 1:
        raise ValueError("Provide trajectories and a positive max_paths.")
    figure, axes = plt.subplots(
        1, len(trajectories), figsize=(5 * len(trajectories), 4), squeeze=False,
        sharex=True, sharey=True,
    )
    for axis, (name, paths) in zip(axes[0], trajectories.items()):
        for i in range(min(max_paths, paths.shape[1])):
            axis.plot(paths[:, i, 0], paths[:, i, 1], alpha=0.55, linewidth=1)
        axis.scatter(paths[0, :max_paths, 0], paths[0, :max_paths, 1], s=12, label="source")
        axis.scatter(paths[-1, :max_paths, 0], paths[-1, :max_paths, 1], s=12, label="endpoint")
        axis.set(title=name, xlabel="x", ylabel="y", aspect="equal")
        axis.legend()
    figure.tight_layout()
    return figure
