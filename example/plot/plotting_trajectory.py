from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def draw(
    baseline_weights: Dict[str, float],
    baseline_traj: np.ndarray,
    learned_weights: Dict[str, float],
    learned_traj: np.ndarray,
    details: Dict[str, float],
    ref_track,
    data_path: Path,
    exp_idx: int,
):
    """Render and persist a side-by-side trajectory plot with run details."""
    plt.figure(figsize=(12, 6))
    plt.plot(
        baseline_traj[:, 0], baseline_traj[:, 1], "bo-", label=f"Baseline trajectory", linewidth=1, markersize=1
    )
    track_coords = np.array(ref_track.coords)
    plt.plot(track_coords[:, 0], track_coords[:, 1], "k--", label="Reference Track", linewidth=2)
    plt.plot(
        learned_traj[:, 0], learned_traj[:, 1], "go-", label=f"Learned trajectory", linewidth=1, markersize=1
    )
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Vehicle Trajectory Using MPC")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    # Add details text on the right side
    details_text = ""
    details_text += "Baseline Weights:\n"
    for key, value in baseline_weights.items():
        details_text += f"  {key}: {value}\n"
    details_text += "\nFinal Learned Weights:\n"
    for key, value in learned_weights.items():
        details_text += f"  {key}: {value}\n"
    details_text += "\nTraining Details:\n"
    for key, value in details.items():
        details_text += f"  {key}: {value}\n"
    plt.text(
        1.02,
        0.5,
        details_text,
        transform=plt.gca().transAxes,
        verticalalignment="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
    )

    name = f"experiment_{exp_idx}_trajectory.png"
    plt.savefig(data_path / name, dpi=300, bbox_inches="tight")
    plt.close()

def plot_sampled_trajectories(
    trajectories: List[np.ndarray],
    target_trajectory: np.ndarray,
    iter_idx: int,
    exp_idx: int,
    data_path: Path,
):
    """Plot all sampled trajectories at a given snapshot iteration.

    Args:
        trajectories      (list of np.ndarray, each shape (T, state_dim)):
                              One trajectory per sampled weight vector.
        target_trajectory (np.ndarray, shape (T, state_dim)):
                              Baseline/target trajectory for reference.
        iter_idx          (int):   Iteration index this snapshot was taken at.
        exp_idx           (int):   Experiment index (for file naming).
        data_path         (Path):  Directory to save the image.
    """
    _SAMPLE_COLORS = [
        "#8B0000",  # dark red
        "#A0522D",  # sienna (dark brownish)
        "#6B3A2A",  # dark brown
        "#B8860B",  # dark goldenrod
        "#556B2F",  # dark olive green
        "#2F4F4F",  # dark slate gray
        "#4B0082",  # indigo
        "#800080",  # purple
        "#8B4513",  # saddle brown
        "#C0392B",  # dark crimson
    ]

    plt.figure(figsize=(12, 6))

    for traj in trajectories:
        color = _SAMPLE_COLORS[np.random.randint(len(_SAMPLE_COLORS))]
        plt.plot(traj[:, 0], traj[:, 1], "-", color=color, alpha=0.4, linewidth=0.8)

    plt.plot(
        target_trajectory[:, 0], target_trajectory[:, 1],
        "k--", linewidth=2, label="Target trajectory",
    )

    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Vehicle Trajectory Using MPC")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")

    details_text = f"Experiment: {exp_idx}\nIteration: {iter_idx}\nSamples: {len(trajectories)}"
    plt.text(
        1.02, 0.5,
        details_text,
        transform=plt.gca().transAxes,
        verticalalignment="center",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black"),
    )

    plt.savefig(data_path / f"iter_{iter_idx}_sampled_trajectories.png", dpi=300, bbox_inches="tight")
    plt.close()