from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

def plot_policy_evolution(
    initial_weights: Dict[str, float],
    history: Dict,
    data_path: Path,
    exp_idx: int,
    weight_names: List[str] = ["goal_speed", "tracking", "orientation", "acceleration"],
):
    """Plot the evolution of policy mean and std deviation over iterations."""
    means = np.array(history["means"])
    stds = np.array(history["stds"])
    iterations = np.arange(len(means))

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f"Policy Distribution Evolution - Experiment {exp_idx}", fontsize=14)

    # Prepare initial weight values in the same order as weight_names
    initial_vals = [initial_weights.get(name, np.nan) for name in weight_names]
    x_max = iterations[-1] if len(iterations) > 0 else 0

    for idx, (ax, name) in enumerate(zip(axes.flat, weight_names)):
        ax.plot(iterations, means[:, idx], "b-", linewidth=2, label="Mean")
        ax.fill_between(
            iterations,
            means[:, idx] - stds[:, idx],
            means[:, idx] + stds[:, idx],
            alpha=0.3,
            label="±1 std",
        )

        init_val = initial_vals[idx]
        if not np.isnan(init_val):
            ax.hlines(init_val, 0, x_max, colors="r", linestyles="--", linewidth=1.5, label="Initial")

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Weight Value")
        ax.set_title(f"{name}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(data_path / f"experiment_{exp_idx}_policy_evolution.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_rewards_history(history: Dict, data_path: Path, exp_idx: int):
    """Plot reward statistics over training iterations."""
    rewards_mean = np.array(history["rewards_mean"])
    rewards_max = np.array(history["rewards_max"])
    rewards_min = np.array(history["rewards_min"])
    iterations = np.arange(len(rewards_mean))

    plt.figure(figsize=(10, 6))
    plt.plot(iterations, rewards_mean, "b-", linewidth=2, label="Mean Reward")
    plt.plot(iterations, rewards_max, "g--", linewidth=1, label="Max Reward")
    plt.plot(iterations, rewards_min, "r--", linewidth=1, label="Min Reward")
    plt.fill_between(iterations, rewards_min, rewards_max, alpha=0.2)

    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.title(f"Reward Evolution - Experiment {exp_idx}")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(data_path / f"experiment_{exp_idx}_rewards.png", dpi=300, bbox_inches="tight")
    plt.close()


def plot_all_experiments_summary(all_histories: List[Dict], data_path: Path):
    """Plot average rewards across all experiments."""
    num_experiments = len(all_histories)

    # Extract reward means from all experiments
    all_rewards_mean = [np.array(h["rewards_mean"]) for h in all_histories]

    # Find the minimum length (in case experiments have different iteration counts)
    min_len = min(len(r) for r in all_rewards_mean)
    all_rewards_mean = [r[:min_len] for r in all_rewards_mean]

    # Convert to array for easy computation
    rewards_array = np.array(all_rewards_mean)  # shape: (num_exp, num_iters)

    mean_across_exp = np.mean(rewards_array, axis=0)
    std_across_exp = np.std(rewards_array, axis=0)
    iterations = np.arange(min_len)

    plt.figure(figsize=(12, 6))

    for i, rewards in enumerate(all_rewards_mean):
        plt.plot(iterations, rewards, alpha=0.3, linewidth=1, color="gray")

    plt.plot(iterations, mean_across_exp, "b-", linewidth=3, label=f"Average (n={num_experiments})")
    plt.fill_between(
        iterations,
        mean_across_exp - std_across_exp,
        mean_across_exp + std_across_exp,
        alpha=0.3,
        color="blue",
        label="±1 std across experiments",
    )

    plt.xlabel("Iteration", fontsize=12)
    plt.ylabel("Mean Reward", fontsize=12)
    plt.title(f"Training Progress Across All Experiments (n={num_experiments})", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    plt.savefig(data_path / "all_experiments_summary.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"📊 Summary plot saved: all_experiments_summary.png")

def plot_initial_distribution(data_path: Path, mean: float, var: float):
    """Plot the initial weight distribution used for HighMPC.

    Parameters:
        data_path: directory to save the plot
        mean: mean of initial distribution
        var: variance of initial distribution
    """
    std = np.sqrt(var)

    x = np.linspace(mean - 4 * std, mean + 4 * std, 1000)
    y = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="Initial Weight Distribution", color="blue")

    for i in range(-3, 4):
        plt.axvline(mean + i * std, color="green", linestyle="--", alpha=0.8)
        plt.text(mean + i * std, max(y) * 0.1, f"{mean + i * std:.2f}", rotation=270, verticalalignment="bottom", color="black")

    plt.title("Initial Weight Distribution for HighMPC")
    plt.xlabel("Weight Value")
    plt.ylabel("Probability Density")
    plt.legend()
    plt.grid(True)

    plt.savefig(data_path / "initial_weight_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
