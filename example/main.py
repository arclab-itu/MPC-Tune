from pathlib import Path
from datetime import datetime
import argparse
import time
from typing import Dict, List, Tuple

import numpy as np

from driving_imitation import DrivingImitation
from mpc.traffic.simulate import (
    run_mpc_simulation as traffic_run_mpc_simulation,
)

from utils.plotting import (
    plot_policy_evolution,
    plot_rewards_history,
    plot_all_experiments_summary,
    plot_initial_distribution,
)
from plot.plotting_trajectory import (draw,plot_sampled_trajectories)
# Per-variable clip bounds:[goal_speed, tracking, orientation, acceleration]
CLIP_LOW  = np.array([-np.inf, -np.inf, -np.inf, -np.inf])
CLIP_HIGH = np.array([ np.inf,  np.inf,  np.inf,  np.inf])

# Iterations at which to snapshot sampled weight vectors for trajectory visualisation (index so start from 0 = first iteration)
SNAPSHOT_ITERS = [0,1,2,14]


def build_training_config() -> Dict[str, float]:
    """Centralize training hyperparameters and plotting metadata."""
    return {
        "max_iter": 15,
        "samples_per_iter": 10,  # HighMPC.Policy.N
        "beta": 3.0,              # Hyperparameter for scaling rewards in the expectation step
        "clip_low": CLIP_LOW,     # shape (4,): [goal_speed, tracking, orientation, acceleration]
        "clip_high": CLIP_HIGH,   # shape (4,): [goal_speed, tracking, orientation, acceleration]
        # Initial Gaussian for weights [gs, tracking, orientation, acceleration]
        "mean_init": 13.0,
        "var_init":  25.0,
    }


def weights_from_vector(vec: np.ndarray) -> Dict[str, float]:
    """Map a 4-dim vector to the MPC weight dictionary with rounding."""
    return {
        "goal_speed": round(float(vec[0]), 2),
        "tracking": round(float(vec[1]), 2),
        "orientation": round(float(vec[2]), 2),
        "acceleration": round(float(vec[3]), 2),
    }

def run_experiment(
    exp_idx: int,
    initial_state: np.ndarray,
    baseline_weights: Dict[str, float],
    progress_plots_dir: Path,
    example_folder: Path,
) -> Tuple[Dict[str, float], Dict]:
    """Run a single experiment: baseline rollout, train HighMPC, compare rollout, save plot."""
    cfg = build_training_config()

    # 1) Baseline rollout to create target trajectory
    baseline_states, _, _, ref_track = traffic_run_mpc_simulation(
        initial_state, baseline_weights
    )

    # 2) HighMPC setup and training
    D = 4  # number of weights
    mean = [cfg["mean_init"]] * D
    covariance = np.diag([cfg["var_init"]] * D)
    N = cfg["samples_per_iter"]

    trainer = DrivingImitation(
        mean=mean,
        covariance=covariance,
        N=N,
        cliplow=cfg["clip_low"],
        cliphigh=cfg["clip_high"],
        target_trajectory=baseline_states,
        MPC=traffic_run_mpc_simulation,
    )

    t0 = time.time()
    mean_vec, history = trainer.policy_search(
        initial_state=initial_state,
        max_iter=cfg["max_iter"],
        beta=cfg["beta"],
        track_history=True,
        snapshot_iters=SNAPSHOT_ITERS,
    )
    elapsed = time.time() - t0

    learned_weights = weights_from_vector(mean_vec)

    # 3) Rollout with learned weights
    learned_states, _, _, _ = traffic_run_mpc_simulation(initial_state, learned_weights)

    # 4) Persist plots
    details = {
        "iterations": cfg["max_iter"],
        "samples_per_iter": cfg["samples_per_iter"],
        "beta": cfg["beta"],
        "train_time_sec": round(elapsed, 2),
    }

    # example_folder/: trajectory
    draw(
        baseline_weights=baseline_weights,
        baseline_traj=baseline_states,
        learned_weights=learned_weights,
        learned_traj=learned_states,
        details=details,
        ref_track=ref_track,
        data_path=example_folder,
        exp_idx=exp_idx,
    )

    # progress_plots/: policy evolution and reward
    plot_policy_evolution(baseline_weights, history, progress_plots_dir, exp_idx)
    plot_rewards_history(history, progress_plots_dir, exp_idx)

    # example_folder/progress_over_iterations/: snapshot trajectory plots
    progress_dir = example_folder / "progress_over_iterations" / f"experiment_{exp_idx}"
    progress_dir.mkdir(parents=True, exist_ok=True)

    for iter_idx, z in history["sampled_weights"].items():
        trajectories = []
        for i in range(len(z)):
            w = weights_from_vector(z[i])
            traj, _, _, _ = traffic_run_mpc_simulation(initial_state, w)
            trajectories.append(traj)
        plot_sampled_trajectories(
            trajectories=trajectories,
            target_trajectory=baseline_states,
            iter_idx=iter_idx,
            exp_idx=exp_idx,
            data_path=progress_dir,
        )

    print(
        f"Experiment {exp_idx}: learned {learned_weights} from baseline {baseline_weights} in {elapsed:.2f}s"
    )

    return learned_weights, history

def generate_random_experiments(n: int) -> Tuple[List[np.ndarray], List[Dict[str, float]]]:
    """Generate n initial states and baseline weight dictionaries."""
    states: List[np.ndarray] = []
    weights: List[Dict[str, float]] = []

    for _ in range(n):
        x = round(np.random.uniform(1, 5), 2)
        y = round(np.random.uniform(1, 15), 2)
        weights.append(
            {
                "goal_speed": round(np.random.uniform(2, 8.0), 2),
                "tracking": round(np.random.uniform(2, 8.0), 2),
                "orientation": round(np.random.uniform(2, 8.0), 2),
                "acceleration": round(np.random.uniform(2, 8.0), 2),
            }
        )
        
        # [x, y, heading, vx, vy, omega]
        states.append(np.array([x, y, 0.0, 1.0, 0.0, 0.0]))

    return states, weights


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--progress-plots", type=str, required=True, help="Path to folder for progress plots (policy evolution, rewards, initial distribution)")
    parser.add_argument("--example-folder", type=str, required=True, help="Path to folder for experiment results (trajectory, summary, progress_over_iterations)")
    args = parser.parse_args()

    progress_plots_dir = Path(args.progress_plots)
    example_folder = Path(args.example_folder)

    if not progress_plots_dir.exists():
        raise FileNotFoundError(f"progress-plots folder does not exist: {progress_plots_dir}")
    if not example_folder.exists():
        raise FileNotFoundError(f"example-folder does not exist: {example_folder}")

    NUM_EXPERIMENTS = 1
    states, weights = generate_random_experiments(NUM_EXPERIMENTS)
    # Plot initial distribution using configured mean/variance
    cfg = build_training_config()
    plot_initial_distribution(progress_plots_dir, cfg["mean_init"], cfg["var_init"])

    overall_start = time.time()
    all_histories = []
    finals = []
    for idx, (state, w) in enumerate(zip(states, weights), start=1):
        print(f"Running experiment {idx} with state={state} and baseline weights={w}")
        learned_weights, history = run_experiment(
            exp_idx=idx,
            initial_state=state,
            baseline_weights=w,
            progress_plots_dir=progress_plots_dir,
            example_folder=example_folder,
        )
        all_histories.append(history)
        finals.append({"experiment": idx, "state": state, "baseline_weights": w, "learned_weights": learned_weights})

    overall_elapsed = time.time() - overall_start

    # Save final results
    for res in finals:
        print(f"\nExperiment {res['experiment']}:")
        print(f"  Initial State: {res['state']}")
        print(f"  Baseline Weights: {res['baseline_weights']}")
        print(f"  Learned Weights: {res['learned_weights']}")

    # Create summary plot across all experiments
    print(f"\nCreating summary plot across all {NUM_EXPERIMENTS} experiments...")
    plot_all_experiments_summary(all_histories, example_folder)

    print(f"\n✅ All experiments completed in {overall_elapsed:.2f}s (~{overall_elapsed/60:.2f} min)")
    print(f"📁 Progress plots saved to: {progress_plots_dir}")
    print(f"📁 Results saved to: {example_folder}")



if __name__ == "__main__":
    main()

    