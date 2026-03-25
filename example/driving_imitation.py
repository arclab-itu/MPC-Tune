
from policy.Policy import BasePolicySearch
from mpc.traffic.simulate import run_mpc_simulation as traffic_run_mpc_simulation
import numpy as np
from multiprocessing import Pool
from typing import Callable

class DrivingImitation(BasePolicySearch):
    """
        Probabilistic Policy Search for MPC
    """

    def __init__(self, mean, covariance, N, cliplow=-np.inf, cliphigh=np.inf, target_trajectory=None, MPC: Callable = None):
        super().__init__(mean, covariance, N, cliplow, cliphigh)
        self.target_trajectory = target_trajectory
        self.MPC = MPC

    def reward(self, sampled_trajectory: np.ndarray):                      # reward is negative distance between sampled trajectory and target trajectory, averaged over the length of the shorter trajectory
        # choose len of smaller trajectory and calculate reward using equilidian distance input is array x y points
        dimensions = sampled_trajectory.shape[1]
        min_len = min(len(sampled_trajectory), len(self.target_trajectory))

        total_reward = 0
        for i in range(min_len):
            reward = 0
            for d in range(dimensions):
                reward += (((sampled_trajectory[i][d] -
                           self.target_trajectory[i][d])**2))
            # negative of distance as reward
            total_reward += (-np.sqrt(reward))

        return total_reward / min_len

    def policy_search(self, initial_state, max_iter, beta, track_history=True, snapshot_iters=None):
        """
        Run policy search and optionally track training history.

        Args:
            initial_state  (np.ndarray, shape (state_dim,)):  Starting state for MPC rollouts.
            max_iter       (int):                             Number of EM iterations.
            beta           (float):                           Inverse temperature for expectation step.
            track_history  (bool):                            Record per-iteration statistics.
            snapshot_iters (list[int] | None):                Iteration indices at which to save
                                                              the full sample matrix z for later
                                                              trajectory visualisation.

        Returns:
            mean    (np.ndarray, shape (num_weights,)):  Learned policy mean.
            history (dict | None):                       Per-iteration stats when track_history=True.
                                                         Includes 'sampled_weights': dict mapping
                                                         snapshot iteration index -> z (N, num_weights).
        """
        snapshot_iters = set(snapshot_iters) if snapshot_iters is not None else set()

        history = {
            'means': [],
            'stds': [],
            'rewards_mean': [],
            'rewards_max': [],
            'rewards_min': [],
            'sampled_weights': {},   # iter_idx -> np.ndarray (N, num_weights)
        } if track_history else None

        for iter_idx in range(max_iter):
            rewards = np.zeros(shape=(self.policy.N))
            z = self.policy.sample()

            # Save sample matrix for snapshot iterations
            if track_history and iter_idx in snapshot_iters:
                history['sampled_weights'][iter_idx] = z.copy()

            with Pool() as pool:
                results = []
                for i in range(self.policy.N):
                    __weights = {
                        "goal_speed": round(float(z[i][0]), 2),
                        "tracking": round(float(z[i][1]), 2),
                        "orientation": round(float(z[i][2]), 2),
                        "acceleration": round(float(z[i][3]), 2),
                    }
                    results.append(
                        pool.apply_async(
                            self.MPC, args=(initial_state, __weights)
                        )
                    )

                for i in range(self.policy.N):
                    sampled_trajectory, _, _, _ = results[i].get()
                    rewards[i] = self.reward(sampled_trajectory)

            weights = self.policy.expectation(rewards, beta)

            self.policy.update(weights, z)

            if track_history:
                history['means'].append(self.policy.mean.copy())
                history['stds'].append(np.sqrt(np.diag(self.policy.covariance)).copy())
                history['rewards_mean'].append(np.mean(rewards))
                history['rewards_max'].append(np.max(rewards))
                history['rewards_min'].append(np.min(rewards))

        if track_history:
            return self.policy.mean, history
        return self.policy.mean
        

