import numpy as np
from abc import ABC, abstractmethod
from typing import Callable
from multiprocessing import Pool

class Policy:

    def __init__(self, mean, covariance, N, cliplow, cliphigh):
        """
        Initialize the Policy (Gaussian distribution over MPC weight vectors).

        Args:
            mean      (list | np.ndarray, shape (num_weights,)):
                          Initial mean of the Gaussian — one value per MPC weight
            covariance (list | np.ndarray, shape (num_weights, num_weights)):
                          Initial covariance matrix
            N         (int):
                          Number of samples to draw per policy-search iteration.
            For cliplow and cliphigh, a per-variable bound can be specified by passing a length-num_weights array, or a scalar can be passed to use the same bound for all weights.
            cliplow   (float | list | np.ndarray, shape (num_weights,)):
                          Per-variable lower clip bound applied after sampling.
                          Pass a scalar to use the same bound for all weights,
                          or a length-4 array for per-weight bounds.
            cliphigh  (float | list | np.ndarray, shape (num_weights,)):
                          Per-variable upper clip bound applied after sampling.
                          Same convention as cliplow.
        """
        self.mean = np.array(mean, dtype=float)
        self.covariance = np.array(covariance, dtype=float)
        self.N = N
        self.cliplow  = np.asarray(cliplow,  dtype=float) 
        self.cliphigh = np.asarray(cliphigh, dtype=float)  

    def sample(self):
        """
        Draw N weight-vector samples from the current policy distribution.
softmax
        Returns
        -------
        samples : np.ndarray, shape (N, num_weights)
            Sampled weight vectors, clipped to [cliplow, cliphigh] per variable.
        """
        samples = np.random.multivariate_normal(
            self.mean, self.covariance, self.N)                               
        samples = np.clip(samples, self.cliplow, self.cliphigh)
        return samples 

    def update(self, weights, samples):
        """
        Update policy mean and covariance via importance-weighted EM.

        Args:
            weights (np.ndarray, shape (N,)):
                Non-negative importance weights from the expectation step.
            samples (np.ndarray, shape (N, num_weights)):
                The same sample matrix returned by sample().

        Effects
        ------------
        self.mean       updated in-place, shape (num_weights,)
        self.covariance updated in-place, shape (num_weights, num_weights)
        """
        self.mean = weights.dot(samples) / np.sum(weights + 1e-8)              # mean update as weighted average of samples, shape (num_weights,) = (N,) dot (N, num_weights) / scalar
        Y = (np.sum(weights)**2 - np.sum(weights**2)) / np.sum(weights)        
        sum = 0.0
        for i in range(len(weights)):                                          
            diff = samples[i] - self.mean
            sum += weights[i] * np.outer(diff, diff)                           # covariance update as weighted average of outer products of sample deviations, shape (num_weights, num_weights) = scalar * (num_weights, num_weights)
        self.covariance = sum / (Y + 1e-8)

    def expectation(self, rewards, beta):
        """
        Normalize rewards
        Compute importance weights from a reward vector[scale using beta , exponential]

        Args:
            rewards (np.ndarray, shape (N,)):
                Scalar reward for each of the N sampled trajectories.
            beta (float):
                hyperparameter for scaling

        Returns
        -------
        weights : np.ndarray, shape (N,)
            Non-negative importance weights
        """
        mean = np.mean(rewards)
        std = np.std(rewards)

        std = 1e-8 if std == 0 else std                                 # prevent division by zero
        normalized_rewards = (rewards - mean) / std
        weights = np.exp(beta * normalized_rewards)
        return weights


class BasePolicySearch(ABC):                                                # Abstract base class for policy search algorithms, requires implementation of reward and policy_search methods by subclasses

    def __init__(self, mean, covariance, N, cliplow=-np.inf, cliphigh=np.inf):
        self.policy = Policy(mean, covariance, N, cliplow, cliphigh)        # composiing the policy search class with a policy instance

    @abstractmethod
    def reward(self, sampled_trajectory: np.ndarray) -> float: ...

    @abstractmethod
    def policy_search(self, initial_state, max_iter, beta, **kwargs): ...