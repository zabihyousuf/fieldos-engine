"""Baseline policies for play-calling."""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict

from ..core.models import DownDistanceBucket


class BasePolicy:
    """Base policy interface."""

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Predict action given observation."""
        raise NotImplementedError

    def learn(self, *args, **kwargs):
        """Learn from experience (for trainable policies)."""
        pass

    def save(self, path: str):
        """Save policy to disk."""
        raise NotImplementedError

    def load(self, path: str):
        """Load policy from disk."""
        raise NotImplementedError


class RandomPolicy(BasePolicy):
    """Random play selection."""

    def __init__(self, num_actions: int, seed: Optional[int] = None):
        self.num_actions = num_actions
        self.rng = np.random.Generator(np.random.PCG64(seed))

    def predict(self, observation: np.ndarray, deterministic: bool = True) -> int:
        """Select random action."""
        return int(self.rng.integers(0, self.num_actions))

    def save(self, path: str):
        pass  # Nothing to save

    def load(self, path: str):
        pass


class EpsilonGreedyBandit(BasePolicy):
    """
    Epsilon-greedy contextual bandit for play-calling.

    Learns Q(bucket, play) and selects greedily with epsilon exploration.
    """

    def __init__(
        self,
        num_actions: int,
        epsilon: float = 0.1,
        learning_rate: float = 0.1,
        seed: Optional[int] = None
    ):
        self.num_actions = num_actions
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.rng = np.random.Generator(np.random.PCG64(seed))

        # Q values: bucket -> action -> q_value
        self.q_values: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions)
        )
        self.action_counts: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions, dtype=int)
        )

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select action using epsilon-greedy."""
        bucket_idx = self._get_bucket_from_obs(observation)

        if not deterministic and self.rng.random() < self.epsilon:
            # Explore
            return int(self.rng.integers(0, self.num_actions))
        else:
            # Exploit
            q_vals = self.q_values[bucket_idx]
            return int(np.argmax(q_vals))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float
    ):
        """Update Q-value based on reward."""
        bucket_idx = self._get_bucket_from_obs(observation)

        # Incremental mean update
        count = self.action_counts[bucket_idx][action]
        old_q = self.q_values[bucket_idx][action]

        # Q(s,a) <- Q(s,a) + alpha * (reward - Q(s,a))
        self.q_values[bucket_idx][action] = old_q + self.learning_rate * (reward - old_q)

        self.action_counts[bucket_idx][action] += 1

    def _get_bucket_from_obs(self, observation: np.ndarray) -> int:
        """Extract bucket index from observation (first 7 dims are one-hot)."""
        bucket_onehot = observation[:7]
        return int(np.argmax(bucket_onehot))

    def get_best_actions_per_bucket(self) -> Dict[int, int]:
        """Get best action for each bucket."""
        best_actions = {}
        for bucket_idx, q_vals in self.q_values.items():
            best_actions[bucket_idx] = int(np.argmax(q_vals))
        return best_actions

    def get_q_table(self) -> Dict[int, np.ndarray]:
        """Get Q-value table."""
        return dict(self.q_values)

    def save(self, path: str):
        """Save policy."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                "q_values": dict(self.q_values),
                "action_counts": dict(self.action_counts),
                "epsilon": self.epsilon,
                "learning_rate": self.learning_rate,
                "num_actions": self.num_actions
            }, f)

    def load(self, path: str):
        """Load policy."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.epsilon = data["epsilon"]
            self.learning_rate = data["learning_rate"]
            self.num_actions = data["num_actions"]
            
            # Restore defaultdicts
            self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
            self.q_values.update(data["q_values"])
            
            self.action_counts = defaultdict(lambda: np.zeros(self.num_actions, dtype=int))
            self.action_counts.update(data["action_counts"])


class UpperConfidenceBound(BasePolicy):
    """
    UCB (Upper Confidence Bound) bandit for play-calling.

    Balances exploration and exploitation using confidence bounds.
    """

    def __init__(
        self,
        num_actions: int,
        c: float = 2.0,
        seed: Optional[int] = None
    ):
        self.num_actions = num_actions
        self.c = c  # Exploration constant
        self.rng = np.random.Generator(np.random.PCG64(seed))

        self.q_values: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions)
        )
        self.action_counts: Dict[int, np.ndarray] = defaultdict(
            lambda: np.zeros(num_actions, dtype=int)
        )
        self.total_counts: Dict[int, int] = defaultdict(int)

    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> int:
        """Select action using UCB."""
        bucket_idx = self._get_bucket_from_obs(observation)

        q_vals = self.q_values[bucket_idx]
        counts = self.action_counts[bucket_idx]
        total = self.total_counts[bucket_idx]

        if total == 0:
            # First time seeing this bucket, random action
            return int(self.rng.integers(0, self.num_actions))

        # Compute UCB values
        ucb_values = np.zeros(self.num_actions)
        for a in range(self.num_actions):
            if counts[a] == 0:
                ucb_values[a] = float('inf')  # Always try untried actions
            else:
                bonus = self.c * np.sqrt(np.log(total) / counts[a])
                ucb_values[a] = q_vals[a] + bonus

        return int(np.argmax(ucb_values))

    def update(
        self,
        observation: np.ndarray,
        action: int,
        reward: float
    ):
        """Update Q-value."""
        bucket_idx = self._get_bucket_from_obs(observation)

        count = self.action_counts[bucket_idx][action]
        old_q = self.q_values[bucket_idx][action]

        # Incremental mean
        self.q_values[bucket_idx][action] = (old_q * count + reward) / (count + 1)

        self.action_counts[bucket_idx][action] += 1
        self.total_counts[bucket_idx] += 1

    def _get_bucket_from_obs(self, observation: np.ndarray) -> int:
        """Extract bucket index from observation."""
        bucket_onehot = observation[:7]
        return int(np.argmax(bucket_onehot))

    def get_best_actions_per_bucket(self) -> Dict[int, int]:
        """Get best action for each bucket."""
        best_actions = {}
        for bucket_idx, q_vals in self.q_values.items():
            best_actions[bucket_idx] = int(np.argmax(q_vals))
        return best_actions

    def save(self, path: str):
        """Save policy."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                "q_values": dict(self.q_values),
                "action_counts": dict(self.action_counts),
                "total_counts": dict(self.total_counts),
                "c": self.c,
                "num_actions": self.num_actions
            }, f)

    def load(self, path: str):
        """Load policy."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.c = data["c"]
            self.num_actions = data["num_actions"]
            
            self.q_values = defaultdict(lambda: np.zeros(self.num_actions))
            self.q_values.update(data["q_values"])
            
            self.action_counts = defaultdict(lambda: np.zeros(self.num_actions, dtype=int))
            self.action_counts.update(data["action_counts"])
            
            self.total_counts = defaultdict(int)
            self.total_counts.update(data["total_counts"])
