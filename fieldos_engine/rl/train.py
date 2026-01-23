"""Training routines for RL agents."""

import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field

from .env import PlayCallingEnv
from .policy import EpsilonGreedyBandit, UpperConfidenceBound, RandomPolicy
from ..core.models import Role


@dataclass
class TrainingConfig:
    """Training configuration."""
    total_steps: int = 1000
    eval_frequency: int = 100
    eval_episodes: int = 20
    algorithm: str = "BANDIT"  # BANDIT, UCB, or PPO
    seed: Optional[int] = None

    # Bandit-specific
    epsilon: float = 0.1
    learning_rate: float = 0.1

    # UCB-specific
    ucb_c: float = 2.0


@dataclass
class TrainingResult:
    """Training results."""
    algorithm: str
    total_steps: int
    final_reward_mean: float
    final_reward_std: float
    reward_history: List[float] = field(default_factory=list)
    eval_history: List[Dict] = field(default_factory=list)
    best_actions_per_bucket: Dict[int, int] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "algorithm": self.algorithm,
            "total_steps": self.total_steps,
            "final_reward_mean": self.final_reward_mean,
            "final_reward_std": self.final_reward_std,
            "reward_history_length": len(self.reward_history),
            "eval_history": self.eval_history,
            "best_actions_per_bucket": self.best_actions_per_bucket
        }


def train_bandit(
    env: PlayCallingEnv,
    config: TrainingConfig
) -> Tuple[TrainingResult, Any]:
    """
    Train epsilon-greedy or UCB bandit.

    Args:
        env: PlayCallingEnv
        config: Training configuration

    Returns:
        (TrainingResult, Policy)
    """
    # Create policy
    if config.algorithm == "UCB":
        policy = UpperConfidenceBound(
            num_actions=env.action_space.n,
            c=config.ucb_c,
            seed=config.seed
        )
    else:  # BANDIT (epsilon-greedy)
        policy = EpsilonGreedyBandit(
            num_actions=env.action_space.n,
            epsilon=config.epsilon,
            learning_rate=config.learning_rate,
            seed=config.seed
        )

    reward_history = []
    eval_history = []

    for step in range(config.total_steps):
        # Reset environment
        obs, info = env.reset()

        # Select action
        action = policy.predict(obs, deterministic=False)

        # Take step
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        # Update policy
        policy.update(obs, action, reward)

        reward_history.append(reward)

        # Evaluation
        if (step + 1) % config.eval_frequency == 0:
            eval_metrics = evaluate_policy(env, policy, config.eval_episodes, config.seed)
            eval_history.append({
                "step": step + 1,
                "mean_reward": eval_metrics["mean_reward"],
                "std_reward": eval_metrics["std_reward"]
            })

    # Final evaluation
    final_eval = evaluate_policy(env, policy, config.eval_episodes, config.seed)

    # Get best actions per bucket
    best_actions = policy.get_best_actions_per_bucket()

    result = TrainingResult(
        algorithm=config.algorithm,
        total_steps=config.total_steps,
        final_reward_mean=final_eval["mean_reward"],
        final_reward_std=final_eval["std_reward"],
        reward_history=reward_history,
        eval_history=eval_history,
        best_actions_per_bucket=best_actions
    )

    return result, policy


def train_ppo(
    env: PlayCallingEnv,
    config: TrainingConfig
) -> Tuple[TrainingResult, Any]:
    """
    Train PPO agent using stable-baselines3.

    Args:
        env: PlayCallingEnv
        config: Training configuration

    Returns:
        (TrainingResult, PolicyModel)
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv

        # Wrap env
        vec_env = DummyVecEnv([lambda: env])

        # Create PPO model
        model = PPO(
            "MlpPolicy",
            vec_env,
            verbose=0,
            seed=config.seed
        )

        # Train
        model.learn(total_timesteps=config.total_steps)

        # Evaluate
        eval_metrics = evaluate_sb3_policy(env, model, config.eval_episodes)

        result = TrainingResult(
            algorithm="PPO",
            total_steps=config.total_steps,
            final_reward_mean=eval_metrics["mean_reward"],
            final_reward_std=eval_metrics["std_reward"],
            reward_history=[],  # Not tracked for SB3
            eval_history=[]
        )

        return result, model

    except ImportError:
        raise ImportError(
            "stable-baselines3 not installed. Install with: pip install stable-baselines3"
        )


def evaluate_policy(
    env: PlayCallingEnv,
    policy: Any,
    num_episodes: int,
    seed: Optional[int] = None
) -> Dict:
    """
    Evaluate policy performance.

    Args:
        env: Environment
        policy: Policy to evaluate
        num_episodes: Number of episodes
        seed: Random seed

    Returns:
        Dictionary with evaluation metrics
    """
    rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode if seed else None)
        action = policy.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards))
    }


def evaluate_sb3_policy(env: PlayCallingEnv, model: Any, num_episodes: int) -> Dict:
    """Evaluate stable-baselines3 policy."""
    rewards = []

    for episode in range(num_episodes):
        obs, info = env.reset()
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        rewards.append(reward)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "min_reward": float(np.min(rewards)),
        "max_reward": float(np.max(rewards))
    }


def create_env_from_config(
    playbook: List[str],
    scenarios: List[str],
    offensive_players: Dict[Role, str],
    defensive_players: Dict[Role, str],
    seed: Optional[int] = None
) -> PlayCallingEnv:
    """Create environment from configuration."""
    return PlayCallingEnv(
        playbook=playbook,
        scenarios=scenarios,
        offensive_players=offensive_players,
        defensive_players=defensive_players,
        seed=seed
    )
