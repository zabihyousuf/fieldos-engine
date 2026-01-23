"""Evaluation and analysis of trained policies."""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict

from .env import PlayCallingEnv
from .policy import BasePolicy
from ..core.models import DownDistanceBucket, GameSituation


@dataclass
class BucketPerformance:
    """Performance metrics for a specific bucket."""
    bucket: str
    num_episodes: int
    mean_reward: float
    std_reward: float
    best_play_id: Optional[str] = None
    play_distribution: Dict[str, int] = field(default_factory=dict)


@dataclass
class EvaluationReport:
    """Complete evaluation report."""
    policy_name: str
    total_episodes: int
    overall_mean_reward: float
    overall_std_reward: float
    bucket_performance: List[BucketPerformance] = field(default_factory=list)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "policy_name": self.policy_name,
            "total_episodes": self.total_episodes,
            "overall_mean_reward": self.overall_mean_reward,
            "overall_std_reward": self.overall_std_reward,
            "bucket_performance": [
                {
                    "bucket": bp.bucket,
                    "num_episodes": bp.num_episodes,
                    "mean_reward": bp.mean_reward,
                    "std_reward": bp.std_reward,
                    "best_play_id": bp.best_play_id,
                    "play_distribution": bp.play_distribution
                }
                for bp in self.bucket_performance
            ]
        }


def evaluate_policy_detailed(
    env: PlayCallingEnv,
    policy: BasePolicy,
    playbook: List[str],
    num_episodes: int = 100,
    seed: Optional[int] = None
) -> EvaluationReport:
    """
    Detailed policy evaluation with per-bucket analysis.

    Args:
        env: Environment
        policy: Policy to evaluate
        playbook: List of play IDs
        num_episodes: Number of episodes
        seed: Random seed

    Returns:
        EvaluationReport
    """
    all_rewards = []
    bucket_rewards: Dict[str, List[float]] = defaultdict(list)
    bucket_plays: Dict[str, List[str]] = defaultdict(list)


    for episode in range(num_episodes):
        obs, info = env.reset(seed=seed + episode if seed else None)

        # Get bucket from info
        situation_dict = info.get("situation", {})
        # Convert dict back to GameSituation to use shared bucket logic
        situation = GameSituation(**situation_dict)
        bucket = situation.bucket.value

        # Predict action
        action = policy.predict(obs, deterministic=True)
        play_id = playbook[action]

        # Step
        next_obs, reward, terminated, truncated, step_info = env.step(action)

        all_rewards.append(reward)
        bucket_rewards[bucket].append(reward)
        bucket_plays[bucket].append(play_id)

    # Compute overall metrics
    overall_mean = float(np.mean(all_rewards))
    overall_std = float(np.std(all_rewards))

    # Compute per-bucket metrics
    bucket_performance = []
    for bucket, rewards in bucket_rewards.items():
        plays = bucket_plays[bucket]

        # Find most common play
        play_counts = defaultdict(int)
        for play in plays:
            play_counts[play] += 1

        best_play = max(play_counts.items(), key=lambda x: x[1])[0] if play_counts else None

        bucket_performance.append(BucketPerformance(
            bucket=bucket,
            num_episodes=len(rewards),
            mean_reward=float(np.mean(rewards)),
            std_reward=float(np.std(rewards)),
            best_play_id=best_play,
            play_distribution=dict(play_counts)
        ))

    report = EvaluationReport(
        policy_name=policy.__class__.__name__,
        total_episodes=num_episodes,
        overall_mean_reward=overall_mean,
        overall_std_reward=overall_std,
        bucket_performance=bucket_performance
    )

    return report


def compare_policies(
    env: PlayCallingEnv,
    policies: Dict[str, BasePolicy],
    playbook: List[str],
    num_episodes: int = 100,
    seed: Optional[int] = None
) -> Dict[str, EvaluationReport]:
    """
    Compare multiple policies.

    Args:
        env: Environment
        policies: Dict of policy_name -> policy
        playbook: List of play IDs
        num_episodes: Number of episodes per policy
        seed: Random seed

    Returns:
        Dict of policy_name -> EvaluationReport
    """
    reports = {}

    for name, policy in policies.items():
        report = evaluate_policy_detailed(
            env, policy, playbook, num_episodes, seed
        )
        reports[name] = report

    return reports
