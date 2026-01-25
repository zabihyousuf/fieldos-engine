#!/usr/bin/env python3
"""
Train Policy Script - Train the RL system to learn optimal play-calling.

This script uses reinforcement learning to learn which plays work best
in which game situations. After training, you get a "smart playbook"
that can recommend plays based on the current situation.

The key difference from full_report.py:
- full_report.py: Tests plays and shows raw stats (no learning)
- train_policy.py: LEARNS which plays to call when (builds intelligence)

Usage:
    python3 scripts/train_policy.py
    python3 scripts/train_policy.py --steps 5000 --algo BANDIT
    python3 scripts/train_policy.py --steps 10000 --algo UCB --save my_policy.pkl
"""

import sys
import json
import argparse
import pickle
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.core.registry import registry
from fieldos_engine.core.models import Role
from fieldos_engine.rl.env import PlayCallingEnv
from fieldos_engine.rl.train import train_bandit, TrainingConfig
from fieldos_engine.rl.policy import EpsilonGreedyBandit, UpperConfidenceBound
from fieldos_engine.api.main import load_demo_data


# Situation bucket names for display
BUCKET_NAMES = {
    0: "1ST_ANY",
    1: "2ND_SHORT",
    2: "2ND_LONG",
    3: "3RD_SHORT",
    4: "3RD_LONG",
    5: "REDZONE",
    6: "GOALLINE"
}


def train_policy(
    steps: int = 2000,
    algorithm: str = "BANDIT",
    epsilon: float = 0.1,
    learning_rate: float = 0.1,
    ucb_c: float = 2.0,
    seed: int = 42,
    save_path: str = None,
    verbose: bool = True
):
    """
    Train an RL policy to learn optimal play-calling.

    Args:
        steps: Number of training steps (more = better learning)
        algorithm: "BANDIT" (epsilon-greedy) or "UCB" (upper confidence bound)
        epsilon: Exploration rate for BANDIT (0.1 = 10% random exploration)
        learning_rate: How fast to update Q-values (0.1 is good default)
        ucb_c: Exploration constant for UCB (higher = more exploration)
        seed: Random seed for reproducibility
        save_path: Path to save the trained policy
        verbose: Print training progress

    Returns:
        (result, policy) - Training results and the trained policy
    """

    # Load demo data
    if verbose:
        print("Loading demo data...")
    load_demo_data()

    # Get all plays (this is our "playbook" - the actions the RL agent can choose from)
    plays = registry.plays.list()
    play_ids = [p.id for p in plays]

    # Get all scenarios (different defensive looks to train against)
    scenarios = registry.scenarios.list()
    scenario_ids = [s.id for s in scenarios]

    if verbose:
        print(f"\nPlaybook ({len(play_ids)} plays):")
        for i, play in enumerate(plays):
            print(f"  Action {i}: {play.name}")

        print(f"\nScenarios ({len(scenario_ids)} defensive looks):")
        for scenario in scenarios:
            print(f"  - {scenario.name} ({scenario.defense_call.type.value} {scenario.defense_call.shell.value})")

    # Get player IDs
    all_players = registry.players.list()
    off_players = {
        Role.QB: next(p.id for p in all_players if p.role == Role.QB),
        Role.CENTER: next(p.id for p in all_players if p.role == Role.CENTER),
        Role.WR1: next(p.id for p in all_players if p.role == Role.WR1),
        Role.WR2: next(p.id for p in all_players if p.role == Role.WR2),
        Role.WR3: next(p.id for p in all_players if p.role == Role.WR3),
    }
    # Use new D1-D5 defensive roles
    def_players = {
        Role.D1: next(p.id for p in all_players if p.role == Role.D1),
        Role.D2: next(p.id for p in all_players if p.role == Role.D2),
        Role.D3: next(p.id for p in all_players if p.role == Role.D3),
        Role.D4: next(p.id for p in all_players if p.role == Role.D4),
        Role.D5: next(p.id for p in all_players if p.role == Role.D5),
    }

    # Create the RL environment
    if verbose:
        print(f"\nCreating RL environment...")
    env = PlayCallingEnv(
        playbook=play_ids,
        scenarios=scenario_ids,
        offensive_players=off_players,
        defensive_players=def_players,
        seed=seed
    )

    # Training configuration
    config = TrainingConfig(
        total_steps=steps,
        eval_frequency=max(100, steps // 10),  # Evaluate 10 times during training
        eval_episodes=50,
        algorithm=algorithm,
        epsilon=epsilon,
        learning_rate=learning_rate,
        ucb_c=ucb_c,
        seed=seed
    )

    if verbose:
        print(f"\n{'='*60}")
        print("TRAINING CONFIGURATION")
        print(f"{'='*60}")
        print(f"  Algorithm: {algorithm}")
        print(f"  Training Steps: {steps}")
        if algorithm == "BANDIT":
            print(f"  Epsilon (exploration): {epsilon}")
            print(f"  Learning Rate: {learning_rate}")
        else:  # UCB
            print(f"  UCB C (exploration): {ucb_c}")
        print(f"  Seed: {seed}")
        print()

    # Train!
    if verbose:
        print("Starting training...")
        print("(The system is learning which plays work best in each situation)")
        print()

    result, policy = train_bandit(env, config)

    if verbose:
        print(f"\n{'='*60}")
        print("TRAINING COMPLETE")
        print(f"{'='*60}")
        print(f"  Total Steps: {result.total_steps}")
        print(f"  Final Avg Reward: {result.final_reward_mean:.2f}")
        print(f"  Reward Std Dev: {result.final_reward_std:.2f}")

        # Show what was learned
        print(f"\n{'='*60}")
        print("LEARNED PLAY-CALLING STRATEGY")
        print(f"{'='*60}")
        print("\nThe system learned which play to call in each situation:\n")

        best_actions = result.best_actions_per_bucket
        print(f"{'Situation':<20} {'Recommended Play':<35} {'Play ID'}")
        print("-" * 70)

        for bucket_idx in sorted(best_actions.keys()):
            action_idx = best_actions[bucket_idx]
            bucket_name = BUCKET_NAMES.get(bucket_idx, f"Bucket {bucket_idx}")
            play_name = plays[action_idx].name
            play_id = plays[action_idx].id
            print(f"{bucket_name:<20} {play_name:<35} {play_id}")

        # Show Q-values if available
        if hasattr(policy, 'get_q_table'):
            print(f"\n{'='*60}")
            print("Q-VALUE TABLE (Expected Rewards)")
            print(f"{'='*60}")
            print("\nHigher values = better expected performance\n")

            q_table = policy.get_q_table()

            # Header
            header = "Situation".ljust(15)
            for i, play in enumerate(plays):
                short_name = play.name[:12]
                header += f"{short_name:>14}"
            print(header)
            print("-" * (15 + 14 * len(plays)))

            for bucket_idx in sorted(q_table.keys()):
                q_vals = q_table[bucket_idx]
                bucket_name = BUCKET_NAMES.get(bucket_idx, f"Bucket {bucket_idx}")
                row = bucket_name[:14].ljust(15)
                for q in q_vals:
                    row += f"{q:>14.2f}"
                print(row)

    # Save policy if requested
    if save_path:
        policy_path = Path(save_path)
        policy.save(str(policy_path))
        if verbose:
            print(f"\nPolicy saved to: {policy_path}")

        # Also save metadata
        meta_path = policy_path.with_suffix('.json')
        with open(meta_path, 'w') as f:
            json.dump({
                'algorithm': algorithm,
                'steps': steps,
                'epsilon': epsilon,
                'learning_rate': learning_rate,
                'ucb_c': ucb_c,
                'seed': seed,
                'final_reward_mean': result.final_reward_mean,
                'final_reward_std': result.final_reward_std,
                'playbook': play_ids,
                'best_actions': {
                    BUCKET_NAMES.get(k, f"bucket_{k}"): {
                        'action_idx': v,
                        'play_id': play_ids[v],
                        'play_name': plays[v].name
                    }
                    for k, v in result.best_actions_per_bucket.items()
                }
            }, f, indent=2)
        if verbose:
            print(f"Metadata saved to: {meta_path}")

    if verbose:
        print(f"\n{'='*60}")
        print("HOW TO USE YOUR TRAINED POLICY")
        print(f"{'='*60}")
        print("""
To get play recommendations from your trained policy:

    from fieldos_engine.rl.policy import EpsilonGreedyBandit
    import numpy as np

    # Load the policy
    policy = EpsilonGreedyBandit(num_actions=6)
    policy.load("path/to/policy.pkl")

    # Create observation for 3rd & Short
    obs = np.zeros(15, dtype=np.float32)
    obs[3] = 1.0  # 3RD_SHORT bucket

    # Get recommendation
    action = policy.predict(obs, deterministic=True)
    print(f"Recommended play index: {action}")
""")

    return result, policy


def get_recommendation(policy, situation: str, playbook: list) -> str:
    """
    Get a play recommendation from a trained policy.

    Args:
        policy: Trained policy object
        situation: One of "1ST_ANY", "2ND_SHORT", "2ND_LONG", "3RD_SHORT",
                   "3RD_LONG", "REDZONE", "GOALLINE"
        playbook: List of play objects

    Returns:
        Recommended play name
    """
    import numpy as np

    bucket_map = {
        "1ST_ANY": 0, "2ND_SHORT": 1, "2ND_LONG": 2,
        "3RD_SHORT": 3, "3RD_LONG": 4, "REDZONE": 5, "GOALLINE": 6
    }

    obs = np.zeros(15, dtype=np.float32)
    obs[bucket_map[situation]] = 1.0

    action = policy.predict(obs, deterministic=True)
    return playbook[action].name


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train RL policy for optimal play-calling",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/train_policy.py                          # Quick training (2000 steps)
  python scripts/train_policy.py --steps 10000            # Longer training
  python scripts/train_policy.py --algo UCB               # Use UCB algorithm
  python scripts/train_policy.py --save my_policy.pkl     # Save the trained policy

The trained policy learns which plays work best in different game situations.
After training, you can use the policy to get play recommendations.
        """
    )
    parser.add_argument("--steps", type=int, default=2000,
                        help="Training steps (default: 2000, more = better)")
    parser.add_argument("--algo", type=str, default="BANDIT", choices=["BANDIT", "UCB"],
                        help="Algorithm: BANDIT (epsilon-greedy) or UCB (default: BANDIT)")
    parser.add_argument("--epsilon", type=float, default=0.1,
                        help="Exploration rate for BANDIT (default: 0.1)")
    parser.add_argument("--lr", type=float, default=0.1,
                        help="Learning rate (default: 0.1)")
    parser.add_argument("--ucb-c", type=float, default=2.0,
                        help="UCB exploration constant (default: 2.0)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--save", type=str, default="models/policy.pkl",
                        help="Path to save trained policy (default: models/policy.pkl)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress verbose output")

    args = parser.parse_args()

    train_policy(
        steps=args.steps,
        algorithm=args.algo,
        epsilon=args.epsilon,
        learning_rate=args.lr,
        ucb_c=args.ucb_c,
        seed=args.seed,
        save_path=args.save,
        verbose=not args.quiet
    )
