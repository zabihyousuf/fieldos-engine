#!/usr/bin/env python3
"""
Train Policy Script - Train the RL system to learn optimal play-calling.

This script uses reinforcement learning to learn which plays work best
in which game situations. After training, you get a "smart playbook"
that can recommend plays based on the current situation.

Saves detailed training results to training_results/ folder showing:
- What play was run
- When it worked (situation/scenario)
- Who the target was
- The outcome and yards gained

Usage:
    python3 scripts/train_policy.py
    python3 scripts/train_policy.py --steps 5000 --algo BANDIT
    python3 scripts/train_policy.py --steps 10000 --algo UCB --save my_policy.pkl
"""

import sys
import json
import argparse
import pickle
import csv
from pathlib import Path
from datetime import datetime

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
    use_generated_plays: bool = True,
    num_generated_plays: int = 20,
    visualize: bool = True,
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
        use_generated_plays: If True, generate additional random plays for exploration
        num_generated_plays: Number of random plays to generate (default 20)
        visualize: If True, generate visualizations of top plays after training
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
        print(f"\nBase Playbook ({len(play_ids)} plays):")
        for i, play in enumerate(plays):
            print(f"  Action {i}: {play.name}")
        
        if use_generated_plays:
            print(f"\n+ {num_generated_plays} generated plays will be added for exploration")

        print(f"\nScenarios ({len(scenario_ids)} defensive looks):")
        for scenario in scenarios:
            print(f"  - {scenario.name} ({scenario.defense_call.type.value} {scenario.defense_call.shell.value})")
        
        print(f"\n📊 Situation-aware defense: ENABLED (coverage adjusts to down/distance)")

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

    # Create the RL environment with generated plays
    if verbose:
        print(f"\nCreating RL environment...")
    env = PlayCallingEnv(
        playbook=play_ids,
        scenarios=scenario_ids,
        offensive_players=off_players,
        defensive_players=def_players,
        use_generated_plays=use_generated_plays,
        num_generated_plays=num_generated_plays,
        seed=seed
    )
    
    # Update plays list to include generated plays if enabled
    if use_generated_plays:
        # Get all plays from the env's expanded playbook
        all_play_ids = env.playbook if hasattr(env, 'playbook') else play_ids
        if verbose:
            print(f"  Total plays (base + generated): {env.action_space.n}")

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

    # Train with detailed tracking
    if verbose:
        print("Starting training...")
        print("(The system is learning which plays work best in each situation)")
        print()

    # Generate timestamp for this training run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create a single folder for this training run (contains CSV, JSON, and visualizations)
    run_dir = Path("training_runs") / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"📁 Training outputs will be saved to: {run_dir}")
        print()
    
    # Track detailed results for each step
    detailed_results = []
    
    # Custom training loop with detailed tracking
    if config.algorithm == "UCB":
        policy = UpperConfidenceBound(
            num_actions=env.action_space.n,
            c=config.ucb_c,
            seed=config.seed
        )
    else:
        policy = EpsilonGreedyBandit(
            num_actions=env.action_space.n,
            epsilon=config.epsilon,
            learning_rate=config.learning_rate,
            seed=config.seed
        )
    
    reward_history = []
    
    for step in range(steps):
        obs, info = env.reset()
        action = policy.predict(obs, deterministic=False)
        next_obs, reward, terminated, truncated, step_info = env.step(action)
        policy.update(obs, action, reward)
        reward_history.append(reward)
        
        # Extract detailed info for this step
        outcome_data = step_info.get('outcome', {})
        situation_data = step_info.get('situation', {})
        
        detailed_results.append({
            'step': step,
            'play_id': step_info.get('play_id', ''),
            'play_name': plays[action].name if action < len(plays) else f'action_{action}',
            'situation': BUCKET_NAMES.get(situation_data.get('bucket_idx', -1), situation_data.get('bucket', '')),
            'down': situation_data.get('down', ''),
            'yards_to_gain': situation_data.get('yards_to_gain', ''),
            'scenario_id': info.get('scenario_id', ''),
            'outcome': outcome_data.get('outcome', ''),
            'yards_gained': outcome_data.get('yards_gained', 0),
            'target_role': outcome_data.get('target_role', ''),
            'time_to_throw_ms': outcome_data.get('time_to_throw_ms', ''),
            'completion_probability': outcome_data.get('completion_probability', ''),
            'reward': reward,
            'is_generated_play': step_info.get('is_generated', False)
        })
        
        # Print progress every 500 steps
        if verbose and (step + 1) % 500 == 0:
            avg_reward = sum(reward_history[-100:]) / min(100, len(reward_history))
            print(f"  Step {step + 1}/{steps} - Avg reward (last 100): {avg_reward:.2f}")
    
    # Save detailed results to CSV
    csv_path = run_dir / "results.csv"
    if detailed_results:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=detailed_results[0].keys())
            writer.writeheader()
            writer.writerows(detailed_results)
        if verbose:
            print(f"\nDetailed results saved to: {csv_path}")
    
    # Also save a summary JSON
    summary_path = run_dir / "summary.json"
    
    # Compute best actions per bucket
    best_actions = policy.get_best_actions_per_bucket()
    
    # Calculate per-play and per-situation stats
    play_stats = {}
    situation_stats = {}
    for r in detailed_results:
        play_name = r['play_name']
        if play_name not in play_stats:
            play_stats[play_name] = {'total': 0, 'completions': 0, 'yards': 0, 'rewards': 0}
        play_stats[play_name]['total'] += 1
        if r['outcome'] == 'COMPLETE':
            play_stats[play_name]['completions'] += 1
        play_stats[play_name]['yards'] += r['yards_gained'] if isinstance(r['yards_gained'], (int, float)) else 0
        play_stats[play_name]['rewards'] += r['reward']
        
        situation = r['situation']
        if situation and situation not in situation_stats:
            situation_stats[situation] = {'total': 0, 'completions': 0, 'yards': 0}
        if situation:
            situation_stats[situation]['total'] += 1
            if r['outcome'] == 'COMPLETE':
                situation_stats[situation]['completions'] += 1
            situation_stats[situation]['yards'] += r['yards_gained'] if isinstance(r['yards_gained'], (int, float)) else 0
    
    # Compute averages
    for play_name, stats in play_stats.items():
        if stats['total'] > 0:
            stats['avg_yards'] = stats['yards'] / stats['total']
            stats['avg_reward'] = stats['rewards'] / stats['total']
            stats['completion_rate'] = stats['completions'] / stats['total']
    
    summary = {
        'timestamp': timestamp,
        'algorithm': algorithm,
        'total_steps': steps,
        'final_avg_reward': sum(reward_history[-100:]) / min(100, len(reward_history)),
        'best_plays_by_situation': {
            BUCKET_NAMES.get(k, f"bucket_{k}"): {
                'play_name': plays[v].name if v < len(plays) else f'action_{v}',
                'play_id': plays[v].id if v < len(plays) else f'action_{v}'
            }
            for k, v in best_actions.items()
        },
        'play_performance': {
            name: {
                'attempts': s['total'],
                'completion_rate': round(s.get('completion_rate', 0), 3),
                'avg_yards': round(s.get('avg_yards', 0), 2),
                'avg_reward': round(s.get('avg_reward', 0), 2)
            }
            for name, s in sorted(play_stats.items(), key=lambda x: x[1].get('avg_reward', 0), reverse=True)
        },
        'situation_performance': situation_stats
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    if verbose:
        print(f"Training summary saved to: {summary_path}")
    
    # Create a simple result object for compatibility
    from dataclasses import dataclass
    @dataclass
    class TrainingResultCompat:
        total_steps: int
        final_reward_mean: float
        final_reward_std: float
        best_actions_per_bucket: dict
        algorithm: str
        reward_history: list
    
    import numpy as np
    result = TrainingResultCompat(
        total_steps=steps,
        final_reward_mean=float(np.mean(reward_history[-100:])),
        final_reward_std=float(np.std(reward_history[-100:])),
        best_actions_per_bucket=best_actions,
        algorithm=algorithm,
        reward_history=reward_history
    )

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
            # Handle both base and generated plays
            if action_idx < len(plays):
                play_name = plays[action_idx].name
                play_id = plays[action_idx].id
            else:
                # Generated play - get from env's playbook
                play_id = env.playbook[action_idx] if hasattr(env, 'playbook') and action_idx < len(env.playbook) else f"gen_play_{action_idx}"
                play_name = f"Generated Play #{action_idx - len(plays) + 1}"
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
        
        # Show top plays by performance
        print(f"\n{'='*60}")
        print("TOP PLAYS BY PERFORMANCE")
        print(f"{'='*60}")
        print(f"\n{'Play Name':<30} {'Attempts':>10} {'Comp%':>10} {'Avg Yards':>12} {'Avg Reward':>12}")
        print("-" * 74)
        for name, stats in list(summary['play_performance'].items())[:10]:
            print(f"{name:<30} {stats['attempts']:>10} {stats['completion_rate']*100:>9.1f}% {stats['avg_yards']:>12.2f} {stats['avg_reward']:>12.2f}")

    # Save policy if requested
    if save_path:
        policy_path = Path(save_path)
        policy_path.parent.mkdir(parents=True, exist_ok=True)
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
                'playbook': play_ids + (env.playbook[len(play_ids):] if hasattr(env, 'playbook') else []),
                'best_actions': {
                    BUCKET_NAMES.get(k, f"bucket_{k}"): {
                        'action_idx': v,
                        'play_id': play_ids[v] if v < len(play_ids) else f"gen_play_{v}",
                        'play_name': plays[v].name if v < len(plays) else f"Generated Play #{v - len(plays) + 1}"
                    }
                    for k, v in result.best_actions_per_bucket.items()
                },
                'results_csv': str(csv_path),
                'results_summary': str(summary_path)
            }, f, indent=2)
        if verbose:
            print(f"Metadata saved to: {meta_path}")

    if verbose:
        print(f"\n{'='*60}")
        print("TRAINING RESULTS")
        print(f"{'='*60}")
        print(f"""
All outputs saved to: {run_dir}
  ├── results.csv       (every play attempt with target, outcome, yards)
  ├── summary.json      (aggregated stats and best plays)
  └── visualizations/   (PNG diagrams of top plays)

To analyze results:
  import pandas as pd
  df = pd.read_csv('{run_dir}/results.csv')
  
  # See what plays worked best
  df.groupby('play_name')['yards_gained'].mean().sort_values(ascending=False)
""")

    # Generate visualizations if requested
    if visualize:
        try:
            from fieldos_engine.utils.viz import visualize_play_static
            from fieldos_engine.sim.engine import SimulationEngine
            from fieldos_engine.sim.coverage import CoverageAssignment
            import numpy as np
            
            # Create visualizations subfolder in this training run
            viz_dir = run_dir / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            if verbose:
                print(f"\n{'='*60}")
                print("GENERATING VISUALIZATIONS")
                print(f"{'='*60}")
            
            # Get top 5 plays by performance
            top_plays = list(summary['play_performance'].items())[:5]
            
            # Get offensive and defensive player objects
            off_player_objs = {role: registry.players.get(pid) for role, pid in off_players.items()}
            def_player_objs = {role: registry.players.get(pid) for role, pid in def_players.items()}
            
            engine = SimulationEngine(seed=seed)
            
            for play_name, stats in top_plays:
                # Find the play object
                play = next((p for p in plays if p.name == play_name), None)
                if not play:
                    continue
                
                # Analyze training data to find best target and scenario for this play
                play_results = [r for r in detailed_results if r['play_name'] == play_name]
                
                # Find most common successful target
                target_counts = {}
                for r in play_results:
                    if r['outcome'] == 'COMPLETE' and r['target_role']:
                        target = r['target_role']
                        target_counts[target] = target_counts.get(target, 0) + 1
                best_target = max(target_counts, key=target_counts.get) if target_counts else None
                
                # Find best scenario (highest avg yards)
                scenario_yards = {}
                scenario_counts = {}
                for r in play_results:
                    scen_id = r['scenario_id']
                    if scen_id:
                        if scen_id not in scenario_yards:
                            scenario_yards[scen_id] = 0
                            scenario_counts[scen_id] = 0
                        scenario_yards[scen_id] += r['yards_gained'] if isinstance(r['yards_gained'], (int, float)) else 0
                        scenario_counts[scen_id] += 1
                
                best_scenario_id = None
                best_scenario_avg = 0
                for scen_id, total_yards in scenario_yards.items():
                    avg = total_yards / scenario_counts[scen_id] if scenario_counts[scen_id] > 0 else 0
                    if avg > best_scenario_avg:
                        best_scenario_avg = avg
                        best_scenario_id = scen_id
                
                # Get scenario object for best scenario
                best_scenario_obj = next((s for s in scenarios if s.id == best_scenario_id), scenarios[0]) if best_scenario_id else scenarios[0]
                
                if verbose:
                    print(f"\n  Visualizing: {play_name}")
                    print(f"    vs {best_scenario_obj.name}")
                    if best_target:
                        print(f"    Best target: {best_target}")
                
                # Get coverage assignments with receiver positions for proper matching
                rng = np.random.default_rng(seed)
                
                # Extract receiver positions from the play's formation
                receiver_positions = {
                    slot.role: slot.position
                    for slot in play.formation.slots
                }
                
                coverage = CoverageAssignment(
                    best_scenario_obj.defense_call,
                    best_scenario_obj.defender_start_positions,
                    list(play.qb_plan.progression_roles),
                    rng=rng,
                    receiver_positions=receiver_positions
                )
                
                # Generate filename with play name
                safe_name = play_name.replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")[:40]
                static_path = viz_dir / f"{safe_name}.png"
                
                # Build info for visualization
                target_info_str = best_target if best_target else "1st read"
                best_scenario_info_str = best_scenario_obj.name
                stats_dict = {
                    'completion_rate': stats.get('completion_rate', 0),
                    'avg_yards': stats.get('avg_yards', 0),
                    'attempts': stats.get('attempts', 0)
                }
                
                visualize_play_static(
                    play, best_scenario_obj, str(static_path), coverage.assignments,
                    target_info=target_info_str,
                    best_scenario_info=best_scenario_info_str,
                    stats_info=stats_dict
                )
                if verbose:
                    print(f"    ✓ Saved: {static_path.name}")
            
            if verbose:
                print(f"\n  Visualizations saved to: {viz_dir}")
        
        except ImportError as e:
            if verbose:
                print(f"\n  ⚠ Could not generate visualizations: {e}")
                print("    Install matplotlib: pip install matplotlib")
        except Exception as e:
            if verbose:
                print(f"\n  ⚠ Visualization error: {e}")

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
  python scripts/train_policy.py --no-generated           # Use only base plays
  python scripts/train_policy.py --visualize              # Generate play visualizations

The trained policy learns which plays work best in different game situations.
Uses situation-aware defense (coverage adjusts to down/distance) and generated plays.
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
    parser.add_argument("--no-generated", action="store_true",
                        help="Disable generated plays (use only base playbook)")
    parser.add_argument("--num-generated", type=int, default=20,
                        help="Number of generated plays to add (default: 20)")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate visualizations of top plays")
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
        use_generated_plays=not args.no_generated,
        num_generated_plays=args.num_generated,
        visualize=args.visualize,
        verbose=not args.quiet
    )

