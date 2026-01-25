#!/usr/bin/env python3
"""
Full Report Script - Test all plays against all scenarios.

This script runs every play against every defensive scenario and generates
a comprehensive performance report. This is pure testing/analysis - no learning.

Usage:
    python scripts/full_report.py
    python scripts/full_report.py --episodes 100
    python scripts/full_report.py --output report.json
"""

import sys
import json
import argparse
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.core.registry import registry
from fieldos_engine.core.models import GameSituation, Role
from fieldos_engine.sim.engine import SimulationEngine
from fieldos_engine.sim.metrics import MetricsCollector
from fieldos_engine.api.main import load_demo_data


def run_full_report(episodes_per_matchup: int = 50, seed: int = 42, output_file: str = None):
    """Run all plays against all scenarios and generate report."""

    # Load demo data
    print("Loading demo data...")
    load_demo_data()

    # Get all plays and scenarios
    plays = registry.plays.list()
    scenarios = registry.scenarios.list()

    print(f"Found {len(plays)} plays and {len(scenarios)} scenarios")
    print(f"Running {len(plays) * len(scenarios)} matchups with {episodes_per_matchup} episodes each")
    print(f"Total simulations: {len(plays) * len(scenarios) * episodes_per_matchup}")
    print()

    # Get default players
    off_roles = [Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3]
    def_roles = [Role.RUSHER, Role.CB1, Role.CB2, Role.SAFETY, Role.LB]
    all_players = registry.players.list()

    off_players = {role: next(p for p in all_players if p.role == role) for role in off_roles}
    def_players = {role: next(p for p in all_players if p.role == role) for role in def_roles}

    # Situations to sample from
    situations = [
        GameSituation(down=1, yards_to_gain=25.0, yardline_to_goal=40.0),
        GameSituation(down=2, yards_to_gain=10.0, yardline_to_goal=35.0),
        GameSituation(down=3, yards_to_gain=5.0, yardline_to_goal=30.0),
        GameSituation(down=3, yards_to_gain=10.0, yardline_to_goal=25.0),
    ]

    engine = SimulationEngine(seed=seed)
    results = []

    # Header
    print("=" * 100)
    print("FIELDOS ENGINE - FULL PLAY ANALYSIS REPORT")
    print("=" * 100)
    print()

    for play in plays:
        print(f"\n{'='*80}")
        print(f"PLAY: {play.name}")
        print(f"Formation: {play.formation.name}")
        print(f"QB Progression: {' → '.join(r.value for r in play.qb_plan.progression_roles)}")
        print(f"{'='*80}")
        print()
        print(f"{'Scenario':<35} {'Comp%':>8} {'Sack%':>8} {'Yards':>8} {'P50':>8} {'P90':>8}")
        print("-" * 80)

        play_results = []

        for scenario in scenarios:
            collector = MetricsCollector()

            for episode in range(episodes_per_matchup):
                situation = situations[episode % len(situations)]
                outcome, _ = engine.simulate_play(play, scenario, off_players, def_players)
                collector.record(play.id, outcome, situation)

            metrics = collector.get_metrics()
            overall = metrics.overall

            result = {
                'play_id': play.id,
                'play_name': play.name,
                'scenario_id': scenario.id,
                'scenario_name': scenario.name,
                'defense_type': scenario.defense_call.type.value,
                'defense_shell': scenario.defense_call.shell.value,
                'rushers': scenario.defense_call.rushers_count,
                'episodes': episodes_per_matchup,
                'completion_rate': overall.completion_rate,
                'sack_rate': overall.sack_rate,
                'intercept_rate': overall.intercept_rate,
                'yards_mean': overall.yards_mean,
                'yards_p50': overall.yards_p50,
                'yards_p90': overall.yards_p90,
                'time_to_throw_mean': overall.time_to_throw_mean,
                'target_distribution': {k.value: v for k, v in overall.target_counts.items()},
                'failure_modes': {k.value: v for k, v in overall.failure_mode_counts.items()}
            }
            results.append(result)
            play_results.append(result)

            # Print row
            print(f"{scenario.name:<35} {overall.completion_rate:>7.0%} {overall.sack_rate:>7.0%} "
                  f"{overall.yards_mean:>7.1f} {overall.yards_p50:>7.1f} {overall.yards_p90:>7.1f}")

        # Best scenario for this play
        best = max(play_results, key=lambda x: x['yards_mean'])
        worst = min(play_results, key=lambda x: x['yards_mean'])
        print()
        print(f"  Best matchup:  {best['scenario_name']} ({best['yards_mean']:.1f} yards, {best['completion_rate']:.0%} comp)")
        print(f"  Worst matchup: {worst['scenario_name']} ({worst['yards_mean']:.1f} yards, {worst['completion_rate']:.0%} comp)")

    # Summary section
    print("\n" + "=" * 100)
    print("SUMMARY: BEST PLAYS BY SCENARIO")
    print("=" * 100)

    for scenario in scenarios:
        scenario_results = [r for r in results if r['scenario_id'] == scenario.id]
        best = max(scenario_results, key=lambda x: x['yards_mean'])

        print(f"\n{scenario.name} ({scenario.defense_call.type.value} {scenario.defense_call.shell.value}):")
        print(f"  Best Play: {best['play_name']}")
        print(f"  Expected Yards: {best['yards_mean']:.1f}")
        print(f"  Completion Rate: {best['completion_rate']:.0%}")

        # Show top 3
        sorted_results = sorted(scenario_results, key=lambda x: x['yards_mean'], reverse=True)
        print(f"  Rankings:")
        for i, r in enumerate(sorted_results[:3], 1):
            print(f"    {i}. {r['play_name']}: {r['yards_mean']:.1f} yards ({r['completion_rate']:.0%})")

    # Overall best plays
    print("\n" + "=" * 100)
    print("OVERALL PLAY RANKINGS (averaged across all scenarios)")
    print("=" * 100)

    play_averages = {}
    for play in plays:
        play_results = [r for r in results if r['play_id'] == play.id]
        avg_yards = sum(r['yards_mean'] for r in play_results) / len(play_results)
        avg_comp = sum(r['completion_rate'] for r in play_results) / len(play_results)
        play_averages[play.id] = {
            'name': play.name,
            'avg_yards': avg_yards,
            'avg_completion': avg_comp
        }

    sorted_plays = sorted(play_averages.items(), key=lambda x: x[1]['avg_yards'], reverse=True)
    print()
    print(f"{'Rank':<6} {'Play':<30} {'Avg Yards':>12} {'Avg Comp%':>12}")
    print("-" * 65)
    for i, (play_id, stats) in enumerate(sorted_plays, 1):
        print(f"{i:<6} {stats['name']:<30} {stats['avg_yards']:>11.1f} {stats['avg_completion']:>11.0%}")

    # Save to file if requested
    if output_file:
        output_path = Path(output_file)
        with open(output_path, 'w') as f:
            json.dump({
                'metadata': {
                    'episodes_per_matchup': episodes_per_matchup,
                    'seed': seed,
                    'num_plays': len(plays),
                    'num_scenarios': len(scenarios),
                    'total_simulations': len(plays) * len(scenarios) * episodes_per_matchup
                },
                'results': results,
                'play_averages': play_averages,
                'best_by_scenario': {
                    s.id: max([r for r in results if r['scenario_id'] == s.id],
                              key=lambda x: x['yards_mean'])['play_id']
                    for s in scenarios
                }
            }, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 100)
    print("REPORT COMPLETE")
    print("=" * 100)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate full play analysis report")
    parser.add_argument("--episodes", type=int, default=50,
                        help="Episodes per play/scenario matchup (default: 50)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON file path (optional)")

    args = parser.parse_args()

    run_full_report(
        episodes_per_matchup=args.episodes,
        seed=args.seed,
        output_file=args.output
    )
