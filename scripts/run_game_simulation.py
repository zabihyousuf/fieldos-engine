#!/usr/bin/env python3
"""
Run a game simulation between two teams.

Demonstrates the full game simulation system including:
- Team setup with dual-role players
- Drive-by-drive gameplay
- Detailed statistics tracking
- Comprehensive game reports
"""

import sys
from pathlib import Path
from datetime import datetime
import logging
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.core.models import (
    Team, GamePlayer, DualRolePlayerAttributes, GameConfig
)
from fieldos_engine.api.main import load_demo_data
from fieldos_engine.sim.game_simulator import GameSimulator, create_sample_teams
from fieldos_engine.stats.report_generator import generate_game_report, GameReportGenerator

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
GAMES_DIR = RESULTS_DIR / "games"
REPORTS_DIR = RESULTS_DIR / "reports"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("game_simulation")


def create_custom_teams():
    """Create two custom teams with specific playbooks."""

    team1_players = [
        GamePlayer(
            id="bb_qb", name="Zabih YousUf", number=1,
            attributes=DualRolePlayerAttributes(
                off_speed=80,
                off_acceleration=82,
                off_agility=83,
                off_hands=80,
                off_route_running=85,
                throw_power=89,
                short_acc=88,
                mid_acc=80,
                deep_acc=83,
                release_time_ms=380,
                decision_latency_ms=320,
                def_speed=82,
                def_acceleration=85,
                man_coverage=85,
                zone_coverage=82,
                ball_skills=85,
                pass_rush=75
            )
        ),
        GamePlayer(
            id="bb_c", name="Jaylen", number=23,
            attributes=DualRolePlayerAttributes(
                off_speed=82,
                off_acceleration=85,
                off_agility=80,
                off_hands=82,
                off_route_running=85,
                throw_power=45,
                short_acc=48,
                mid_acc=40,
                deep_acc=35,
                def_speed=82,
                def_acceleration=85,
                man_coverage=85,
                zone_coverage=82,
                ball_skills=85,
                pass_rush=75
            )
        ),
        GamePlayer(
            id="bb_wr1", name="Soh", number=11,
            attributes=DualRolePlayerAttributes(
                off_speed=85,
                off_acceleration=88,
                off_agility=85,
                off_hands=99,
                off_route_running=90,
                throw_power=30,
                short_acc=32,
                mid_acc=25,
                deep_acc=20,
                def_speed=85,
                def_acceleration=88,
                man_coverage=85,
                zone_coverage=82,
                ball_skills=85,
                pass_rush=85
            )
        ),
        GamePlayer(
            id="bb_wr2", name="Maddox", number=82,
            attributes=DualRolePlayerAttributes(
                off_speed=90,
                off_acceleration=88,
                off_agility=89,
                off_hands=99,
                off_route_running=95,
                throw_power=30,
                short_acc=32,
                mid_acc=25,
                deep_acc=20,
                def_speed=90,
                def_acceleration=95,
                man_coverage=95,
                zone_coverage=92,
                ball_skills=95,
                pass_rush=95
            )
        ),
        GamePlayer(
            id="bb_wr3", name="Mikail", number=2,
            attributes=DualRolePlayerAttributes(
                off_speed=83,
                off_acceleration=82,
                off_agility=80,
                off_hands=75,
                off_route_running=80,
                throw_power=25,
                short_acc=48,
                mid_acc=40,
                deep_acc=35,
                def_speed=84,
                def_acceleration=82,
                man_coverage=85,
                zone_coverage=82,
                ball_skills=85,
                pass_rush=85
            )
        ),
    ]

    team1 = Team(
        id="butterfingers",
        name="Butterfingers",
        players=team1_players,
        playbook=[
            "play_trips_flood",
            "play_bunch_slants",
            "play_bunch_wheel",
            "play_bunch_stick",
            "play_twins_smash",
            "play_jet_motion_sweep",
            "play_shovel_option",
        ]
    )

    team2_players = [
        GamePlayer(
            id="sd_qb", name="Danny", number=22,
            attributes=DualRolePlayerAttributes(
                off_speed=80,
                off_acceleration=82,
                off_agility=89,
                off_hands=85,
                off_route_running=85,
                throw_power=89,
                short_acc=90,
                mid_acc=85,
                deep_acc=85,
                release_time_ms=310,
                decision_latency_ms=280,
                def_speed=79,
                def_acceleration=82,
                man_coverage=82,
                zone_coverage=85,
                ball_skills=85,
                pass_rush=85
            )
        ),
        GamePlayer(
            id="sd_c", name="Lincoln", number=55,
            attributes=DualRolePlayerAttributes(
                off_speed=60,
                off_acceleration=62,
                off_agility=60,
                off_hands=65,
                off_route_running=60,
                throw_power=99,
                short_acc=92,
                mid_acc=90,
                deep_acc=95,
                def_speed=60,
                def_acceleration=62,
                man_coverage=65,
                zone_coverage=62,
                ball_skills=55,
                pass_rush=55
            )
        ),
        GamePlayer(
            id="sd_wr1", name="Nick", number=1,
            attributes=DualRolePlayerAttributes(
                off_speed=95,
                off_acceleration=92,
                off_agility=88,
                off_hands=99,
                off_route_running=95,
                throw_power=72,
                short_acc=75,
                mid_acc=68,
                deep_acc=62,
                def_speed=92,
                def_acceleration=90,
                man_coverage=85,
                zone_coverage=82,
                ball_skills=85,
                pass_rush=67
            )
        ),
        GamePlayer(
            id="sd_wr2", name="Terrance", number=88,
            attributes=DualRolePlayerAttributes(
                off_speed=92,
                off_acceleration=90,
                off_agility=88,
                off_hands=92,
                off_route_running=95,
                throw_power=32,
                short_acc=25,
                mid_acc=15,
                deep_acc=10,
                def_speed=92,
                def_acceleration=90,
                man_coverage=85,
                zone_coverage=82,
                ball_skills=85,
                pass_rush=85
            )
        ),
        GamePlayer(
            id="sd_wr3", name="Mike", number=17,
            attributes=DualRolePlayerAttributes(
                off_speed=85,
                off_acceleration=82,
                off_agility=80,
                off_hands=25,
                off_route_running=30,
                throw_power=18,
                short_acc=10,
                mid_acc=5,
                deep_acc=10,
                def_speed=86,
                def_acceleration=89,
                man_coverage=89,
                zone_coverage=86,
                ball_skills=89,
                pass_rush=60
            )
        ),
    ]

    team2 = Team(
        id="godbods",
        name="Godbods",
        players=team2_players,
        playbook=[
            "play_spread_vertical",
            "play_bunch_mesh",
            "play_bunch_scissors",
            "play_orbit_screen",
            "play_reverse_pass",
            "play_halfback_pass",
            "play_center_throwback",
        ]
    )

    return team1, team2


def run_single_game(team1: Team, team2: Team, seed: int = 42, save_results: bool = True):
    """Run a single game and generate report."""

    print("\n" + "=" * 60)
    print("GAME SIMULATION")
    print("=" * 60)
    print(f"\n{team1.name} vs {team2.name}")
    print(f"Seed: {seed}")

    # Create game config
    config = GameConfig(
        field_length=60.0,
        endzone_depth=7.0,
        field_width=25.0,
        playing_field=46.0,
        downs_to_first_down=3,
        downs_to_score=3,
        first_down_yards=20.0,
        drives_per_team=10,
        td_points=6,
        start_position=20.0
    )

    # Run simulation
    simulator = GameSimulator(team1, team2, config, seed=seed)
    result = simulator.simulate_game()

    # Generate reports
    text_report = generate_game_report(result, format="text")
    json_report = generate_game_report(result, format="json")
    print(text_report)

    # Save results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        game_name = f"{team1.id}_vs_{team2.id}_{timestamp}"

        # Save text report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        report_path = REPORTS_DIR / f"{game_name}_report.txt"
        with open(report_path, "w") as f:
            f.write(text_report)
        print(f"\nText report saved: {report_path}")

        # Save JSON report
        json_path = REPORTS_DIR / f"{game_name}_report.json"
        with open(json_path, "w") as f:
            f.write(json_report)
        print(f"JSON report saved: {json_path}")

        # Save game data
        GAMES_DIR.mkdir(parents=True, exist_ok=True)
        game_data_path = GAMES_DIR / f"{game_name}_data.json"
        with open(game_data_path, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        print(f"Game data saved: {game_data_path}")

    return result


def run_batch_games(team1: Team, team2: Team, num_games: int = 10, save_results: bool = True):
    """Run multiple games and aggregate statistics."""

    print("\n" + "=" * 60)
    print(f"BATCH SIMULATION: {num_games} GAMES")
    print("=" * 60)
    print(f"\n{team1.name} vs {team2.name}")

    config = GameConfig(
        field_length=60.0,
        endzone_depth=7.0,
        field_width=25.0,
        playing_field=46.0,
        drives_per_team=10,
    )

    # Track results
    team1_wins = 0
    team2_wins = 0
    ties = 0
    team1_total_points = 0
    team2_total_points = 0

    all_results = []

    for i in range(num_games):
        simulator = GameSimulator(team1, team2, config, seed=42 + i)
        result = simulator.simulate_game()
        all_results.append(result)

        team1_total_points += result.home_score
        team2_total_points += result.away_score

        if result.winner == team1.id:
            team1_wins += 1
        elif result.winner == team2.id:
            team2_wins += 1
        else:
            ties += 1

        print(f"Game {i + 1}: {team1.name} {result.home_score} - {result.away_score} {team2.name}")

    # Print summary
    summary_lines = []
    summary_lines.append("\n" + "-" * 40)
    summary_lines.append("BATCH RESULTS")
    summary_lines.append("-" * 40)
    summary_lines.append(f"{team1.name}: {team1_wins} wins ({team1_wins / num_games * 100:.1f}%)")
    summary_lines.append(f"{team2.name}: {team2_wins} wins ({team2_wins / num_games * 100:.1f}%)")
    summary_lines.append(f"Ties: {ties}")
    summary_lines.append(f"\nAverage Score:")
    summary_lines.append(f"  {team1.name}: {team1_total_points / num_games:.1f} pts/game")
    summary_lines.append(f"  {team2.name}: {team2_total_points / num_games:.1f} pts/game")

    # Analyze plays across all games
    summary_lines.append("\n" + "-" * 40)
    summary_lines.append("PLAY EFFECTIVENESS (All Games)")
    summary_lines.append("-" * 40)

    play_stats = {}
    for result in all_results:
        for stats in [result.home_stats, result.away_stats]:
            for play_id, ps in stats.play_stats.items():
                if play_id not in play_stats:
                    play_stats[play_id] = {
                        "name": ps.play_name,
                        "calls": 0,
                        "completions": 0,
                        "yards": 0.0,
                        "tds": 0,
                        "tos": 0
                    }
                play_stats[play_id]["calls"] += ps.times_called
                play_stats[play_id]["completions"] += ps.completions
                play_stats[play_id]["yards"] += ps.total_yards
                play_stats[play_id]["tds"] += ps.touchdowns
                play_stats[play_id]["tos"] += ps.turnovers

    # Sort by success rate
    for play_id, data in sorted(
        play_stats.items(),
        key=lambda x: x[1]["completions"] / x[1]["calls"] if x[1]["calls"] > 0 else 0,
        reverse=True
    ):
        calls = data["calls"]
        if calls > 0:
            success_rate = data["completions"] / calls * 100
            avg_yards = data["yards"] / calls
            summary_lines.append(f"{data['name']}: {calls}x | {success_rate:.0f}% | {avg_yards:.1f} avg | TDs:{data['tds']} TOs:{data['tos']}")

    # Print summary
    for line in summary_lines:
        print(line)

    # Save batch results
    if save_results:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_name = f"batch_{team1.id}_vs_{team2.id}_{num_games}games_{timestamp}"

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)

        # Save summary report
        summary_path = REPORTS_DIR / f"{batch_name}_summary.txt"
        with open(summary_path, "w") as f:
            f.write(f"BATCH SIMULATION: {num_games} GAMES\n")
            f.write(f"{team1.name} vs {team2.name}\n")
            f.write("=" * 60 + "\n\n")
            for line in summary_lines:
                f.write(line + "\n")
        print(f"\nBatch summary saved: {summary_path}")

        # Save play effectiveness JSON
        play_stats_path = REPORTS_DIR / f"{batch_name}_play_stats.json"
        with open(play_stats_path, "w") as f:
            json.dump(play_stats, f, indent=2)
        print(f"Play stats saved: {play_stats_path}")

        # Save batch metadata
        batch_data = {
            "timestamp": timestamp,
            "num_games": num_games,
            "team1": {"id": team1.id, "name": team1.name, "wins": team1_wins, "total_points": team1_total_points},
            "team2": {"id": team2.id, "name": team2.name, "wins": team2_wins, "total_points": team2_total_points},
            "ties": ties,
            "play_stats": play_stats
        }
        batch_data_path = GAMES_DIR / f"{batch_name}_data.json"
        GAMES_DIR.mkdir(parents=True, exist_ok=True)
        with open(batch_data_path, "w") as f:
            json.dump(batch_data, f, indent=2)
        print(f"Batch data saved: {batch_data_path}")

    return all_results


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Run game simulations between two teams")
    parser.add_argument("--games", "-g", type=int, default=10, help="Number of games to simulate (default: 10)")
    parser.add_argument("--single", "-s", action="store_true", help="Run a single game with detailed report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")

    args = parser.parse_args()

    # Load demo data (plays, players, etc.)
    print("Loading demo data...")
    load_demo_data()

    # Create custom teams
    print("Creating teams...")
    team1, team2 = create_custom_teams()

    print(f"\nTeam 1: {team1.name}")
    print(f"  Playbook: {len(team1.playbook)} plays")
    for pid in team1.playbook:
        print(f"    - {pid}")

    print(f"\nTeam 2: {team2.name}")
    print(f"  Playbook: {len(team2.playbook)} plays")
    for pid in team2.playbook:
        print(f"    - {pid}")

    save_results = not args.no_save

    if args.single:
        # Run single game with full report
        run_single_game(team1, team2, seed=args.seed, save_results=save_results)
    else:
        # Run batch of games (default)
        run_batch_games(team1, team2, num_games=args.games, save_results=save_results)


if __name__ == "__main__":
    main()
