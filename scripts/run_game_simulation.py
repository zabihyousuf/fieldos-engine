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
    Team, GamePlayer, DualRolePlayerAttributes, GameConfig, PlayerSpecialty
)
from fieldos_engine.api.main import load_demo_data
from fieldos_engine.sim.game_simulator import GameSimulator, create_sample_teams
from fieldos_engine.stats.report_generator import generate_game_report, GameReportGenerator
from fieldos_engine.stats.scouting import (
    generate_scouting_report, recommend_play_for_situation, ScoutingReportGenerator
)
from fieldos_engine.ai.opponent_model import OpponentModelTrainer, OpponentModel
from fieldos_engine.ai.play_advisor import PlayAdvisor

# Results directory
RESULTS_DIR = Path(__file__).parent.parent / "results"
GAMES_DIR = RESULTS_DIR / "games"
REPORTS_DIR = RESULTS_DIR / "reports"
MODELS_DIR = RESULTS_DIR / "models"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("game_simulation")


def create_custom_teams():
    """Create two custom teams with specific playbooks.

    Each team has 10 players with unified physical attributes (speed/accel/agility)
    that apply to both offense and defense. Players can be:
    - OFFENSE_ONLY: Only selected for offensive plays
    - DEFENSE_ONLY: Only selected for defensive plays
    - TWO_WAY: Can play both offense and defense

    The game dynamically selects the best 5 players for each play based on
    their attributes and the requirements of the position.
    """
    # Team 1: "Butterfingers" - YOUR TEAM (Zabih's team)
    # Players are selected dynamically based on their attribute scores
    team1_players = [
        # TWO-WAY PLAYERS (can play both offense and defense)
        GamePlayer(
            id="bb_p1", name="Zabih Yousuf", number=1,
            attributes=DualRolePlayerAttributes(
                speed=82,
                acceleration=84,
                agility=83,
                height_inches=74,  # 6'2"
                weight_lbs=195,
                # Offense skills
                hands=80,
                route_running=85,
                throw_power=89,
                short_acc=88,
                mid_acc=80,
                deep_acc=83,
                release_time_ms=380,
                decision_latency_ms=320,
                # Defense skills
                man_coverage=82,
                zone_coverage=80,
                ball_skills=85,
                closing_speed=84,
                pass_rush=72,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p2", name="Jaylen", number=23,
            attributes=DualRolePlayerAttributes(
                speed=84,
                acceleration=86,
                agility=82,
                height_inches=70,  # 5'10"
                weight_lbs=175,
                hands=82,
                route_running=85,
                throw_power=45,
                short_acc=48,
                mid_acc=40,
                deep_acc=35,
                man_coverage=85,
                zone_coverage=82,
                ball_skills=85,
                closing_speed=86,
                pass_rush=70,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p3", name="Soh", number=11,
            attributes=DualRolePlayerAttributes(
                speed=87,
                acceleration=88,
                agility=85,
                height_inches=75,  # 6'3" - tall outside WR
                weight_lbs=200,
                hands=99,
                route_running=90,
                throw_power=30,
                short_acc=32,
                mid_acc=25,
                deep_acc=20,
                man_coverage=82,
                zone_coverage=80,
                ball_skills=85,
                closing_speed=86,
                pass_rush=65,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p4", name="Maddox", number=82,
            attributes=DualRolePlayerAttributes(
                speed=92,
                acceleration=90,
                agility=89,
                height_inches=76,  # 6'4" - tall outside WR
                weight_lbs=210,
                hands=99,
                route_running=95,
                throw_power=30,
                short_acc=32,
                mid_acc=25,
                deep_acc=20,
                man_coverage=92,
                zone_coverage=90,
                ball_skills=95,
                closing_speed=91,
                pass_rush=68,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="bb_p5", name="Mikail", number=2,
            attributes=DualRolePlayerAttributes(
                speed=85,
                acceleration=83,
                agility=82,
                height_inches=69,  # 5'9" - quick slot receiver
                weight_lbs=165,
                hands=78,
                route_running=82,
                throw_power=25,
                short_acc=48,
                mid_acc=40,
                deep_acc=35,
                man_coverage=82,
                zone_coverage=80,
                ball_skills=83,
                closing_speed=84,
                pass_rush=60,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        # OFFENSE-ONLY SPECIALISTS
        GamePlayer(
            id="bb_p6", name="DeAndre", number=8,
            attributes=DualRolePlayerAttributes(
                speed=88,
                acceleration=86,
                agility=84,
                height_inches=72,  # 6'0" - versatile receiver
                weight_lbs=185,
                hands=88,
                route_running=85,
                throw_power=28,
                short_acc=30,
                mid_acc=22,
                deep_acc=18,
                man_coverage=65,
                zone_coverage=62,
                ball_skills=70,
                closing_speed=75,
                pass_rush=55,
                specialty=PlayerSpecialty.OFFENSE_ONLY
            )
        ),
        GamePlayer(
            id="bb_p7", name="Trey", number=4,
            attributes=DualRolePlayerAttributes(
                speed=90,
                acceleration=92,
                agility=88,
                height_inches=68,  # 5'8" - fast slot
                weight_lbs=160,
                hands=85,
                route_running=88,
                throw_power=35,
                short_acc=40,
                mid_acc=32,
                deep_acc=25,
                man_coverage=60,
                zone_coverage=58,
                ball_skills=65,
                closing_speed=70,
                pass_rush=50,
                specialty=PlayerSpecialty.OFFENSE_ONLY
            )
        ),
        # DEFENSE-ONLY SPECIALISTS
        GamePlayer(
            id="bb_p8", name="Marcus", number=24,
            attributes=DualRolePlayerAttributes(
                speed=90,
                acceleration=92,
                agility=88,
                height_inches=71,  # 5'11" - quick corner
                weight_lbs=180,
                hands=68,
                route_running=62,
                throw_power=35,
                short_acc=38,
                mid_acc=30,
                deep_acc=25,
                man_coverage=94,
                zone_coverage=90,
                ball_skills=92,
                closing_speed=93,
                pass_rush=65,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
        GamePlayer(
            id="bb_p9", name="Terrell", number=33,
            attributes=DualRolePlayerAttributes(
                speed=86,
                acceleration=88,
                agility=82,
                height_inches=73,  # 6'1" - pass rusher specialist
                weight_lbs=215,
                hands=62,
                route_running=55,
                throw_power=40,
                short_acc=42,
                mid_acc=35,
                deep_acc=30,
                man_coverage=78,
                zone_coverage=82,
                ball_skills=80,
                closing_speed=88,
                pass_rush=95,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
        GamePlayer(
            id="bb_p10", name="Darius", number=31,
            attributes=DualRolePlayerAttributes(
                speed=88,
                acceleration=90,
                agility=85,
                height_inches=74,  # 6'2" - safety/deep zone
                weight_lbs=200,
                hands=70,
                route_running=60,
                throw_power=38,
                short_acc=40,
                mid_acc=32,
                deep_acc=28,
                man_coverage=88,
                zone_coverage=94,
                ball_skills=90,
                closing_speed=92,
                pass_rush=58,
                specialty=PlayerSpecialty.DEFENSE_ONLY
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

    # Team 2: "Godbods" - Speed-focused opponent team
    team2_players = [
        # TWO-WAY PLAYERS
        GamePlayer(
            id="gb_p1", name="Danny", number=22,
            attributes=DualRolePlayerAttributes(
                speed=82,
                acceleration=84,
                agility=89,
                height_inches=73,  # 6'1"
                weight_lbs=190,
                hands=85,
                route_running=85,
                throw_power=89,
                short_acc=90,
                mid_acc=85,
                deep_acc=85,
                release_time_ms=310,
                decision_latency_ms=280,
                man_coverage=80,
                zone_coverage=82,
                ball_skills=85,
                closing_speed=83,
                pass_rush=70,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p2", name="Lincoln", number=55,
            attributes=DualRolePlayerAttributes(
                speed=65,
                acceleration=68,
                agility=62,
                height_inches=72,  # 6'0"
                weight_lbs=200,
                hands=68,
                route_running=65,
                throw_power=92,
                short_acc=88,
                mid_acc=85,
                deep_acc=90,
                man_coverage=62,
                zone_coverage=60,
                ball_skills=58,
                closing_speed=64,
                pass_rush=55,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p3", name="Nick", number=1,
            attributes=DualRolePlayerAttributes(
                speed=94,
                acceleration=92,
                agility=88,
                height_inches=77,  # 6'5" - tall outside WR
                weight_lbs=215,
                hands=99,
                route_running=95,
                throw_power=72,
                short_acc=75,
                mid_acc=68,
                deep_acc=62,
                man_coverage=82,
                zone_coverage=80,
                ball_skills=85,
                closing_speed=90,
                pass_rush=62,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p4", name="Terrance", number=88,
            attributes=DualRolePlayerAttributes(
                speed=92,
                acceleration=90,
                agility=88,
                height_inches=74,  # 6'2" - outside WR
                weight_lbs=195,
                hands=92,
                route_running=95,
                throw_power=32,
                short_acc=25,
                mid_acc=15,
                deep_acc=10,
                man_coverage=82,
                zone_coverage=80,
                ball_skills=85,
                closing_speed=88,
                pass_rush=65,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        GamePlayer(
            id="gb_p5", name="Mike", number=17,
            attributes=DualRolePlayerAttributes(
                speed=88,
                acceleration=85,
                agility=82,
                height_inches=68,  # 5'8" - small but quick
                weight_lbs=160,
                hands=72,
                route_running=70,
                throw_power=18,
                short_acc=15,
                mid_acc=10,
                deep_acc=8,
                man_coverage=88,
                zone_coverage=85,
                ball_skills=88,
                closing_speed=90,
                pass_rush=55,
                specialty=PlayerSpecialty.TWO_WAY
            )
        ),
        # OFFENSE-ONLY SPECIALISTS
        GamePlayer(
            id="gb_p6", name="Tyreek", number=10,
            attributes=DualRolePlayerAttributes(
                speed=99,
                acceleration=98,
                agility=95,
                height_inches=68,  # 5'8" - super fast slot
                weight_lbs=165,
                hands=82,
                route_running=88,
                throw_power=25,
                short_acc=28,
                mid_acc=20,
                deep_acc=15,
                man_coverage=65,
                zone_coverage=62,
                ball_skills=70,
                closing_speed=75,
                pass_rush=50,
                specialty=PlayerSpecialty.OFFENSE_ONLY
            )
        ),
        GamePlayer(
            id="gb_p7", name="Cooper", number=12,
            attributes=DualRolePlayerAttributes(
                speed=86,
                acceleration=84,
                agility=88,
                height_inches=71,  # 5'11"
                weight_lbs=180,
                hands=90,
                route_running=92,
                throw_power=40,
                short_acc=45,
                mid_acc=38,
                deep_acc=30,
                man_coverage=60,
                zone_coverage=58,
                ball_skills=65,
                closing_speed=70,
                pass_rush=48,
                specialty=PlayerSpecialty.OFFENSE_ONLY
            )
        ),
        # DEFENSE-ONLY SPECIALISTS
        GamePlayer(
            id="gb_p8", name="Jalen", number=21,
            attributes=DualRolePlayerAttributes(
                speed=92,
                acceleration=94,
                agility=90,
                height_inches=70,  # 5'10" - lockdown corner
                weight_lbs=175,
                hands=70,
                route_running=65,
                throw_power=38,
                short_acc=40,
                mid_acc=32,
                deep_acc=28,
                man_coverage=96,
                zone_coverage=92,
                ball_skills=90,
                closing_speed=95,
                pass_rush=60,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
        GamePlayer(
            id="gb_p9", name="Khalil", number=99,
            attributes=DualRolePlayerAttributes(
                speed=84,
                acceleration=86,
                agility=78,
                height_inches=75,  # 6'3" - premier pass rusher
                weight_lbs=235,
                hands=58,
                route_running=50,
                throw_power=45,
                short_acc=48,
                mid_acc=40,
                deep_acc=35,
                man_coverage=72,
                zone_coverage=75,
                ball_skills=70,
                closing_speed=85,
                pass_rush=98,
                specialty=PlayerSpecialty.DEFENSE_ONLY
            )
        ),
        GamePlayer(
            id="gb_p10", name="Derwin", number=3,
            attributes=DualRolePlayerAttributes(
                speed=90,
                acceleration=92,
                agility=88,
                height_inches=73,  # 6'1" - rangy safety
                weight_lbs=205,
                hands=72,
                route_running=60,
                throw_power=42,
                short_acc=45,
                mid_acc=38,
                deep_acc=32,
                man_coverage=90,
                zone_coverage=95,
                ball_skills=92,
                closing_speed=94,
                pass_rush=72,
                specialty=PlayerSpecialty.DEFENSE_ONLY
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


def show_scouting_report(team: Team, game_history=None):
    """Show scouting report for a team."""
    print("\n" + "=" * 60)
    print(f"GENERATING SCOUTING REPORT FOR {team.name.upper()}")
    print("=" * 60)

    report = generate_scouting_report(team, game_history, format="text")
    print(report)

    # Save to file
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"scouting_{team.id}_{timestamp}.txt"
    with open(report_path, "w") as f:
        f.write(report)
    print(f"\nScouting report saved: {report_path}")


def interactive_play_caller(team: Team, opponent: Team, game_history=None):
    """Interactive mode for play recommendations."""
    print("\n" + "=" * 60)
    print("INTERACTIVE PLAY CALLER")
    print("=" * 60)
    print(f"\nYour Team: {team.name}")
    print(f"Opponent: {opponent.name}")
    print("\nEnter situations like: '3rd and 5' or 'third and 7 redzone'")
    print("Type 'quit' to exit\n")

    generator = ScoutingReportGenerator(team, game_history)

    # Show opponent analysis upfront
    opp_analysis = generator._analyze_opponent(opponent)
    print(f"\n--- {opponent.name} Defensive Scouting ---")
    if opp_analysis.get("weakness"):
        print(f"  Weakness: {opp_analysis['weakness'].replace('_', ' ').title()}")
    if opp_analysis.get("strength"):
        print(f"  Strength: {opp_analysis['strength'].replace('_', ' ').title()}")
    print(f"  Avg Speed: {opp_analysis['avg_speed']:.0f}")
    print(f"  Weakest: {opp_analysis['weakest_defender']} (Speed: {opp_analysis['weakest_defender_speed']:.0f})")
    for rec in opp_analysis.get("recommendations", []):
        print(f"  >> {rec}")

    while True:
        try:
            user_input = input("\n> Situation: ").strip().lower()

            if user_input in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Parse the input
            down = 1
            yards = 10.0
            zone = "MIDFIELD"
            score_diff = 0

            # Parse down
            if "1st" in user_input or "first" in user_input:
                down = 1
            elif "2nd" in user_input or "second" in user_input:
                down = 2
            elif "3rd" in user_input or "third" in user_input:
                down = 3

            # Parse yards
            import re
            yard_match = re.search(r'and\s+(\d+)', user_input)
            if yard_match:
                yards = float(yard_match.group(1))

            # Parse zone
            if "redzone" in user_input or "red zone" in user_input:
                zone = "REDZONE"
            elif "goal" in user_input or "goalline" in user_input:
                zone = "GOALLINE"
            elif "own" in user_input:
                zone = "OWN_TERRITORY"
            elif "opponent" in user_input or "their" in user_input:
                zone = "OPPONENT_TERRITORY"

            # Parse score
            if "down by" in user_input:
                score_match = re.search(r'down by\s+(\d+)', user_input)
                if score_match:
                    score_diff = -int(score_match.group(1))
            elif "up by" in user_input:
                score_match = re.search(r'up by\s+(\d+)', user_input)
                if score_match:
                    score_diff = int(score_match.group(1))

            # Get recommendation WITH opponent analysis
            rec = generator.recommend_play(down, yards, zone, score_diff, opponent)

            print(f"\n{'-' * 50}")
            print(f"SITUATION: {rec.situation_summary}")
            print(f"{'-' * 50}")
            print("\nRECOMMENDED PLAYS:")

            for i, play in enumerate(rec.recommendations[:3], 1):
                print(f"\n  {i}. {play.play_name}")
                print(f"     Confidence: {play.confidence:.0f}% | Expected: {play.expected_yards:.1f} yds")
                print(f"     {play.reasoning}")

            if rec.recommendations:
                top = rec.recommendations[0]
                print(f"\n>> TOP PICK: {top.play_name}")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error parsing input: {e}")
            print("Try: '3rd and 5' or 'second and 8 redzone'")


def train_opponent_model(my_team: Team, opponent: Team, num_games: int = 100):
    """Train a model against a specific opponent through game simulation."""
    print("\n" + "=" * 60)
    print("OPPONENT MODEL TRAINING")
    print("=" * 60)
    print(f"\nYour Team: {my_team.name}")
    print(f"Training against: {opponent.name}")
    print(f"Games to simulate: {num_games}")
    print()

    # Create trainer and train
    trainer = OpponentModelTrainer(my_team, MODELS_DIR)
    model = trainer.train(opponent, num_games=num_games, verbose=True)

    return model


def show_model_info(my_team: Team, opponent: Team):
    """Show information about a trained opponent model."""
    model_path = MODELS_DIR / f"opponent_{opponent.id}.json"

    if not model_path.exists():
        print(f"\nNo trained model found for {opponent.name}")
        print(f"Train one with: python run_game_simulation.py --train 100")
        return

    model = OpponentModel.load(model_path)
    summary = model.get_model_summary()

    print("\n" + "=" * 60)
    print(f"OPPONENT MODEL: {opponent.name}")
    print("=" * 60)
    print(f"\nYour Team: {my_team.name}")
    print(f"\n--- Training Stats ---")
    print(f"  Games Trained: {summary['games_played']}")
    print(f"  Record: {summary['record']} ({summary['win_rate']:.1f}% win rate)")
    print(f"  Avg Points For: {summary['avg_points_for']:.1f}")
    print(f"  Avg Points Against: {summary['avg_points_against']:.1f}")
    print(f"  Plays Learned: {summary['plays_in_model']}")
    print(f"  Total Plays Recorded: {summary['total_plays_recorded']}")
    print(f"  Last Trained: {summary['last_trained']}")

    # Show top plays overall
    print(f"\n--- Top Plays vs {opponent.name} ---")
    plays = []
    for play_id, data in model.overall_play_stats.items():
        stats = data["stats"]
        if stats["attempts"] >= 5:
            plays.append((data["play_name"], stats))

    plays.sort(key=lambda x: (x[1]["success_rate"], x[1]["avg_yards"]), reverse=True)

    for i, (play_name, stats) in enumerate(plays[:5], 1):
        print(f"  {i}. {play_name}")
        print(f"     Success: {stats['success_rate']:.0f}% | Avg: {stats['avg_yards']:.1f} yds | "
              f"TDs: {stats['touchdowns']} | TOs: {stats['turnovers']} ({stats['attempts']} attempts)")


def interactive_advisor(my_team: Team, opponent: Team):
    """Interactive LLM-powered play advisor that uses trained models."""
    print("\n" + "=" * 60)
    print("AI PLAY ADVISOR")
    print("=" * 60)
    print(f"\nYour Team: {my_team.name}")
    print(f"Opponent: {opponent.name}")
    print("\nThis advisor uses a trained model to recommend plays.")
    print("It learns what works against THIS specific opponent.")
    print()

    # Create the advisor
    advisor = PlayAdvisor(my_team, MODELS_DIR)

    # Check if model exists and get/train it
    model_path = MODELS_DIR / f"opponent_{opponent.id}.json"
    if not model_path.exists():
        print(f"No model exists for {opponent.name}. Training now...")
        print("(This happens automatically - training 50 games)")
        model = advisor.get_model(opponent, auto_train=True, min_games=50)
    else:
        model = advisor.get_model(opponent, auto_train=False)
        if model:
            summary = model.get_model_summary()
            print(f"Model loaded: {summary['games_played']} games trained")
            print(f"Record vs {opponent.name}: {summary['record']} ({summary['win_rate']:.1f}%)")
        else:
            print(f"Training model...")
            model = advisor.get_model(opponent, auto_train=True, min_games=50)

    print()
    print("Enter situations like:")
    print("  '3rd and 5'")
    print("  '2nd and 8 redzone'")
    print("  '1st and 10 down by 7'")
    print("  'third and 2 goalline up by 3'")
    print("\nType 'quit' to exit, 'train N' to train N more games")
    print()

    import re

    while True:
        try:
            user_input = input("\n> What's the situation? ").strip().lower()

            if user_input in ('quit', 'exit', 'q'):
                print("Goodbye!")
                break

            if not user_input:
                continue

            # Handle training command
            if user_input.startswith("train"):
                match = re.search(r'train\s+(\d+)', user_input)
                num_games = int(match.group(1)) if match else 50
                print(f"\nTraining {num_games} more games...")
                model = advisor.get_model(opponent, auto_train=True, min_games=model.games_played + num_games if model else num_games)
                advisor._model_cache[opponent.id] = model
                print("Training complete! Try asking for a play recommendation.")
                continue

            # Parse the input
            down = 1
            yards = 10.0
            field_position = 23.0  # Default midfield
            score_diff = 0

            # Parse down
            if "1st" in user_input or "first" in user_input:
                down = 1
            elif "2nd" in user_input or "second" in user_input:
                down = 2
            elif "3rd" in user_input or "third" in user_input:
                down = 3

            # Parse yards
            yard_match = re.search(r'and\s+(\d+)', user_input)
            if yard_match:
                yards = float(yard_match.group(1))

            # Parse zone -> convert to field_position
            if "redzone" in user_input or "red zone" in user_input:
                field_position = 38.0  # About 8 yards from goal
            elif "goal" in user_input or "goalline" in user_input:
                field_position = 43.0  # About 3 yards from goal
            elif "own" in user_input:
                field_position = 10.0  # Deep in own territory
            elif "opponent" in user_input or "their" in user_input:
                field_position = 30.0

            # Parse score
            if "down by" in user_input:
                score_match = re.search(r'down by\s+(\d+)', user_input)
                if score_match:
                    score_diff = -int(score_match.group(1))
            elif "up by" in user_input:
                score_match = re.search(r'up by\s+(\d+)', user_input)
                if score_match:
                    score_diff = int(score_match.group(1))

            # Get recommendation from the advisor
            response = advisor.recommend(
                opponent=opponent,
                down=down,
                yards_to_go=yards,
                field_position=field_position,
                score_diff=score_diff,
                auto_train=False  # Don't auto-train during interactive session
            )

            if response:
                # Print the formatted response
                print(advisor.format_response_text(response))
            else:
                print("\nNo recommendation available. Try training more games with 'train 50'")

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Try: '3rd and 5' or 'second and 8 redzone'")


def get_advisor_recommendation(my_team: Team, opponent: Team, down: int, yards: float,
                                field_position: float = 23.0, score_diff: int = 0):
    """Get a single play recommendation from the trained model."""
    advisor = PlayAdvisor(my_team, MODELS_DIR)

    # Get model (auto-train if needed)
    model = advisor.get_model(opponent, auto_train=True, min_games=50)

    if not model:
        print(f"Could not get model for {opponent.name}")
        return

    response = advisor.recommend(
        opponent=opponent,
        down=down,
        yards_to_go=yards,
        field_position=field_position,
        score_diff=score_diff,
        auto_train=False
    )

    if response:
        print(advisor.format_response_text(response))
    else:
        print("No recommendation available.")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run game simulations and get play recommendations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_game_simulation.py                  # Run 10 games
  python run_game_simulation.py --single         # Run 1 detailed game
  python run_game_simulation.py --scout          # Show scouting report
  python run_game_simulation.py --play-caller    # Interactive play recommendations
  python run_game_simulation.py --recommend 3 5  # What play for 3rd and 5?

  # AI Model Training & Advisor (learns opponent-specific tendencies):
  python run_game_simulation.py --train 100      # Train model against opponent (100 games)
  python run_game_simulation.py --model-info     # Show trained model stats
  python run_game_simulation.py --advisor        # Interactive AI advisor (uses trained model)
  python run_game_simulation.py --ai 3 5         # AI recommendation for 3rd and 5
        """
    )
    parser.add_argument("--games", "-g", type=int, default=10, help="Number of games to simulate (default: 10)")
    parser.add_argument("--single", "-s", action="store_true", help="Run a single game with detailed report")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--no-save", action="store_true", help="Don't save results to disk")
    parser.add_argument("--scout", action="store_true", help="Generate scouting report for your team")
    parser.add_argument("--scout-opponent", action="store_true", help="Generate scouting report for opponent")
    parser.add_argument("--play-caller", "-p", action="store_true", help="Interactive play caller mode")
    parser.add_argument("--recommend", nargs=2, type=float, metavar=("DOWN", "YARDS"),
                        help="Get play recommendation for specific situation (e.g., --recommend 3 5)")
    parser.add_argument("--zone", default="MIDFIELD",
                        choices=["OWN_TERRITORY", "MIDFIELD", "OPPONENT_TERRITORY", "REDZONE", "GOALLINE"],
                        help="Field zone for --recommend")
    parser.add_argument("--score-diff", type=int, default=0,
                        help="Score differential for --recommend (positive = winning)")

    # AI Model Training & Advisor
    parser.add_argument("--train", type=int, metavar="GAMES",
                        help="Train opponent model with N simulated games")
    parser.add_argument("--model-info", action="store_true",
                        help="Show info about trained opponent model")
    parser.add_argument("--advisor", "-a", action="store_true",
                        help="Interactive AI advisor (uses trained model)")
    parser.add_argument("--ai", nargs=2, type=float, metavar=("DOWN", "YARDS"),
                        help="Get AI recommendation using trained model (e.g., --ai 3 5)")
    parser.add_argument("--field-pos", type=float, default=23.0,
                        help="Field position in yards from own goal (0-46) for --ai")

    args = parser.parse_args()

    # Load demo data (plays, players, etc.)
    print("Loading demo data...")
    load_demo_data()

    # Create custom teams
    print("Creating teams...")
    team1, team2 = create_custom_teams()

    # Handle scouting report
    if args.scout:
        show_scouting_report(team1)
        return

    if args.scout_opponent:
        show_scouting_report(team2)
        return

    # Handle play caller mode
    if args.play_caller:
        interactive_play_caller(team1, team2)  # Pass opponent for matchup analysis
        return

    # Handle single play recommendation (rule-based)
    if args.recommend:
        down = int(args.recommend[0])
        yards = args.recommend[1]
        rec = recommend_play_for_situation(
            team1, down, yards, args.zone, args.score_diff,
            opponent_team=team2  # Include opponent analysis
        )
        print(rec)
        return

    # Handle AI model training
    if args.train:
        train_opponent_model(team1, team2, num_games=args.train)
        return

    # Handle model info
    if args.model_info:
        show_model_info(team1, team2)
        return

    # Handle interactive AI advisor
    if args.advisor:
        interactive_advisor(team1, team2)
        return

    # Handle single AI recommendation
    if args.ai:
        down = int(args.ai[0])
        yards = args.ai[1]
        get_advisor_recommendation(
            team1, team2, down, yards,
            field_position=args.field_pos,
            score_diff=args.score_diff
        )
        return

    # Show team info
    print(f"\nTeam 1: {team1.name}")
    print(f"  Players: {len(team1.players)}")
    print(f"  Playbook: {len(team1.playbook)} plays")
    for pid in team1.playbook:
        print(f"    - {pid}")

    print(f"\nTeam 2: {team2.name}")
    print(f"  Players: {len(team2.players)}")
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
