#!/usr/bin/env python3
"""
Visualize a play simulation.

Runs a single simulation of a play against a scenario and generates a GIF animation.
"""

import sys
import argparse
from pathlib import Path
import logging
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.core.registry import registry
from fieldos_engine.core.models import Role
from fieldos_engine.sim.engine import SimulationEngine
from fieldos_engine.sim.coverage import CoverageAssignment
from fieldos_engine.api.main import load_demo_data
from fieldos_engine.utils.viz import animate_play, visualize_play_static

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("visualize_play")


def visualize(
    play_name: str,
    scenario_name: str,
    output: str,
    seed: int = 42,
    static: bool = False,
    full_duration: bool = True
):
    """Run simulation and visualize."""

    # Load data
    load_demo_data()

    # Find play
    plays = registry.plays.list()
    play = next((p for p in plays if p.name == play_name or p.id == play_name), None)
    if not play:
        print(f"Error: Play '{play_name}' not found.")
        print("Available plays:")
        for p in plays:
            print(f"  - {p.name}")
        return

    # Find scenario
    scenarios = registry.scenarios.list()
    scenario = next((s for s in scenarios if s.name == scenario_name or s.id == scenario_name), None)
    if not scenario:
        print(f"Error: Scenario '{scenario_name}' not found.")
        print("Available scenarios:")
        for s in scenarios:
            print(f"  - {s.name}")
        return

    print(f"Simulating '{play.name}' vs '{scenario.name}'...")
    print(f"  Defense: {scenario.defense_call.type.value} {scenario.defense_call.shell.value}")
    print(f"  Rushers: {scenario.defense_call.rushers_count}")

    # Setup offensive players
    off_roles = [Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3]
    all_players = registry.players.list()

    off_players = {}
    for role in off_roles:
        player = next((p for p in all_players if p.role == role), None)
        if player:
            off_players[role] = player

    # Setup defensive players - get roles from scenario
    def_players = {}
    for def_role in scenario.defender_start_positions.keys():
        player = next((p for p in all_players if p.role == def_role), None)
        if player:
            def_players[def_role] = player

    # Get coverage assignments for visualization
    rng = np.random.default_rng(seed)
    coverage = CoverageAssignment(
        scenario.defense_call,
        scenario.defender_start_positions,
        list(play.qb_plan.progression_roles),
        rng=rng
    )
    coverage_assignments = coverage.assignments

    print(f"  Coverage Assignments:")
    for def_role, assignment in coverage_assignments.items():
        if assignment is None:
            print(f"    {def_role.value}: Rusher")
        elif isinstance(assignment, Role):
            print(f"    {def_role.value}: Man on {assignment.value}")
        else:
            print(f"    {def_role.value}: Zone ({assignment})")

    # Static diagram only
    if static:
        print(f"Generating static diagram to {output}...")
        path = visualize_play_static(play, scenario, output, coverage_assignments)
        if path:
            print(f"Success! Diagram saved to: {path}")
        else:
            print("Failed to generate diagram.")
        return

    # Run simulation
    engine = SimulationEngine(seed=seed)
    outcome, trace = engine.simulate_play(
        play,
        scenario,
        off_players,
        def_players,
        record_trace=True
    )

    print(f"Outcome: {outcome.outcome.value}, Yards: {outcome.yards_gained:.1f}")
    if outcome.target_role:
        print(f"Target: {outcome.target_role.value}")
    if outcome.time_to_throw_ms:
        print(f"Time to throw: {outcome.time_to_throw_ms/1000:.2f}s")

    if trace:
        print(f"\nTrace has {len(trace.states)} frames")
        print(f"Generating animation to {output}...")

        # Use the enhanced animate_play with play/scenario/coverage info
        path = animate_play(
            trace,
            output,
            play=play,
            scenario=scenario,
            coverage_assignments=coverage_assignments
        )
        if path:
            print(f"Success! Animation saved to: {path}")
        else:
            print("Failed to generate animation.")
    else:
        print("Error: No trace recorded.")


def list_available():
    """List all available plays and scenarios."""
    load_demo_data()

    print("\nAvailable Plays:")
    for p in registry.plays.list():
        routes = [r.value for r, route in p.assignments.items() if route is not None]
        print(f"  - {p.name} (routes: {', '.join(routes)})")

    print("\nAvailable Scenarios:")
    for s in registry.scenarios.list():
        print(f"  - {s.name} ({s.defense_call.type.value} {s.defense_call.shell.value}, rush={s.defense_call.rushers_count})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a play")
    parser.add_argument("--play", type=str, help="Name or ID of the play")
    parser.add_argument("--scenario", type=str, help="Name or ID of the scenario")
    parser.add_argument("--output", type=str, default="play.gif", help="Output file path")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--static", action="store_true", help="Generate static PNG instead of GIF")
    parser.add_argument("--list", action="store_true", help="List available plays and scenarios")

    args = parser.parse_args()

    if args.list:
        list_available()
    elif args.play and args.scenario:
        visualize(args.play, args.scenario, args.output, args.seed, args.static)
    else:
        parser.print_help()
        print("\nExamples:")
        print('  python visualize_play.py --list')
        print('  python visualize_play.py --play "Trips Flood" --scenario "Cover 2 (3-2 No Rush)"')
        print('  python visualize_play.py --play "Trips Flood" --scenario "Man Coverage (With Rush)"')
