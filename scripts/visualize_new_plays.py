#!/usr/bin/env python3
"""
Generate visualizations for motion and trick plays.
"""

import sys
import json
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.core.registry import registry
from fieldos_engine.core.models import Role, Play, Scenario
from fieldos_engine.sim.engine import SimulationEngine
from fieldos_engine.sim.coverage import CoverageAssignment
from fieldos_engine.api.main import load_demo_data
from fieldos_engine.utils.viz import animate_play, visualize_play_static
import numpy as np


def generate_visualizations():
    """Generate visualizations for all motion and trick plays."""

    # Load demo data
    load_demo_data()

    # Create output directory for new plays
    output_dir = Path(__file__).parent.parent / "visualizations" / "new_plays"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all plays and scenarios
    all_plays = registry.plays.list()
    all_scenarios = registry.scenarios.list()

    # Filter to motion and trick plays
    new_play_ids = [
        # Motion plays
        "play_jet_motion_sweep",
        "play_orbit_screen",
        "play_bunch_stack_motion",
        "play_center_motion_screen",
        "play_trade_cross",
        # Trick plays
        "play_shovel_option",
        "play_reverse_pass",
        "play_double_move_pitch",
        "play_center_throwback",
        "play_halfback_pass"
    ]

    new_plays = [p for p in all_plays if p.id in new_play_ids]

    if not new_plays:
        print("No motion/trick plays found. Make sure they are loaded.")
        print(f"Available plays: {[p.id for p in all_plays]}")
        return

    print(f"Found {len(new_plays)} new plays to visualize")

    # Get a good scenario (with rush to make it interesting)
    scenario = next((s for s in all_scenarios if "Rush" in s.name), all_scenarios[0])
    print(f"Using scenario: {scenario.name}")

    # Get players
    all_players = registry.players.list()
    off_roles = [Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3]

    off_players = {}
    for role in off_roles:
        player = next((p for p in all_players if p.role == role), None)
        if player:
            off_players[role] = player

    def_players = {}
    for def_role in scenario.defender_start_positions.keys():
        player = next((p for p in all_players if p.role == def_role), None)
        if player:
            def_players[def_role] = player

    # Generate visualizations for each play
    for play in new_plays:
        print(f"\nVisualizing: {play.name}")

        try:
            # Get coverage assignments
            rng = np.random.default_rng(42)
            coverage = CoverageAssignment(
                scenario.defense_call,
                scenario.defender_start_positions,
                list(play.qb_plan.progression_roles),
                rng=rng
            )
            coverage_assignments = coverage.assignments

            # Generate static PNG
            static_output = output_dir / f"{play.id}.png"
            visualize_play_static(play, scenario, str(static_output), coverage_assignments)
            print(f"  Static: {static_output}")

            # Run simulation for GIF
            engine = SimulationEngine(seed=42)
            outcome, trace = engine.simulate_play(
                play,
                scenario,
                off_players,
                def_players,
                record_trace=True
            )

            print(f"  Outcome: {outcome.outcome.value}, Yards: {outcome.yards_gained:.1f}")

            if trace:
                gif_output = output_dir / f"{play.id}.gif"
                animate_play(
                    trace,
                    str(gif_output),
                    play=play,
                    scenario=scenario,
                    coverage_assignments=coverage_assignments
                )
                print(f"  Animation: {gif_output}")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\nVisualizations saved to: {output_dir}")


if __name__ == "__main__":
    generate_visualizations()
