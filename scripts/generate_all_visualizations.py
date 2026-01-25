#!/usr/bin/env python3
"""
Generate visualizations for all defensive scenarios.

Creates GIF animations and static diagrams for each defensive coverage scheme.
"""

import sys
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

logging.basicConfig(level=logging.WARNING)


def generate_all_defense_visualizations(output_dir: str = "visualizations"):
    """Generate visualizations for all defensive scenarios."""

    # Create output directory
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading demo data...")
    load_demo_data()

    plays = registry.plays.list()
    scenarios = registry.scenarios.list()
    all_players = registry.players.list()

    print(f"Found {len(plays)} plays and {len(scenarios)} scenarios")

    # Setup offensive players
    off_players = {}
    for role in [Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3]:
        player = next((p for p in all_players if p.role == role), None)
        if player:
            off_players[role] = player

    # Use a representative play for defense visualizations
    demo_play = next((p for p in plays if "Trips" in p.name), plays[0])

    print(f"\nGenerating defense visualizations using '{demo_play.name}'...")
    print("=" * 60)

    for scenario in scenarios:
        # Get defensive players for this scenario
        def_players = {}
        for def_role in scenario.defender_start_positions.keys():
            player = next((p for p in all_players if p.role == def_role), None)
            if player:
                def_players[def_role] = player

        # Get coverage assignments
        rng = np.random.default_rng(42)
        coverage = CoverageAssignment(
            scenario.defense_call,
            scenario.defender_start_positions,
            list(demo_play.qb_plan.progression_roles),
            rng=rng
        )

        # Create safe filename
        safe_name = scenario.name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")

        # Generate static diagram
        static_path = out_path / f"defense_{safe_name}.png"
        print(f"\nGenerating: {static_path.name}")
        print(f"  Coverage: {scenario.defense_call.type.value} {scenario.defense_call.shell.value}")
        print(f"  Rushers: {scenario.defense_call.rushers_count}")

        visualize_play_static(
            demo_play,
            scenario,
            str(static_path),
            coverage.assignments
        )
        print(f"  ✓ Static diagram saved")

        # Generate animated GIF
        gif_path = out_path / f"defense_{safe_name}.gif"

        engine = SimulationEngine(seed=42)
        outcome, trace = engine.simulate_play(
            demo_play,
            scenario,
            off_players,
            def_players,
            record_trace=True
        )

        if trace:
            animate_play(
                trace,
                str(gif_path),
                play=demo_play,
                scenario=scenario,
                coverage_assignments=coverage.assignments
            )
            print(f"  ✓ Animation saved ({len(trace.states)} frames)")
            print(f"  Result: {outcome.outcome.value}, {outcome.yards_gained:.1f} yards")
        else:
            print(f"  ✗ No trace recorded")

    print("\n" + "=" * 60)
    print(f"All visualizations saved to: {out_path.absolute()}")
    print("\nGenerated files:")
    for f in sorted(out_path.glob("defense_*")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate defensive coverage visualizations")
    parser.add_argument("--output", default="visualizations", help="Output directory")
    args = parser.parse_args()

    generate_all_defense_visualizations(args.output)
