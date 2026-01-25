#!/usr/bin/env python3
"""
Generate visualizations for all plays in the registry.
"""

import sys
import os
from pathlib import Path
import logging

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.core.registry import registry
from fieldos_engine.core.models import Role
from fieldos_engine.sim.engine import SimulationEngine
from fieldos_engine.api.main import load_demo_data
from fieldos_engine.utils.viz import animate_play

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("generate_all_viz")

def generate_all(output_dir: str = "visualizations", seed: int = 42):
    """Generate GIFs for all plays."""
    
    # Load data
    load_demo_data()
    plays = registry.plays.list()
    
    # Use Zone Cover 2 as generic test scenario
    scenarios = registry.scenarios.list()
    scenario = next((s for s in scenarios if s.name == "Zone Cover 2"), None)
    
    if not scenario:
        # Fallback
        scenario = scenarios[0]
        logger.warning(f"Zone Cover 2 not found, using {scenario.name}")

    # Setup output dir
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Generating visualizations for {len(plays)} plays into {out_path}...")

    # Setup players
    off_roles = [Role.QB, Role.CENTER, Role.WR1, Role.WR2, Role.WR3]
    def_roles = [Role.RUSHER, Role.CB1, Role.CB2, Role.SAFETY, Role.LB]
    all_players = registry.players.list()
    
    off_players = {role: next(p for p in all_players if p.role == role) for role in off_roles}
    def_players = {role: next(p for p in all_players if p.role == role) for role in def_roles}

    engine = SimulationEngine(seed=seed)

    for play in plays:
        print(f"  Processing: {play.name}...")
        
        # Simulate
        outcome, trace = engine.simulate_play(
            play, 
            scenario, 
            off_players, 
            def_players, 
            record_trace=True
        )
        
        if trace:
            filename = f"play_{play.id.replace('play_', '')}.gif"
            file_path = out_path / filename
            animate_play(trace, str(file_path))
            print(f"    -> Saved to {filename}")
        else:
            print(f"    XX Failed to trace {play.name}")

    print("Done!")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="visualizations", help="Output directory")
    args = parser.parse_args()
    
    generate_all(args.output)
