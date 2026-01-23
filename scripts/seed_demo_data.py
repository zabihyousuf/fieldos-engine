#!/usr/bin/env python3
"""Load demo data into the registry."""

import json
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.core.models import (
    Play, Player, Route, Formation, Scenario, Ruleset
)
from fieldos_engine.core.registry import registry


def load_json_file(filepath: Path):
    """Load JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def seed_demo_data():
    """Load all demo data into registry."""
    data_dir = Path(__file__).parent.parent / "fieldos_engine" / "data" / "demo"

    # Clear existing data
    registry.clear_all()

    # Load routes
    routes_file = data_dir / "routes.json"
    routes_data = load_json_file(routes_file)
    for route_dict in routes_data:
        route = Route(**route_dict)
        registry.routes.create(route.id, route)
    print(f"Loaded {len(routes_data)} routes")

    # Load formations
    formations_file = data_dir / "formations.json"
    formations_data = load_json_file(formations_file)
    for formation_dict in formations_data:
        formation = Formation(**formation_dict)
        registry.formations.create(formation.id, formation)
    print(f"Loaded {len(formations_data)} formations")

    # Load players
    players_file = data_dir / "players.json"
    players_data = load_json_file(players_file)
    for player_dict in players_data:
        player = Player(**player_dict)
        registry.players.create(player.id, player)
    print(f"Loaded {len(players_data)} players")

    # Load rulesets
    rules_file = data_dir / "rules.json"
    rules_data = load_json_file(rules_file)
    for ruleset_dict in rules_data:
        ruleset = Ruleset(**ruleset_dict)
        registry.rulesets.create(ruleset.id, ruleset)
    print(f"Loaded {len(rules_data)} rulesets")

    # Load plays
    plays_file = data_dir / "plays.json"
    plays_data = load_json_file(plays_file)
    for play_dict in plays_data:
        play = Play(**play_dict)
        registry.plays.create(play.id, play)
    print(f"Loaded {len(plays_data)} plays")

    # Load scenarios
    scenarios_file = data_dir / "scenarios.json"
    scenarios_data = load_json_file(scenarios_file)
    for scenario_dict in scenarios_data:
        scenario = Scenario(**scenario_dict)
        registry.scenarios.create(scenario.id, scenario)
    print(f"Loaded {len(scenarios_data)} scenarios")

    print("\nDemo data loaded successfully!")
    print(f"Total entities: {registry.plays.count()} plays, "
          f"{registry.players.count()} players, "
          f"{registry.routes.count()} routes, "
          f"{registry.formations.count()} formations, "
          f"{registry.scenarios.count()} scenarios, "
          f"{registry.rulesets.count()} rulesets")


if __name__ == "__main__":
    seed_demo_data()
