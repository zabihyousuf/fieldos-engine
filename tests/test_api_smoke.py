"""API smoke tests."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi.testclient import TestClient
from fieldos_engine.api.main import app
from fieldos_engine.core.registry import registry

# Import seed function
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from seed_demo_data import seed_demo_data


@pytest.fixture(scope="module")
def client():
    """Create test client with seeded data."""
    # Seed data
    seed_demo_data()
    return TestClient(app)


def test_health_check(client):
    """Test health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["ok"] is True


def test_list_plays(client):
    """Test listing plays."""
    response = client.get("/plays")
    assert response.status_code == 200
    plays = response.json()
    assert len(plays) > 0
    assert plays[0]["id"] is not None


def test_get_play(client):
    """Test getting a specific play."""
    # First get list
    response = client.get("/plays")
    plays = response.json()
    play_id = plays[0]["id"]

    # Get specific play
    response = client.get(f"/plays/{play_id}")
    assert response.status_code == 200
    play = response.json()
    assert play["id"] == play_id


def test_list_players(client):
    """Test listing players."""
    response = client.get("/players")
    assert response.status_code == 200
    players = response.json()
    assert len(players) > 0


def test_list_scenarios(client):
    """Test listing scenarios."""
    response = client.get("/scenarios")
    assert response.status_code == 200
    scenarios = response.json()
    assert len(scenarios) > 0


def test_simulate_basic(client):
    """Test basic simulation."""
    # Get first play and scenario
    plays = client.get("/plays").json()
    scenarios = client.get("/scenarios").json()

    play_id = plays[0]["id"]
    scenario_id = scenarios[0]["id"]

    # Simulate
    request_data = {
        "play_id": play_id,
        "scenario_ids": [scenario_id],
        "num_episodes": 5,
        "seed": 42,
        "mode": "EVAL",
        "trace_policy": {"mode": "NONE"}
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "run_id" in result
    assert "metrics" in result
    assert result["num_episodes"] == 5


def test_simulate_with_trace(client):
    """Test simulation with trace sampling."""
    plays = client.get("/plays").json()
    scenarios = client.get("/scenarios").json()

    request_data = {
        "play_id": plays[0]["id"],
        "scenario_ids": [scenarios[0]["id"]],
        "num_episodes": 10,
        "seed": 42,
        "mode": "EVAL",
        "trace_policy": {
            "mode": "TOP_N",
            "top_n": 3
        }
    }

    response = client.post("/simulate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "artifacts" in result
    assert "traces" in result["artifacts"]
    # Should have up to 3 traces
    assert len(result["artifacts"]["traces"]) <= 3


def test_train_bandit(client):
    """Test RL training with bandit."""
    plays = client.get("/plays").json()
    scenarios = client.get("/scenarios").json()
    players = client.get("/players").json()

    # Get player IDs by role
    off_players = {}
    def_players = {}

    for player in players:
        role = player["role"]
        side = player["side"]

        if side == "OFFENSE" and role not in off_players:
            off_players[role] = player["id"]
        elif side == "DEFENSE" and role not in def_players:
            def_players[role] = player["id"]

    # Take subset of plays
    play_ids = [p["id"] for p in plays[:3]]
    scenario_ids = [s["id"] for s in scenarios[:2]]

    request_data = {
        "play_ids": play_ids,
        "scenario_ids": scenario_ids,
        "offensive_players": off_players,
        "defensive_players": def_players,
        "seed": 42,
        "steps": 100,  # Small for test
        "algo": "BANDIT",
        "epsilon": 0.1,
        "learning_rate": 0.1
    }

    response = client.post("/train", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "training_id" in result
    assert "summary" in result


def test_evaluate_policy(client):
    """Test policy evaluation."""
    plays = client.get("/plays").json()
    scenarios = client.get("/scenarios").json()
    players = client.get("/players").json()

    # Get player IDs
    off_players = {}
    def_players = {}

    for player in players:
        role = player["role"]
        side = player["side"]

        if side == "OFFENSE" and role not in off_players:
            off_players[role] = player["id"]
        elif side == "DEFENSE" and role not in def_players:
            def_players[role] = player["id"]

    play_ids = [p["id"] for p in plays[:2]]
    scenario_ids = [s["id"] for s in scenarios[:2]]

    request_data = {
        "policy_id": "baseline",
        "play_ids": play_ids,
        "scenario_ids": scenario_ids,
        "offensive_players": off_players,
        "defensive_players": def_players,
        "num_episodes": 20,
        "seed": 42
    }

    response = client.post("/evaluate", json=request_data)
    assert response.status_code == 200

    result = response.json()
    assert "report" in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
