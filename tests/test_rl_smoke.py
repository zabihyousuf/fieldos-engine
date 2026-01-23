"""RL component smoke tests."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from fieldos_engine.rl.env import PlayCallingEnv
from fieldos_engine.rl.policy import RandomPolicy, EpsilonGreedyBandit
from fieldos_engine.rl.train import TrainingConfig, train_bandit
from fieldos_engine.core.models import Role

# Import seed function
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from seed_demo_data import seed_demo_data


@pytest.fixture(scope="module")
def setup_data():
    """Seed demo data."""
    seed_demo_data()


def test_env_creation(setup_data):
    """Test environment creation."""
    env = PlayCallingEnv(
        playbook=["play_trips_flood", "play_bunch_quick_slants"],
        scenarios=["scenario_man_cover0", "scenario_zone_cover2"],
        offensive_players={
            Role.QB: "player_qb1",
            Role.CENTER: "player_center1",
            Role.WR1: "player_wr1_1",
            Role.WR2: "player_wr2_1",
            Role.WR3: "player_wr3_1"
        },
        defensive_players={
            Role.RUSHER: "player_rusher1",
            Role.CB1: "player_cb1_1",
            Role.CB2: "player_cb2_1",
            Role.SAFETY: "player_safety1",
            Role.LB: "player_lb1"
        },
        seed=42
    )

    assert env.action_space.n == 2
    assert env.observation_space.shape == (15,)


def test_env_reset(setup_data):
    """Test environment reset."""
    env = PlayCallingEnv(
        playbook=["play_trips_flood"],
        scenarios=["scenario_man_cover0"],
        offensive_players={
            Role.QB: "player_qb1",
            Role.CENTER: "player_center1",
            Role.WR1: "player_wr1_1",
            Role.WR2: "player_wr2_1",
            Role.WR3: "player_wr3_1"
        },
        defensive_players={
            Role.RUSHER: "player_rusher1",
            Role.CB1: "player_cb1_1",
            Role.CB2: "player_cb2_1",
            Role.SAFETY: "player_safety1",
            Role.LB: "player_lb1"
        },
        seed=42
    )

    obs, info = env.reset()
    assert obs.shape == (15,)
    assert "situation" in info


def test_env_step(setup_data):
    """Test environment step."""
    env = PlayCallingEnv(
        playbook=["play_trips_flood"],
        scenarios=["scenario_man_cover0"],
        offensive_players={
            Role.QB: "player_qb1",
            Role.CENTER: "player_center1",
            Role.WR1: "player_wr1_1",
            Role.WR2: "player_wr2_1",
            Role.WR3: "player_wr3_1"
        },
        defensive_players={
            Role.RUSHER: "player_rusher1",
            Role.CB1: "player_cb1_1",
            Role.CB2: "player_cb2_1",
            Role.SAFETY: "player_safety1",
            Role.LB: "player_lb1"
        },
        seed=42
    )

    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(0)

    assert isinstance(reward, float)
    assert terminated is True  # Bandit task
    assert "outcome" in info


def test_random_policy(setup_data):
    """Test random policy."""
    policy = RandomPolicy(num_actions=3, seed=42)

    obs = [0.0] * 15
    action = policy.predict(obs)

    assert 0 <= action < 3


def test_bandit_policy(setup_data):
    """Test epsilon-greedy bandit."""
    policy = EpsilonGreedyBandit(num_actions=3, epsilon=0.1, seed=42)

    obs = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.5, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0]

    # Predict
    action = policy.predict(obs, deterministic=False)
    assert 0 <= action < 3

    # Update
    policy.update(obs, action, 10.0)


def test_train_bandit_smoke(setup_data):
    """Test training bandit for a few steps."""
    env = PlayCallingEnv(
        playbook=["play_trips_flood", "play_bunch_quick_slants"],
        scenarios=["scenario_man_cover0"],
        offensive_players={
            Role.QB: "player_qb1",
            Role.CENTER: "player_center1",
            Role.WR1: "player_wr1_1",
            Role.WR2: "player_wr2_1",
            Role.WR3: "player_wr3_1"
        },
        defensive_players={
            Role.RUSHER: "player_rusher1",
            Role.CB1: "player_cb1_1",
            Role.CB2: "player_cb2_1",
            Role.SAFETY: "player_safety1",
            Role.LB: "player_lb1"
        },
        seed=42
    )

    config = TrainingConfig(
        total_steps=50,
        eval_frequency=25,
        eval_episodes=5,
        algorithm="BANDIT",
        seed=42
    )

    result = train_bandit(env, config)

    assert result.total_steps == 50
    assert result.algorithm == "BANDIT"
    assert result.final_reward_mean is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
