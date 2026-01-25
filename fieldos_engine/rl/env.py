"""Gymnasium environment for play-calling RL."""

import gymnasium as gym
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

from ..core.models import (
    Play, Player, Scenario, GameSituation,
    DownDistanceBucket, Role, OutcomeType, Point2D
)
from ..core.registry import registry
from ..sim.engine import SimulationEngine
from .play_generator import PlayGenerator, generate_expanded_playbook


@dataclass
class PlayCallingState:
    """State for play-calling task."""
    situation_bucket: DownDistanceBucket
    yards_to_gain: float
    yardline: float
    defense_shell: str
    defense_type: str
    situation: Optional[GameSituation] = None


class PlayCallingEnv(gym.Env):
    """
    Gymnasium environment for play-calling as contextual bandit/RL.

    State: Game situation features (down/distance/field position) + defense features
    Action: Choose play from playbook
    Reward: Yards gained (clipped) + bonus for conversion

    Features:
    - Supports both registered plays and dynamically generated plays
    - Situation-aware defensive adjustments (3rd & long = deeper coverage)
    - Can discover new play combinations through generation
    """

    def __init__(
        self,
        playbook: List[str],  # play_ids
        scenarios: List[str],  # scenario_ids
        offensive_players: Dict[Role, str],  # role -> player_id
        defensive_players: Dict[Role, str],  # role -> player_id
        situation_distribution: Optional[List[GameSituation]] = None,
        seed: Optional[int] = None,
        deterministic: bool = False,
        use_generated_plays: bool = False,
        num_generated_plays: int = 20
    ):
        super().__init__()

        self.playbook = playbook
        self.scenarios = scenarios
        self.offensive_player_ids = offensive_players
        self.defensive_player_ids = defensive_players
        self.situation_distribution = situation_distribution or self._default_situations()
        self.seed_value = seed
        self.deterministic = deterministic

        self.rng = np.random.Generator(np.random.PCG64(seed))
        self.sim_engine = SimulationEngine(seed=seed)

        # Generated plays support
        self.use_generated_plays = use_generated_plays
        self.generated_plays: List[Play] = []
        if use_generated_plays:
            self._initialize_generated_plays(num_generated_plays)

        # Action space: discrete play selection
        total_plays = len(playbook) + len(self.generated_plays)
        self.action_space = gym.spaces.Discrete(total_plays)

        # Observation space: situation features
        # [bucket_onehot(7), yards_to_gain(normalized), yardline(normalized),
        #  defense_shell_onehot(4), defense_type_onehot(2)]
        # Total: 7 + 1 + 1 + 4 + 2 = 15
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(15,), dtype=np.float32
        )

        self.current_state: Optional[PlayCallingState] = None
        self.episode_steps = 0
        self.max_episode_steps = 1  # Single-step episodes for bandit

        # Track play performance for analysis
        self.play_stats: Dict[str, Dict] = {}

    def _initialize_generated_plays(self, num_plays: int):
        """Initialize generated plays for exploration."""
        generator = PlayGenerator(seed=self.seed_value)

        # Generate concept plays
        concepts = ["flood", "mesh", "smash", "stick", "verticals", "slants"]
        formations = ["trips_right", "bunch_tight", "tight_bunch_right", "spread"]

        for concept in concepts:
            gen = generator.generate_concept_play(concept, formations[0])
            self.generated_plays.append(gen.play)

        # Generate random plays
        remaining = max(0, num_plays - len(concepts))
        random_batch = generator.generate_play_batch(remaining)
        for gen in random_batch:
            self.generated_plays.append(gen.play)

    def get_play_by_action(self, action: int) -> Play:
        """Get play object for an action index."""
        if action < len(self.playbook):
            # Registered play
            play_id = self.playbook[action]
            play = registry.plays.get(play_id)
            if play is None:
                raise ValueError(f"Play {play_id} not found in registry")
            return play
        else:
            # Generated play
            gen_idx = action - len(self.playbook)
            if gen_idx >= len(self.generated_plays):
                raise ValueError(f"Generated play index {gen_idx} out of range")
            return self.generated_plays[gen_idx]

    def get_play_id(self, action: int) -> str:
        """Get play ID for an action index."""
        if action < len(self.playbook):
            return self.playbook[action]
        else:
            gen_idx = action - len(self.playbook)
            return self.generated_plays[gen_idx].id

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to new situation."""
        if seed is not None:
            self.rng = np.random.Generator(np.random.PCG64(seed))
            self.sim_engine = SimulationEngine(seed=seed)

        # Sample situation and scenario
        situation = self._sample_situation()
        scenario = self._sample_scenario()

        # Create state with situation for coverage adjustment
        self.current_state = PlayCallingState(
            situation_bucket=situation.bucket,
            yards_to_gain=situation.yards_to_gain,
            yardline=situation.yardline_to_goal,
            defense_shell=scenario.defense_call.shell.value,
            defense_type=scenario.defense_call.type.value,
            situation=situation
        )

        self.current_situation = situation
        self.current_scenario_id = scenario.id
        self.episode_steps = 0

        obs = self._get_observation()
        info = {
            "situation": situation.model_dump(),
            "scenario_id": scenario.id,
            "total_plays": len(self.playbook) + len(self.generated_plays)
        }

        return obs, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute action (choose play).

        Args:
            action: Index into playbook (registered + generated)

        Returns:
            (observation, reward, terminated, truncated, info)
        """
        if self.current_state is None:
            raise RuntimeError("Must call reset() before step()")

        # Get play
        play = self.get_play_by_action(action)
        play_id = self.get_play_id(action)

        # Get scenario
        scenario = registry.scenarios.get(self.current_scenario_id)
        if scenario is None:
            raise ValueError(f"Scenario {self.current_scenario_id} not found")

        # Get players
        off_players = {
            role: registry.players.get(player_id)
            for role, player_id in self.offensive_player_ids.items()
        }
        def_players = {
            role: registry.players.get(player_id)
            for role, player_id in self.defensive_player_ids.items()
        }

        # Simulate play with situation-aware defense
        outcome, _ = self.sim_engine.simulate_play(
            play, scenario, off_players, def_players,
            record_trace=False,
            situation=self.current_situation
        )

        # Compute reward
        reward = self._compute_reward(outcome)

        # Track play performance
        if play_id not in self.play_stats:
            self.play_stats[play_id] = {
                "attempts": 0,
                "completions": 0,
                "yards_total": 0,
                "rewards_total": 0
            }
        self.play_stats[play_id]["attempts"] += 1
        if outcome.outcome == OutcomeType.COMPLETE:
            self.play_stats[play_id]["completions"] += 1
        self.play_stats[play_id]["yards_total"] += outcome.yards_gained
        self.play_stats[play_id]["rewards_total"] += reward

        # Episode ends after one step (bandit)
        terminated = True
        truncated = False

        info = {
            "play_id": play_id,
            "outcome": outcome.model_dump(),
            "yards_gained": outcome.yards_gained,
            "situation": self.current_situation.model_dump(),
            "is_generated": action >= len(self.playbook)
        }

        # Next observation (doesn't matter for terminated episode)
        obs = self._get_observation()

        return obs, reward, terminated, truncated, info

    def _get_observation(self) -> np.ndarray:
        """Convert state to observation vector."""
        if self.current_state is None:
            return np.zeros(15, dtype=np.float32)

        obs = []

        # Bucket one-hot (7 buckets)
        bucket_map = {
            DownDistanceBucket.FIRST_ANY: 0,
            DownDistanceBucket.SECOND_SHORT: 1,
            DownDistanceBucket.SECOND_LONG: 2,
            DownDistanceBucket.THIRD_SHORT: 3,
            DownDistanceBucket.THIRD_LONG: 4,
            DownDistanceBucket.REDZONE: 5,
            DownDistanceBucket.GOALLINE: 6
        }
        bucket_onehot = np.zeros(7)
        bucket_onehot[bucket_map[self.current_state.situation_bucket]] = 1.0
        obs.extend(bucket_onehot)

        # Yards to gain (normalized 0-25)
        obs.append(min(1.0, self.current_state.yards_to_gain / 25.0))

        # Yardline (normalized 0-50)
        obs.append(min(1.0, self.current_state.yardline / 50.0))

        # Defense shell one-hot (4)
        shell_map = {"COVER0": 0, "COVER1": 1, "COVER2": 2, "COVER3": 3}
        shell_onehot = np.zeros(4)
        shell_onehot[shell_map[self.current_state.defense_shell]] = 1.0
        obs.extend(shell_onehot)

        # Defense type one-hot (2)
        type_map = {"MAN": 0, "ZONE": 1}
        type_onehot = np.zeros(2)
        type_onehot[type_map[self.current_state.defense_type]] = 1.0
        obs.extend(type_onehot)

        return np.array(obs, dtype=np.float32)

    def _compute_reward(self, outcome) -> float:
        """
        Compute reward for outcome.

        - Base: yards gained (clipped to -5, +25)
        - Bonus: +10 for conversion on 3rd/4th down
        - Penalty: -5 for sack, -10 for interception
        """
        reward = np.clip(outcome.yards_gained, -5.0, 25.0)

        # Conversion bonus
        if self.current_situation:
            if self.current_situation.down >= 3:
                if outcome.yards_gained >= self.current_situation.yards_to_gain:
                    reward += 10.0  # Conversion!

        # Outcome penalties
        if outcome.outcome == OutcomeType.SACK:
            reward -= 5.0
        elif outcome.outcome == OutcomeType.INTERCEPT:
            reward -= 10.0

        return float(reward)

    def _sample_situation(self) -> GameSituation:
        """Sample a game situation."""
        if self.deterministic:
            idx = self.episode_steps % len(self.situation_distribution)
        else:
            idx = self.rng.integers(0, len(self.situation_distribution))
        return self.situation_distribution[idx]

    def _sample_scenario(self) -> Any:
        """Sample a scenario."""
        if self.deterministic:
            idx = self.episode_steps % len(self.scenarios)
        else:
            idx = self.rng.integers(0, len(self.scenarios))
        scenario_id = self.scenarios[idx]
        return registry.scenarios.get(scenario_id)

    def _default_situations(self) -> List[GameSituation]:
        """Default situation distribution."""
        return [
            GameSituation(down=1, yards_to_gain=25.0, yardline_to_goal=40.0),
            GameSituation(down=2, yards_to_gain=5.0, yardline_to_goal=35.0),
            GameSituation(down=2, yards_to_gain=15.0, yardline_to_goal=35.0),
            GameSituation(down=3, yards_to_gain=3.0, yardline_to_goal=30.0),
            GameSituation(down=3, yards_to_gain=10.0, yardline_to_goal=30.0),
            GameSituation(down=1, yards_to_gain=10.0, yardline_to_goal=15.0),  # redzone
            GameSituation(down=1, yards_to_gain=5.0, yardline_to_goal=5.0),  # goalline
        ]

    def get_best_plays(self, top_n: int = 5) -> List[Tuple[str, float]]:
        """Get the best performing plays by average reward."""
        play_avgs = []
        for play_id, stats in self.play_stats.items():
            if stats["attempts"] > 0:
                avg_reward = stats["rewards_total"] / stats["attempts"]
                play_avgs.append((play_id, avg_reward))

        play_avgs.sort(key=lambda x: x[1], reverse=True)
        return play_avgs[:top_n]

    def get_play_summary(self) -> Dict[str, Dict]:
        """Get summary statistics for all plays."""
        summary = {}
        for play_id, stats in self.play_stats.items():
            if stats["attempts"] > 0:
                summary[play_id] = {
                    "attempts": stats["attempts"],
                    "completion_rate": stats["completions"] / stats["attempts"],
                    "avg_yards": stats["yards_total"] / stats["attempts"],
                    "avg_reward": stats["rewards_total"] / stats["attempts"],
                    "is_generated": play_id.startswith("play_gen_") or play_id.startswith("play_concept_")
                }
        return summary


class DiscoveryEnv(PlayCallingEnv):
    """
    Extended environment for play discovery.

    This environment emphasizes exploration of new play combinations
    and provides additional rewards for discovering high-performing plays.
    """

    def __init__(self, *args, **kwargs):
        # Force generated plays
        kwargs["use_generated_plays"] = True
        kwargs["num_generated_plays"] = kwargs.get("num_generated_plays", 50)
        super().__init__(*args, **kwargs)

        # Track novel discoveries
        self.discovery_threshold = 8.0  # Average reward to consider "discovered"
        self.discovered_plays: set = set()

    def _compute_reward(self, outcome) -> float:
        """Add discovery bonus to reward."""
        base_reward = super()._compute_reward(outcome)

        # Discovery bonus for first-time high performance
        play_id = self.get_play_id(self.last_action) if hasattr(self, 'last_action') else None
        if play_id and play_id not in self.discovered_plays:
            if base_reward > self.discovery_threshold:
                self.discovered_plays.add(play_id)
                base_reward += 5.0  # Discovery bonus

        return base_reward

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """Track last action for discovery bonus."""
        self.last_action = action
        return super().step(action)

    def get_discovered_plays(self) -> List[str]:
        """Get list of discovered high-performing plays."""
        return list(self.discovered_plays)
