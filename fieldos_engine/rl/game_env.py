"""
Game-level RL Environment for training play-calling policies.

This environment simulates complete games between two teams
and trains an agent to select plays that maximize win probability.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging

from ..core.models import (
    Team, GameConfig, GameState, Play, OutcomeType,
    FieldZone, DriveResult
)
from ..core.registry import registry
from ..sim.game_simulator import GameSimulator, create_sample_teams

logger = logging.getLogger("fieldos_engine.rl.game_env")


class GamePlayCallingEnv(gym.Env):
    """
    RL environment for game-level play calling.

    The agent controls one team's play selection throughout a game.
    The goal is to maximize points scored (and ideally, win the game).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        home_team: Optional[Team] = None,
        away_team: Optional[Team] = None,
        optimize_for: str = "home",
        config: Optional[GameConfig] = None,
        seed: int = 42,
    ):
        super().__init__()

        # Use sample teams if none provided
        if home_team is None or away_team is None:
            home_team, away_team = create_sample_teams()

        self.home_team = home_team
        self.away_team = away_team
        self.optimize_for = optimize_for
        self.config = config or GameConfig()
        self.seed_value = seed

        # Determine which team we're optimizing
        self.our_team = home_team if optimize_for == "home" else away_team
        self.opponent_team = away_team if optimize_for == "home" else home_team

        # Load playbook
        self._load_playbook()

        # Action space: play selection from playbook
        self.action_space = spaces.Discrete(len(self.playbook))

        # Observation space: game state features
        # Features:
        # - Down (1-hot, 3 dims)
        # - Yards to first (normalized, 1 dim)
        # - Field zone (1-hot, 5 dims)
        # - Score differential (normalized, 1 dim)
        # - Our score (normalized, 1 dim)
        # - Their score (normalized, 1 dim)
        # - Drive number (normalized, 1 dim)
        # - Possession (1 if ours, 0 otherwise)
        # - First down achieved (1 dim)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(16,),
            dtype=np.float32
        )

        self.simulator: Optional[GameSimulator] = None
        self.current_drive_plays = 0
        self.episode_reward = 0.0

    def _load_playbook(self):
        """Load plays from team's playbook."""
        all_plays = registry.plays.list()
        play_lookup = {p.id: p for p in all_plays}

        self.playbook: List[Play] = []
        for play_id in self.our_team.playbook:
            if play_id in play_lookup:
                self.playbook.append(play_lookup[play_id])

        if not self.playbook:
            # Use first 5 plays from registry
            self.playbook = all_plays[:5]
            logger.warning(f"No plays in playbook, using first {len(self.playbook)} from registry")

        self.play_to_action = {p.id: i for i, p in enumerate(self.playbook)}
        self.action_to_play = {i: p for i, p in enumerate(self.playbook)}

    def _get_obs(self) -> np.ndarray:
        """Convert game state to observation vector."""
        state = self.simulator.state

        # Down one-hot (3 dims for downs 1-3)
        down_onehot = np.zeros(3)
        if 1 <= state.down <= 3:
            down_onehot[state.down - 1] = 1.0

        # Yards to first (normalized by 20)
        yards_to_first_norm = min(state.yards_to_first / 20.0, 1.0)

        # Field zone one-hot (5 zones)
        zone_onehot = np.zeros(5)
        zone_idx = list(FieldZone).index(state.field_zone)
        zone_onehot[zone_idx] = 1.0

        # Score info (normalized by 42 = 7 TDs)
        our_score = state.home_score if self.optimize_for == "home" else state.away_score
        their_score = state.away_score if self.optimize_for == "home" else state.home_score
        score_diff = (our_score - their_score) / 42.0
        our_score_norm = our_score / 42.0
        their_score_norm = their_score / 42.0

        # Drive and possession info
        drive_norm = state.current_drive / state.total_drives
        is_our_possession = 1.0 if (
            (self.optimize_for == "home" and state.possession == "home") or
            (self.optimize_for == "away" and state.possession == "away")
        ) else 0.0
        first_down_achieved = 1.0 if state.first_down_achieved else 0.0

        obs = np.array([
            *down_onehot,           # 3 dims
            yards_to_first_norm,    # 1 dim
            *zone_onehot,           # 5 dims
            score_diff,             # 1 dim
            our_score_norm,         # 1 dim
            their_score_norm,       # 1 dim
            drive_norm,             # 1 dim
            is_our_possession,      # 1 dim
            first_down_achieved,    # 1 dim
        ], dtype=np.float32)

        return obs

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict] = None
    ) -> Tuple[np.ndarray, Dict]:
        """Reset environment for new game."""
        super().reset(seed=seed)

        actual_seed = seed if seed is not None else self.seed_value
        self.simulator = GameSimulator(
            self.home_team,
            self.away_team,
            self.config,
            seed=actual_seed
        )

        self.current_drive_plays = 0
        self.episode_reward = 0.0

        return self._get_obs(), {}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute a play and return results.

        During our possession, we select plays.
        During opponent possession, plays are selected randomly.
        """
        state = self.simulator.state
        reward = 0.0
        info = {}

        # Check if it's our turn to select
        is_our_possession = (
            (self.optimize_for == "home" and state.possession == "home") or
            (self.optimize_for == "away" and state.possession == "away")
        )

        if is_our_possession:
            # We select the play
            play = self.action_to_play.get(action)
            if play is None:
                play = self.playbook[0]  # Fallback

            off_team = self.our_team
            def_team = self.opponent_team
        else:
            # Opponent selects randomly
            play = self.simulator._select_play(self.opponent_team)
            if play is None:
                play = self.playbook[0]

            off_team = self.opponent_team
            def_team = self.our_team

        # Simulate the play
        result = self.simulator.simulate_play(play, off_team, def_team)

        # Update stats
        self.simulator._update_stats(result, off_team, play)

        # Update game state
        old_our_score = state.home_score if self.optimize_for == "home" else state.away_score
        self.simulator._update_game_state(result)
        new_our_score = state.home_score if self.optimize_for == "home" else state.away_score

        # Calculate reward
        if is_our_possession:
            # Reward for our plays
            reward = self._calculate_reward(result, old_our_score, new_our_score)
        else:
            # Negative reward when opponent scores
            old_their_score = state.away_score if self.optimize_for == "home" else state.home_score
            new_their_score = state.away_score if self.optimize_for == "home" else state.home_score
            if new_their_score > old_their_score:
                reward = -10.0  # Opponent scored

        self.current_drive_plays += 1

        # Check for drive end
        drive_over = (
            result.resulted_in_touchdown or
            result.resulted_in_turnover or
            state.down > 3
        )

        if drive_over:
            self.simulator._switch_possession()
            self.current_drive_plays = 0

        # Check for game end
        done = state.current_drive > state.total_drives

        if done:
            # Final game reward
            our_final = state.home_score if self.optimize_for == "home" else state.away_score
            their_final = state.away_score if self.optimize_for == "home" else state.home_score

            if our_final > their_final:
                reward += 100.0  # Win bonus
                info["won"] = True
            elif our_final < their_final:
                reward -= 50.0  # Loss penalty
                info["won"] = False
            else:
                info["won"] = None  # Tie

            info["final_score"] = (our_final, their_final)

        self.episode_reward += reward
        info["episode_reward"] = self.episode_reward
        info["play_result"] = result

        return self._get_obs(), reward, done, False, info

    def _calculate_reward(
        self,
        result,
        old_score: int,
        new_score: int
    ) -> float:
        """Calculate reward for a play outcome."""
        reward = 0.0

        # Base yards reward
        reward += result.yards_gained * 0.1

        # Completion bonus
        if result.outcome == OutcomeType.COMPLETE:
            reward += 1.0

        # First down bonus
        if result.resulted_in_first_down:
            reward += 5.0

        # Touchdown bonus
        if result.resulted_in_touchdown:
            reward += 20.0

        # Turnover penalty
        if result.resulted_in_turnover:
            reward -= 15.0

        # Sack penalty
        if result.outcome == OutcomeType.SACK:
            reward -= 3.0

        return reward

    def render(self, mode="human"):
        """Render game state."""
        state = self.simulator.state
        our_score = state.home_score if self.optimize_for == "home" else state.away_score
        their_score = state.away_score if self.optimize_for == "home" else state.home_score

        print(f"\n{'='*50}")
        print(f"Drive {state.current_drive}/{state.total_drives}")
        print(f"Score: Us {our_score} - Them {their_score}")
        print(f"Field Position: {state.field_position:.1f} yards")
        print(f"Down: {state.down}, Yards to First: {state.yards_to_first:.1f}")
        print(f"Zone: {state.field_zone.value}")
        print(f"{'='*50}\n")


class SelfPlayEnv(GamePlayCallingEnv):
    """
    Self-play environment where both teams are controlled by learned policies.

    This enables training against increasingly strong opponents.
    """

    def __init__(
        self,
        home_team: Optional[Team] = None,
        away_team: Optional[Team] = None,
        config: Optional[GameConfig] = None,
        seed: int = 42,
    ):
        super().__init__(
            home_team=home_team,
            away_team=away_team,
            optimize_for="home",  # Always optimize for home in self-play
            config=config,
            seed=seed
        )

        self.opponent_policy = None

    def set_opponent_policy(self, policy):
        """Set the opponent's policy for self-play."""
        self.opponent_policy = policy

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """
        Execute plays for both teams.

        Home team uses provided action, away team uses opponent policy.
        """
        state = self.simulator.state

        if state.possession == "home":
            # Home team (agent) plays
            return super().step(action)
        else:
            # Away team (opponent policy) plays
            if self.opponent_policy is not None:
                obs = self._get_obs()
                opp_action = self.opponent_policy.predict(obs)[0]
            else:
                opp_action = self.action_space.sample()

            return super().step(opp_action)


class BatchGameSimulator:
    """
    Simulate many games to gather statistics and train policies.
    """

    def __init__(
        self,
        home_team: Team,
        away_team: Team,
        config: Optional[GameConfig] = None,
    ):
        self.home_team = home_team
        self.away_team = away_team
        self.config = config or GameConfig()

    def simulate_batch(
        self,
        num_games: int = 100,
        seed_base: int = 42
    ) -> Dict[str, Any]:
        """Simulate multiple games and aggregate statistics."""

        results = {
            "num_games": num_games,
            "home_wins": 0,
            "away_wins": 0,
            "ties": 0,
            "home_total_points": 0,
            "away_total_points": 0,
            "game_results": [],
        }

        for i in range(num_games):
            simulator = GameSimulator(
                self.home_team,
                self.away_team,
                self.config,
                seed=seed_base + i
            )

            game_result = simulator.simulate_game()
            results["game_results"].append(game_result)

            results["home_total_points"] += game_result.home_score
            results["away_total_points"] += game_result.away_score

            if game_result.winner == self.home_team.id:
                results["home_wins"] += 1
            elif game_result.winner == self.away_team.id:
                results["away_wins"] += 1
            else:
                results["ties"] += 1

        # Calculate averages
        results["home_avg_points"] = results["home_total_points"] / num_games
        results["away_avg_points"] = results["away_total_points"] / num_games
        results["home_win_pct"] = results["home_wins"] / num_games * 100
        results["away_win_pct"] = results["away_wins"] / num_games * 100

        return results

    def analyze_play_effectiveness(
        self,
        game_results: List
    ) -> Dict[str, Dict]:
        """Analyze play effectiveness across multiple games."""

        play_stats = {}

        for game in game_results:
            for stats in [game.home_stats, game.away_stats]:
                for play_id, ps in stats.play_stats.items():
                    if play_id not in play_stats:
                        play_stats[play_id] = {
                            "name": ps.play_name,
                            "total_calls": 0,
                            "total_completions": 0,
                            "total_yards": 0.0,
                            "touchdowns": 0,
                            "turnovers": 0,
                            "first_downs": 0,
                            "by_down": {1: {"calls": 0, "success": 0},
                                       2: {"calls": 0, "success": 0},
                                       3: {"calls": 0, "success": 0}},
                            "by_zone": {},
                        }

                    play_stats[play_id]["total_calls"] += ps.times_called
                    play_stats[play_id]["total_completions"] += ps.completions
                    play_stats[play_id]["total_yards"] += ps.total_yards
                    play_stats[play_id]["touchdowns"] += ps.touchdowns
                    play_stats[play_id]["turnovers"] += ps.turnovers
                    play_stats[play_id]["first_downs"] += ps.first_down_conversions

                    for down, count in ps.times_called_by_down.items():
                        play_stats[play_id]["by_down"][down]["calls"] += count
                        play_stats[play_id]["by_down"][down]["success"] += ps.success_by_down.get(down, 0)

                    for zone, count in ps.times_called_by_zone.items():
                        if zone not in play_stats[play_id]["by_zone"]:
                            play_stats[play_id]["by_zone"][zone] = 0
                        play_stats[play_id]["by_zone"][zone] += count

        # Calculate derived stats
        for play_id, stats in play_stats.items():
            if stats["total_calls"] > 0:
                stats["success_rate"] = stats["total_completions"] / stats["total_calls"] * 100
                stats["avg_yards"] = stats["total_yards"] / stats["total_calls"]
            else:
                stats["success_rate"] = 0.0
                stats["avg_yards"] = 0.0

            # Best down
            best_down = None
            best_rate = 0.0
            for down, data in stats["by_down"].items():
                if data["calls"] > 0:
                    rate = data["success"] / data["calls"]
                    if rate > best_rate:
                        best_rate = rate
                        best_down = down
            stats["best_down"] = best_down

        return play_stats
