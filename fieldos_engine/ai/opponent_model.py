"""
Opponent Model - Learns opponent-specific play effectiveness through simulation.

This module:
1. Runs many games against a specific opponent
2. Tracks what plays work in what situations against THAT opponent
3. Builds a statistical model of play effectiveness
4. Saves/loads models for persistence
5. Provides an interface for LLM-powered play recommendations
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
import numpy as np

from ..core.models import (
    Team, GameResult, PlayResult, DriveResult, OutcomeType, GameConfig
)
from ..sim.game_simulator import GameSimulator

logger = logging.getLogger("fieldos_engine.opponent_model")


@dataclass
class PlaySituationStats:
    """Statistics for a play in a specific situation."""
    attempts: int = 0
    completions: int = 0
    total_yards: float = 0.0
    touchdowns: int = 0
    first_downs: int = 0
    turnovers: int = 0

    @property
    def success_rate(self) -> float:
        return (self.completions / self.attempts * 100) if self.attempts > 0 else 0.0

    @property
    def avg_yards(self) -> float:
        return (self.total_yards / self.attempts) if self.attempts > 0 else 0.0

    @property
    def turnover_rate(self) -> float:
        return (self.turnovers / self.attempts * 100) if self.attempts > 0 else 0.0

    def to_dict(self) -> Dict:
        return {
            "attempts": self.attempts,
            "completions": self.completions,
            "total_yards": self.total_yards,
            "touchdowns": self.touchdowns,
            "first_downs": self.first_downs,
            "turnovers": self.turnovers,
            "success_rate": self.success_rate,
            "avg_yards": self.avg_yards,
            "turnover_rate": self.turnover_rate
        }


@dataclass
class OpponentModel:
    """
    Learned model for a specific opponent.

    Tracks play effectiveness by:
    - Down (1st, 2nd, 3rd)
    - Distance bucket (short: 1-3, medium: 4-7, long: 8+)
    - Field zone (own territory, midfield, redzone, goalline)
    - Score differential bucket (losing big, losing, tied, winning, winning big)
    """
    opponent_id: str
    opponent_name: str
    my_team_id: str
    my_team_name: str

    # Training metadata
    games_played: int = 0
    total_plays: int = 0
    wins: int = 0
    losses: int = 0
    ties: int = 0
    total_points_for: int = 0
    total_points_against: int = 0
    last_trained: Optional[str] = None

    # Play effectiveness by situation
    # Structure: play_id -> situation_key -> PlaySituationStats
    # situation_key format: "down_{d}_dist_{bucket}_zone_{zone}_score_{bucket}"
    play_stats: Dict[str, Dict[str, Dict]] = field(default_factory=dict)

    # Overall play stats (not situation-specific)
    overall_play_stats: Dict[str, Dict] = field(default_factory=dict)

    # Opponent defensive tendencies learned
    opponent_tendencies: Dict[str, Any] = field(default_factory=dict)

    def _get_distance_bucket(self, yards: float) -> str:
        if yards <= 3:
            return "short"
        elif yards <= 7:
            return "medium"
        else:
            return "long"

    def _get_zone_bucket(self, field_position: float, playing_field: float = 46.0) -> str:
        yards_to_goal = playing_field - field_position
        if yards_to_goal <= 5:
            return "goalline"
        elif yards_to_goal <= 15:
            return "redzone"
        elif field_position >= playing_field / 2:
            return "opponent_territory"
        elif field_position >= playing_field / 4:
            return "midfield"
        else:
            return "own_territory"

    def _get_score_bucket(self, score_diff: int) -> str:
        if score_diff <= -14:
            return "losing_big"
        elif score_diff < 0:
            return "losing"
        elif score_diff == 0:
            return "tied"
        elif score_diff < 14:
            return "winning"
        else:
            return "winning_big"

    def _make_situation_key(
        self,
        down: int,
        yards_to_go: float,
        field_position: float,
        score_diff: int
    ) -> str:
        dist = self._get_distance_bucket(yards_to_go)
        zone = self._get_zone_bucket(field_position)
        score = self._get_score_bucket(score_diff)
        return f"down_{down}_dist_{dist}_zone_{zone}_score_{score}"

    def record_play(
        self,
        play_result: PlayResult,
        score_diff: int
    ):
        """Record a play result to update the model."""
        play_id = play_result.play_id

        # Initialize play stats if needed
        if play_id not in self.play_stats:
            self.play_stats[play_id] = {}
        if play_id not in self.overall_play_stats:
            self.overall_play_stats[play_id] = {
                "play_name": play_result.play_name,
                "stats": PlaySituationStats().to_dict()
            }

        # Get situation key
        situation_key = self._make_situation_key(
            play_result.down,
            play_result.yards_to_first,
            play_result.field_position_before,
            score_diff
        )

        # Initialize situation stats if needed
        if situation_key not in self.play_stats[play_id]:
            self.play_stats[play_id][situation_key] = PlaySituationStats().to_dict()

        # Update situation-specific stats
        stats = self.play_stats[play_id][situation_key]
        stats["attempts"] += 1
        if play_result.outcome == OutcomeType.COMPLETE:
            stats["completions"] += 1
        stats["total_yards"] += play_result.yards_gained
        if play_result.resulted_in_touchdown:
            stats["touchdowns"] += 1
        if play_result.resulted_in_first_down:
            stats["first_downs"] += 1
        if play_result.resulted_in_turnover:
            stats["turnovers"] += 1

        # Recalculate derived stats
        stats["success_rate"] = (stats["completions"] / stats["attempts"] * 100) if stats["attempts"] > 0 else 0.0
        stats["avg_yards"] = (stats["total_yards"] / stats["attempts"]) if stats["attempts"] > 0 else 0.0
        stats["turnover_rate"] = (stats["turnovers"] / stats["attempts"] * 100) if stats["attempts"] > 0 else 0.0

        # Update overall stats
        overall = self.overall_play_stats[play_id]["stats"]
        overall["attempts"] += 1
        if play_result.outcome == OutcomeType.COMPLETE:
            overall["completions"] += 1
        overall["total_yards"] += play_result.yards_gained
        if play_result.resulted_in_touchdown:
            overall["touchdowns"] += 1
        if play_result.resulted_in_first_down:
            overall["first_downs"] += 1
        if play_result.resulted_in_turnover:
            overall["turnovers"] += 1
        overall["success_rate"] = (overall["completions"] / overall["attempts"] * 100) if overall["attempts"] > 0 else 0.0
        overall["avg_yards"] = (overall["total_yards"] / overall["attempts"]) if overall["attempts"] > 0 else 0.0
        overall["turnover_rate"] = (overall["turnovers"] / overall["attempts"] * 100) if overall["attempts"] > 0 else 0.0

        self.total_plays += 1

    def record_game(self, game_result: GameResult, my_team_is_home: bool):
        """Record a complete game result."""
        self.games_played += 1

        if my_team_is_home:
            my_score = game_result.home_score
            their_score = game_result.away_score
        else:
            my_score = game_result.away_score
            their_score = game_result.home_score

        self.total_points_for += my_score
        self.total_points_against += their_score

        if game_result.winner == self.my_team_id:
            self.wins += 1
        elif game_result.winner == self.opponent_id:
            self.losses += 1
        else:
            self.ties += 1

        # Process each drive for my team
        for drive in game_result.drive_records:
            if (my_team_is_home and drive.team_id == game_result.home_team_id) or \
               (not my_team_is_home and drive.team_id == game_result.away_team_id):
                # This is my team's drive
                running_score_diff = 0  # Would need to track properly in real impl
                for play in drive.plays:
                    self.record_play(play, running_score_diff)

        self.last_trained = datetime.now().isoformat()

    def get_best_plays_for_situation(
        self,
        down: int,
        yards_to_go: float,
        field_position: float,
        score_diff: int,
        min_attempts: int = 3
    ) -> List[Tuple[str, str, Dict]]:
        """
        Get the best plays for a specific situation based on learned data.

        Returns list of (play_id, play_name, stats) tuples sorted by success rate.
        """
        situation_key = self._make_situation_key(down, yards_to_go, field_position, score_diff)

        plays_for_situation = []

        for play_id, situations in self.play_stats.items():
            # First try exact situation match
            if situation_key in situations:
                stats = situations[situation_key]
                if stats["attempts"] >= min_attempts:
                    play_name = self.overall_play_stats.get(play_id, {}).get("play_name", play_id)
                    plays_for_situation.append((play_id, play_name, stats, "exact"))

            # If not enough data, try partial matches
            if not any(p[0] == play_id for p in plays_for_situation):
                # Try matching just down and distance
                partial_key = f"down_{down}_dist_{self._get_distance_bucket(yards_to_go)}"
                matching_stats = []
                for sit_key, stats in situations.items():
                    if sit_key.startswith(partial_key) and stats["attempts"] >= 1:
                        matching_stats.append(stats)

                if matching_stats:
                    # Aggregate partial matches
                    agg_stats = {
                        "attempts": sum(s["attempts"] for s in matching_stats),
                        "completions": sum(s["completions"] for s in matching_stats),
                        "total_yards": sum(s["total_yards"] for s in matching_stats),
                        "touchdowns": sum(s["touchdowns"] for s in matching_stats),
                        "first_downs": sum(s["first_downs"] for s in matching_stats),
                        "turnovers": sum(s["turnovers"] for s in matching_stats),
                    }
                    agg_stats["success_rate"] = (agg_stats["completions"] / agg_stats["attempts"] * 100) if agg_stats["attempts"] > 0 else 0
                    agg_stats["avg_yards"] = (agg_stats["total_yards"] / agg_stats["attempts"]) if agg_stats["attempts"] > 0 else 0
                    agg_stats["turnover_rate"] = (agg_stats["turnovers"] / agg_stats["attempts"] * 100) if agg_stats["attempts"] > 0 else 0

                    if agg_stats["attempts"] >= min_attempts:
                        play_name = self.overall_play_stats.get(play_id, {}).get("play_name", play_id)
                        plays_for_situation.append((play_id, play_name, agg_stats, "partial"))

        # Sort by success rate, then by avg yards
        plays_for_situation.sort(key=lambda x: (x[2]["success_rate"], x[2]["avg_yards"]), reverse=True)

        return plays_for_situation

    def get_play_summary(self, play_id: str) -> Optional[Dict]:
        """Get overall summary for a specific play against this opponent."""
        if play_id in self.overall_play_stats:
            return self.overall_play_stats[play_id]
        return None

    def get_model_summary(self) -> Dict:
        """Get a summary of the learned model."""
        return {
            "opponent": self.opponent_name,
            "my_team": self.my_team_name,
            "games_played": self.games_played,
            "record": f"{self.wins}-{self.losses}-{self.ties}",
            "win_rate": (self.wins / self.games_played * 100) if self.games_played > 0 else 0,
            "avg_points_for": (self.total_points_for / self.games_played) if self.games_played > 0 else 0,
            "avg_points_against": (self.total_points_against / self.games_played) if self.games_played > 0 else 0,
            "total_plays_recorded": self.total_plays,
            "plays_in_model": len(self.play_stats),
            "last_trained": self.last_trained,
        }

    def to_dict(self) -> Dict:
        """Convert model to dictionary for serialization."""
        return {
            "opponent_id": self.opponent_id,
            "opponent_name": self.opponent_name,
            "my_team_id": self.my_team_id,
            "my_team_name": self.my_team_name,
            "games_played": self.games_played,
            "total_plays": self.total_plays,
            "wins": self.wins,
            "losses": self.losses,
            "ties": self.ties,
            "total_points_for": self.total_points_for,
            "total_points_against": self.total_points_against,
            "last_trained": self.last_trained,
            "play_stats": self.play_stats,
            "overall_play_stats": self.overall_play_stats,
            "opponent_tendencies": self.opponent_tendencies,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "OpponentModel":
        """Create model from dictionary."""
        model = cls(
            opponent_id=data["opponent_id"],
            opponent_name=data["opponent_name"],
            my_team_id=data["my_team_id"],
            my_team_name=data["my_team_name"],
        )
        model.games_played = data.get("games_played", 0)
        model.total_plays = data.get("total_plays", 0)
        model.wins = data.get("wins", 0)
        model.losses = data.get("losses", 0)
        model.ties = data.get("ties", 0)
        model.total_points_for = data.get("total_points_for", 0)
        model.total_points_against = data.get("total_points_against", 0)
        model.last_trained = data.get("last_trained")
        model.play_stats = data.get("play_stats", {})
        model.overall_play_stats = data.get("overall_play_stats", {})
        model.opponent_tendencies = data.get("opponent_tendencies", {})
        return model

    def save(self, filepath: Path):
        """Save model to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Saved opponent model to {filepath}")

    @classmethod
    def load(cls, filepath: Path) -> "OpponentModel":
        """Load model from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        logger.info(f"Loaded opponent model from {filepath}")
        return cls.from_dict(data)


class OpponentModelTrainer:
    """Trains opponent models through game simulation."""

    def __init__(
        self,
        my_team: Team,
        models_dir: Path = Path("results/models")
    ):
        self.my_team = my_team
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def get_model_path(self, opponent_id: str) -> Path:
        """Get the file path for an opponent model."""
        return self.models_dir / f"opponent_{opponent_id}.json"

    def load_or_create_model(self, opponent: Team) -> OpponentModel:
        """Load existing model or create new one."""
        model_path = self.get_model_path(opponent.id)

        if model_path.exists():
            model = OpponentModel.load(model_path)
            logger.info(f"Loaded existing model for {opponent.name} ({model.games_played} games)")
        else:
            model = OpponentModel(
                opponent_id=opponent.id,
                opponent_name=opponent.name,
                my_team_id=self.my_team.id,
                my_team_name=self.my_team.name,
            )
            logger.info(f"Created new model for {opponent.name}")

        return model

    def train(
        self,
        opponent: Team,
        num_games: int = 100,
        config: Optional[GameConfig] = None,
        save_interval: int = 10,
        verbose: bool = True
    ) -> OpponentModel:
        """
        Train the model by simulating games against the opponent.

        Args:
            opponent: The opponent team to train against
            num_games: Number of games to simulate
            config: Game configuration
            save_interval: Save model every N games
            verbose: Print progress

        Returns:
            The trained OpponentModel
        """
        model = self.load_or_create_model(opponent)
        config = config or GameConfig()

        if verbose:
            print(f"\n{'='*60}")
            print(f"TRAINING MODEL: {self.my_team.name} vs {opponent.name}")
            print(f"{'='*60}")
            print(f"Starting from: {model.games_played} games")
            print(f"Training for: {num_games} additional games")
            print()

        for i in range(num_games):
            # Alternate home/away
            if i % 2 == 0:
                simulator = GameSimulator(self.my_team, opponent, config, seed=42 + model.games_played + i)
                my_team_is_home = True
            else:
                simulator = GameSimulator(opponent, self.my_team, config, seed=42 + model.games_played + i)
                my_team_is_home = False

            result = simulator.simulate_game()
            model.record_game(result, my_team_is_home)

            if verbose and (i + 1) % 10 == 0:
                summary = model.get_model_summary()
                print(f"  Game {i + 1}/{num_games}: Record {summary['record']}, "
                      f"Win Rate: {summary['win_rate']:.1f}%")

            # Save periodically
            if (i + 1) % save_interval == 0:
                model.save(self.get_model_path(opponent.id))

        # Final save
        model.save(self.get_model_path(opponent.id))

        if verbose:
            print()
            print(f"Training complete!")
            summary = model.get_model_summary()
            print(f"  Total games: {summary['games_played']}")
            print(f"  Record: {summary['record']} ({summary['win_rate']:.1f}%)")
            print(f"  Avg Points For: {summary['avg_points_for']:.1f}")
            print(f"  Avg Points Against: {summary['avg_points_against']:.1f}")
            print(f"  Plays learned: {summary['plays_in_model']}")
            print(f"  Model saved to: {self.get_model_path(opponent.id)}")

        return model


def format_situation_for_llm(
    model: OpponentModel,
    down: int,
    yards_to_go: float,
    field_position: float,
    score_diff: int
) -> str:
    """
    Format situation and recommendations for LLM consumption.

    This creates a structured prompt that an LLM can use to provide
    play recommendations with reasoning.
    """
    best_plays = model.get_best_plays_for_situation(
        down, yards_to_go, field_position, score_diff, min_attempts=2
    )

    # Build situation description
    down_str = f"{down}{'st' if down == 1 else 'nd' if down == 2 else 'rd'}"
    if yards_to_go <= 3:
        distance_str = "short"
    elif yards_to_go <= 7:
        distance_str = "medium"
    else:
        distance_str = "long"

    zone = model._get_zone_bucket(field_position)
    zone_str = zone.replace("_", " ")

    if score_diff > 0:
        score_str = f"winning by {score_diff}"
    elif score_diff < 0:
        score_str = f"losing by {abs(score_diff)}"
    else:
        score_str = "tied"

    summary = model.get_model_summary()

    # Build the context
    lines = [
        "=" * 60,
        f"PLAY RECOMMENDATION REQUEST",
        "=" * 60,
        "",
        f"OPPONENT: {model.opponent_name}",
        f"MODEL TRAINED ON: {summary['games_played']} games",
        f"HISTORICAL RECORD: {summary['record']} ({summary['win_rate']:.1f}% win rate)",
        "",
        f"CURRENT SITUATION:",
        f"  - {down_str} and {distance_str} ({yards_to_go:.0f} yards to go)",
        f"  - Field Position: {zone_str}",
        f"  - Score: {score_str}",
        "",
        "LEARNED PLAY EFFECTIVENESS (from simulations against this opponent):",
        "-" * 40,
    ]

    if best_plays:
        for i, (play_id, play_name, stats, match_type) in enumerate(best_plays[:5], 1):
            confidence = "HIGH" if stats["attempts"] >= 10 else "MEDIUM" if stats["attempts"] >= 5 else "LOW"
            match_note = "" if match_type == "exact" else " (similar situations)"
            lines.extend([
                f"",
                f"{i}. {play_name}{match_note}",
                f"   Success Rate: {stats['success_rate']:.1f}% ({stats['completions']}/{stats['attempts']})",
                f"   Avg Yards: {stats['avg_yards']:.1f}",
                f"   Touchdowns: {stats['touchdowns']} | First Downs: {stats['first_downs']}",
                f"   Turnover Risk: {stats['turnover_rate']:.1f}%",
                f"   Confidence: {confidence} ({stats['attempts']} data points)",
            ])
    else:
        lines.append("No play data available for this situation. Need more training.")

    lines.extend([
        "",
        "=" * 60,
    ])

    return "\n".join(lines)
