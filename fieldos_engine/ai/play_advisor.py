"""
Play Advisor - LLM-powered play calling assistant.

Uses trained opponent models to provide intelligent play recommendations
with natural language reasoning.

This can be integrated with any LLM (Claude, GPT, etc.) or used standalone
with rule-based reasoning.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

from ..core.models import Team
from .opponent_model import OpponentModel, OpponentModelTrainer, format_situation_for_llm

logger = logging.getLogger("fieldos_engine.play_advisor")


@dataclass
class PlayRecommendation:
    """A play recommendation with reasoning."""
    play_id: str
    play_name: str
    confidence: float  # 0-100
    reasoning: str
    stats: Dict[str, Any]
    rank: int


@dataclass
class AdvisorResponse:
    """Complete response from the play advisor."""
    situation_summary: str
    top_recommendation: PlayRecommendation
    alternatives: List[PlayRecommendation]
    model_confidence: str  # "HIGH", "MEDIUM", "LOW" based on data availability
    strategic_notes: List[str]


class PlayAdvisor:
    """
    LLM-powered play calling advisor.

    Uses trained opponent models to provide recommendations.
    Can be used with or without an LLM - provides structured data
    that can be passed to an LLM for natural language generation.
    """

    def __init__(
        self,
        my_team: Team,
        models_dir: Path = Path("results/models")
    ):
        self.my_team = my_team
        self.models_dir = models_dir
        self.trainer = OpponentModelTrainer(my_team, models_dir)

        # Cache loaded models
        self._model_cache: Dict[str, OpponentModel] = {}

    def get_model(self, opponent: Team, auto_train: bool = True, min_games: int = 50) -> Optional[OpponentModel]:
        """
        Get or create a trained model for an opponent.

        Args:
            opponent: The opponent team
            auto_train: If True, automatically train if model doesn't exist or is undertrained
            min_games: Minimum games required for the model

        Returns:
            The OpponentModel or None if not available
        """
        # Check cache first
        if opponent.id in self._model_cache:
            model = self._model_cache[opponent.id]
            if model.games_played >= min_games:
                return model

        # Try to load from disk
        model_path = self.trainer.get_model_path(opponent.id)
        if model_path.exists():
            model = OpponentModel.load(model_path)
            self._model_cache[opponent.id] = model

            if model.games_played >= min_games:
                return model
            elif auto_train:
                # Need more training
                games_needed = min_games - model.games_played
                print(f"Model needs more training ({model.games_played}/{min_games} games)")
                print(f"Training {games_needed} more games...")
                model = self.trainer.train(opponent, num_games=games_needed, verbose=True)
                self._model_cache[opponent.id] = model
                return model
        elif auto_train:
            # No model exists - train from scratch
            print(f"No model found for {opponent.name}. Training {min_games} games...")
            model = self.trainer.train(opponent, num_games=min_games, verbose=True)
            self._model_cache[opponent.id] = model
            return model

        return None

    def _generate_reasoning(
        self,
        play_name: str,
        stats: Dict,
        situation: Dict,
        rank: int
    ) -> str:
        """Generate human-readable reasoning for a play recommendation."""
        reasons = []

        # Success rate reasoning
        if stats["success_rate"] >= 70:
            reasons.append(f"very high success rate ({stats['success_rate']:.0f}%) against this opponent")
        elif stats["success_rate"] >= 55:
            reasons.append(f"solid success rate ({stats['success_rate']:.0f}%)")
        elif stats["success_rate"] < 40:
            reasons.append(f"lower success rate ({stats['success_rate']:.0f}%) but may be worth the risk")

        # Yards reasoning
        if stats["avg_yards"] >= situation.get("yards_needed", 5):
            reasons.append(f"averages {stats['avg_yards']:.1f} yards (enough for the first down)")
        elif stats["avg_yards"] >= 5:
            reasons.append(f"gets good yardage ({stats['avg_yards']:.1f} avg)")

        # Touchdown potential
        if stats["touchdowns"] > 0:
            reasons.append(f"has scored {stats['touchdowns']} TD(s) in this situation")

        # Turnover risk
        if stats["turnover_rate"] > 10:
            reasons.append(f"BUT has {stats['turnover_rate']:.0f}% turnover risk")
        elif stats["turnover_rate"] == 0:
            reasons.append("no turnovers in this situation")

        # Data confidence
        if stats["attempts"] >= 15:
            reasons.append(f"high confidence ({stats['attempts']} data points)")
        elif stats["attempts"] >= 8:
            reasons.append(f"moderate confidence ({stats['attempts']} data points)")
        else:
            reasons.append(f"limited data ({stats['attempts']} attempts)")

        return "; ".join(reasons).capitalize()

    def _generate_strategic_notes(
        self,
        model: OpponentModel,
        down: int,
        yards_to_go: float,
        field_position: float,
        score_diff: int,
        best_plays: List
    ) -> List[str]:
        """Generate strategic notes about the situation."""
        notes = []

        # Down-specific notes
        if down == 3:
            if yards_to_go <= 3:
                notes.append("Short yardage 3rd down - high percentage plays recommended")
            elif yards_to_go >= 8:
                notes.append("Long 3rd down - may need to take a risk for chunk yards")

        # Score-specific notes
        if score_diff <= -14:
            notes.append("Down big - consider aggressive/explosive plays")
        elif score_diff >= 14:
            notes.append("Big lead - play conservatively, avoid turnovers")

        # Field position notes
        zone = model._get_zone_bucket(field_position)
        if zone == "redzone":
            notes.append("Redzone - condensed field, quick routes work well")
        elif zone == "goalline":
            notes.append("Goalline - very condensed, consider fade or quick slant")

        # Model confidence notes
        summary = model.get_model_summary()
        if summary["games_played"] >= 100:
            notes.append(f"High confidence model ({summary['games_played']} games trained)")
        elif summary["games_played"] >= 50:
            notes.append(f"Good model confidence ({summary['games_played']} games)")
        else:
            notes.append(f"Model still learning ({summary['games_played']} games) - recommendations may improve with more training")

        return notes

    def recommend(
        self,
        opponent: Team,
        down: int,
        yards_to_go: float,
        field_position: float = 23.0,  # Default to midfield
        score_diff: int = 0,
        auto_train: bool = True
    ) -> Optional[AdvisorResponse]:
        """
        Get play recommendation for a specific situation.

        Args:
            opponent: The opponent team
            down: Current down (1, 2, or 3)
            yards_to_go: Yards needed for first down/touchdown
            field_position: Yards from own goal (0-46)
            score_diff: Point differential (positive = winning)
            auto_train: Automatically train model if needed

        Returns:
            AdvisorResponse with recommendations, or None if no model
        """
        # Get the trained model
        model = self.get_model(opponent, auto_train=auto_train)
        if model is None:
            logger.warning(f"No model available for {opponent.name}")
            return None

        # Get best plays
        best_plays = model.get_best_plays_for_situation(
            down, yards_to_go, field_position, score_diff, min_attempts=2
        )

        if not best_plays:
            # Fall back to overall stats
            logger.info("No situation-specific data, using overall play stats")
            best_plays = []
            for play_id, data in model.overall_play_stats.items():
                stats = data["stats"]
                if stats["attempts"] >= 2:
                    best_plays.append((play_id, data["play_name"], stats, "overall"))
            best_plays.sort(key=lambda x: (x[2]["success_rate"], x[2]["avg_yards"]), reverse=True)

        if not best_plays:
            logger.warning("No play data available")
            return None

        # Build situation info for reasoning
        situation = {
            "down": down,
            "yards_needed": yards_to_go,
            "field_position": field_position,
            "score_diff": score_diff,
        }

        # Create recommendations
        recommendations = []
        for i, (play_id, play_name, stats, match_type) in enumerate(best_plays[:5]):
            confidence = min(100, stats["success_rate"] + (10 if stats["attempts"] >= 10 else 0))
            reasoning = self._generate_reasoning(play_name, stats, situation, i + 1)

            recommendations.append(PlayRecommendation(
                play_id=play_id,
                play_name=play_name,
                confidence=confidence,
                reasoning=reasoning,
                stats=stats,
                rank=i + 1
            ))

        # Build situation summary
        down_str = f"{down}{'st' if down == 1 else 'nd' if down == 2 else 'rd'}"
        zone = model._get_zone_bucket(field_position).replace("_", " ")

        if score_diff > 0:
            score_str = f"up by {score_diff}"
        elif score_diff < 0:
            score_str = f"down by {abs(score_diff)}"
        else:
            score_str = "tied"

        situation_summary = f"{down_str} and {yards_to_go:.0f} in {zone}, {score_str} vs {opponent.name}"

        # Determine model confidence
        summary = model.get_model_summary()
        if summary["games_played"] >= 100 and best_plays[0][2]["attempts"] >= 10:
            model_confidence = "HIGH"
        elif summary["games_played"] >= 50 and best_plays[0][2]["attempts"] >= 5:
            model_confidence = "MEDIUM"
        else:
            model_confidence = "LOW"

        # Generate strategic notes
        strategic_notes = self._generate_strategic_notes(
            model, down, yards_to_go, field_position, score_diff, best_plays
        )

        return AdvisorResponse(
            situation_summary=situation_summary,
            top_recommendation=recommendations[0],
            alternatives=recommendations[1:],
            model_confidence=model_confidence,
            strategic_notes=strategic_notes
        )

    def format_response_text(self, response: AdvisorResponse) -> str:
        """Format an AdvisorResponse as readable text."""
        lines = [
            "=" * 60,
            "PLAY ADVISOR RECOMMENDATION",
            "=" * 60,
            "",
            f"SITUATION: {response.situation_summary}",
            f"MODEL CONFIDENCE: {response.model_confidence}",
            "",
            "-" * 40,
            "TOP RECOMMENDATION",
            "-" * 40,
            f"",
            f">>> {response.top_recommendation.play_name.upper()} <<<",
            f"",
            f"Why: {response.top_recommendation.reasoning}",
            f"",
            f"Stats vs this opponent:",
            f"  - Success Rate: {response.top_recommendation.stats['success_rate']:.0f}%",
            f"  - Avg Yards: {response.top_recommendation.stats['avg_yards']:.1f}",
            f"  - Touchdowns: {response.top_recommendation.stats['touchdowns']}",
            f"  - Turnovers: {response.top_recommendation.stats['turnovers']}",
            "",
        ]

        if response.alternatives:
            lines.extend([
                "-" * 40,
                "ALTERNATIVES",
                "-" * 40,
            ])
            for alt in response.alternatives[:3]:
                lines.extend([
                    f"",
                    f"{alt.rank}. {alt.play_name}",
                    f"   {alt.reasoning}",
                    f"   Success: {alt.stats['success_rate']:.0f}% | Yards: {alt.stats['avg_yards']:.1f}",
                ])

        if response.strategic_notes:
            lines.extend([
                "",
                "-" * 40,
                "STRATEGIC NOTES",
                "-" * 40,
            ])
            for note in response.strategic_notes:
                lines.append(f"• {note}")

        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    def get_llm_prompt(
        self,
        opponent: Team,
        down: int,
        yards_to_go: float,
        field_position: float = 23.0,
        score_diff: int = 0,
        user_question: Optional[str] = None
    ) -> str:
        """
        Generate a prompt for an LLM to answer play calling questions.

        This includes all the context from the trained model so the LLM
        can provide intelligent, data-driven recommendations.
        """
        model = self.get_model(opponent, auto_train=False)
        if model is None:
            return f"No trained model available for {opponent.name}. Please train the model first."

        # Get the formatted situation data
        situation_data = format_situation_for_llm(
            model, down, yards_to_go, field_position, score_diff
        )

        # Build the LLM prompt
        prompt = f"""You are a flag football play calling assistant. You have access to a trained model
that has learned what plays work best against a specific opponent through {model.games_played} simulated games.

Use the data below to recommend the best play for the current situation.
Provide your recommendation with clear reasoning based on the statistics.

{situation_data}

"""
        if user_question:
            prompt += f"""
USER'S QUESTION: {user_question}

Based on the learned data above, provide a recommendation that directly answers their question.
"""
        else:
            prompt += """
Provide your TOP play recommendation with reasoning, then list 2-3 alternatives.
Explain WHY you're recommending each play based on the statistical data.
"""

        return prompt


def create_advisor_cli():
    """Create an interactive CLI for the play advisor."""
    print("""
╔══════════════════════════════════════════════════════════════╗
║              FIELDOS PLAY ADVISOR                            ║
║                                                               ║
║  An AI-powered play calling assistant that learns what       ║
║  works against each opponent through game simulation.        ║
╚══════════════════════════════════════════════════════════════╝
    """)
