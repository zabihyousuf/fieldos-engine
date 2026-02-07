"""
LangChain-powered play calling chatbot.

Provides a conversational interface to the FieldOS engine.
Users can ask questions like "it's 3rd and 5, what should I run?"
and get data-driven recommendations from trained opponent models.
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_anthropic import ChatAnthropic

from ..core.models import Team, GamePlayer, DualRolePlayerAttributes, PlayerSpecialty, GameConfig
from .opponent_model import OpponentModel, OpponentModelTrainer, format_situation_for_llm
from .play_advisor import PlayAdvisor

logger = logging.getLogger("fieldos_engine.chat")

MODELS_DIR = Path(__file__).parent.parent.parent / "results" / "models"


class PlayCallingChat:
    """
    LangChain-powered chatbot for play calling recommendations.

    Wraps the PlayAdvisor with natural language understanding so users
    can just talk to it: "it's 3rd and 5 what should I run against the godbods?"
    """

    def __init__(self, my_team: Team, opponents: Dict[str, Team], models_dir: Path = MODELS_DIR):
        self.my_team = my_team
        self.opponents = opponents  # opponent_id -> Team
        self.models_dir = models_dir
        self.advisor = PlayAdvisor(my_team, models_dir)
        self.conversation_history: List = []

        # Default opponent (first one or None)
        self.current_opponent_id: Optional[str] = next(iter(opponents), None)

    def _get_opponent(self, opponent_id: Optional[str] = None) -> Optional[Team]:
        oid = opponent_id or self.current_opponent_id
        if oid and oid in self.opponents:
            return self.opponents[oid]
        return None

    def _build_system_prompt(self) -> str:
        opponent = self._get_opponent()
        opponent_name = opponent.name if opponent else "unknown"

        # Load model summary if available
        model_info = ""
        if opponent:
            model_path = self.models_dir / f"opponent_{opponent.id}.json"
            if model_path.exists():
                model = OpponentModel.load(model_path)
                summary = model.get_model_summary()
                model_info = f"""
You have a trained model for {opponent_name} based on {summary['games_played']} simulated games.
Historical record: {summary['record']} ({summary['win_rate']:.1f}% win rate)
Avg Points For: {summary['avg_points_for']:.1f} | Avg Points Against: {summary['avg_points_against']:.1f}
Plays in model: {summary['plays_in_model']}
"""
                # Add top plays
                plays = []
                for play_id, data in model.overall_play_stats.items():
                    stats = data["stats"]
                    if stats["attempts"] >= 5:
                        plays.append((data["play_name"], stats))
                plays.sort(key=lambda x: (x[1]["success_rate"], x[1]["avg_yards"]), reverse=True)
                if plays:
                    model_info += "\nOverall play effectiveness vs this opponent:\n"
                    for play_name, stats in plays:
                        model_info += (f"  - {play_name}: {stats['success_rate']:.0f}% success, "
                                       f"{stats['avg_yards']:.1f} avg yds, "
                                       f"{stats['touchdowns']} TDs, {stats['turnovers']} TOs "
                                       f"({stats['attempts']} attempts)\n")
            else:
                model_info = f"\nNo trained model exists yet for {opponent_name}. The user needs to train one first.\n"

        # Build roster info
        roster_info = f"\nYour team ({self.my_team.name}) roster:\n"
        for p in self.my_team.players:
            a = p.attributes
            spec = a.specialty.value if a else "TWO_WAY"
            roster_info += (f"  - {p.name} #{p.number} | {a.height_formatted()} {a.weight_lbs}lbs | "
                           f"Speed:{a.speed} | Specialty: {spec}\n")

        playbook_info = f"\nYour playbook: {', '.join(self.my_team.playbook)}\n"

        return f"""You are an AI flag football play calling assistant for the {self.my_team.name} team.
Your coach is Zabih Yousuf. You help him decide what plays to run in specific game situations.

You have access to a trained simulation model that has learned what plays work best
against specific opponents through hundreds of simulated games.

Current opponent: {opponent_name}
{model_info}
{roster_info}
{playbook_info}

When the user describes a game situation (like "3rd and 5" or "we're in the red zone down by 7"),
you will be given the statistical analysis from the trained model. Use that data to provide
your recommendation with clear reasoning.

IMPORTANT RULES:
- Always recommend specific plays from the playbook with reasoning based on the data
- Explain WHY a play is good in that situation using the stats
- If asked about a specific player, reference their actual attributes
- Be conversational but concise - this is sideline advice during a game
- If no model is trained, tell the user to train one first
- You can discuss strategy, matchups, and player strengths

When you receive tool results with play data, use those statistics to back up your recommendations.
Do NOT make up statistics - only use what the model provides."""

    def _get_situation_context(self, down: int, yards: float, field_pos: float,
                                score_diff: int, opponent: Team) -> str:
        """Get the statistical context from the trained model for a situation."""
        model_path = self.models_dir / f"opponent_{opponent.id}.json"
        if not model_path.exists():
            return f"No trained model for {opponent.name}. Train one first with the /train endpoint."

        model = OpponentModel.load(model_path)
        return format_situation_for_llm(model, down, yards, field_pos, score_diff)

    def chat(self, user_message: str, opponent_id: Optional[str] = None) -> str:
        """
        Process a chat message and return the AI response.

        The message is analyzed to extract game situation context,
        then the trained model data is injected into the LLM prompt.
        """
        if opponent_id:
            self.current_opponent_id = opponent_id

        opponent = self._get_opponent()
        if not opponent:
            return "No opponent set. Please specify an opponent team."

        # Try to extract situation from the message
        situation = self._parse_situation(user_message)

        # Build messages
        system_prompt = self._build_system_prompt()

        # If we detected a situation, inject the model data
        context_note = ""
        if situation:
            context_data = self._get_situation_context(
                situation["down"], situation["yards"],
                situation["field_position"], situation["score_diff"],
                opponent
            )
            context_note = f"\n\n[TRAINED MODEL DATA FOR THIS SITUATION]\n{context_data}\n"

        messages = [SystemMessage(content=system_prompt)]

        # Add conversation history
        for msg in self.conversation_history[-10:]:  # Keep last 10 exchanges
            messages.append(msg)

        # Add current message with context
        full_user_msg = user_message
        if context_note:
            full_user_msg += context_note

        messages.append(HumanMessage(content=full_user_msg))

        # Call the LLM
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            # Fall back to rule-based response
            return self._fallback_response(user_message, situation, opponent)

        try:
            llm = ChatAnthropic(
                model="claude-sonnet-4-20250514",
                api_key=key,
                max_tokens=1024,
            )
            response = llm.invoke(messages)
            ai_message = response.content

            # Save to history (without the injected context)
            self.conversation_history.append(HumanMessage(content=user_message))
            self.conversation_history.append(AIMessage(content=ai_message))

            return ai_message

        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return self._fallback_response(user_message, situation, opponent)

    def _parse_situation(self, message: str) -> Optional[Dict[str, Any]]:
        """Parse a game situation from natural language."""
        import re
        msg = message.lower()

        # Check if this looks like a situation query
        has_down = any(x in msg for x in ["1st", "2nd", "3rd", "first", "second", "third",
                                           "1st and", "2nd and", "3rd and"])
        has_yards = bool(re.search(r'and\s+\d+', msg))

        if not (has_down or has_yards):
            return None

        down = 1
        yards = 10.0
        field_position = 23.0
        score_diff = 0

        # Parse down
        if "1st" in msg or "first" in msg:
            down = 1
        elif "2nd" in msg or "second" in msg:
            down = 2
        elif "3rd" in msg or "third" in msg:
            down = 3

        # Parse yards
        yard_match = re.search(r'and\s+(\d+)', msg)
        if yard_match:
            yards = float(yard_match.group(1))

        # Parse zone
        if "redzone" in msg or "red zone" in msg:
            field_position = 38.0
        elif "goal" in msg or "goalline" in msg:
            field_position = 43.0
        elif "own" in msg:
            field_position = 10.0
        elif "opponent" in msg or "their" in msg:
            field_position = 30.0
        elif "midfield" in msg:
            field_position = 23.0

        # Parse score
        if "down by" in msg:
            score_match = re.search(r'down by\s+(\d+)', msg)
            if score_match:
                score_diff = -int(score_match.group(1))
        elif "up by" in msg:
            score_match = re.search(r'up by\s+(\d+)', msg)
            if score_match:
                score_diff = int(score_match.group(1))

        return {
            "down": down,
            "yards": yards,
            "field_position": field_position,
            "score_diff": score_diff,
        }

    def _fallback_response(self, message: str, situation: Optional[Dict],
                            opponent: Team) -> str:
        """Generate a response without an LLM using the PlayAdvisor directly."""
        if not situation:
            return ("I can help you with play calling! Tell me the situation like:\n"
                    "- \"3rd and 5\"\n"
                    "- \"2nd and 8 in the redzone\"\n"
                    "- \"1st and 10 down by 7\"\n\n"
                    f"I have data trained against {opponent.name}.")

        # Use the advisor
        response = self.advisor.recommend(
            opponent=opponent,
            down=situation["down"],
            yards_to_go=situation["yards"],
            field_position=situation["field_position"],
            score_diff=situation["score_diff"],
            auto_train=False
        )

        if response:
            return self.advisor.format_response_text(response)
        else:
            return (f"No trained model available for {opponent.name}. "
                    f"Train one first: POST /api/advisor/train/{opponent.id}")

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
